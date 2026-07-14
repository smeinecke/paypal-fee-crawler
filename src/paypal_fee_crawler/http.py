"""Safe, deterministic HTTP client for the PayPal fee crawler."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
from lxml import html

from .exceptions import (
    AccessChallengeError,
    ContentSecurityError,
    NetworkError,
    PermanentHttpError,
    PermanentNetworkError,
    RateLimitError,
    TransientNetworkError,
)
from .models import CrawlConfiguration

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "paypal-fee-crawler/0.1.0 (+https://github.com/smeinecke/paypal-fee-crawler)"


def _sanitize_url(url: str) -> str:
    """Return a URL safe for logging by stripping volatile query parameters."""
    parsed = urlparse(url)
    # Strip common volatile or sensitive query parameters
    sensitive = {
        "token",
        "session",
        "sid",
        "auth",
        "nonce",
        "csrf",
        "request_id",
        "correlation_id",
        "visitor_id",
    }
    if not parsed.query:
        return url
    pairs = re.split(r"[;&]", parsed.query)
    kept = []
    for pair in pairs:
        if not pair:
            continue
        if "=" in pair:
            key = pair.split("=", 1)[0]
            if key.lower() in sensitive:
                kept.append(f"{key}=<redacted>")
            else:
                kept.append(pair)
        else:
            kept.append(pair)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{'&'.join(kept)}"


@dataclass
class HttpResponse:
    """Normalized HTTP response."""

    url: str
    status_code: int
    content: bytes
    text: str
    headers: dict[str, str]
    etag: str | None = None
    last_modified: str | None = None
    content_sha256: str | None = None
    from_cache: bool = False

    def __post_init__(self) -> None:
        if self.content_sha256 is None and self.content:
            self.content_sha256 = hashlib.sha256(self.content).hexdigest()


@dataclass
class CachedSource:
    """Previously published source data for conditional requests."""

    etag: str | None = None
    last_modified: str | None = None
    content_sha256: str | None = None


class HttpClient:
    """HTTP client with retries, allowlist, and conditional request support."""

    def __init__(self, config: CrawlConfiguration | None = None) -> None:
        self.config = config or CrawlConfiguration()
        self._semaphore = asyncio.Semaphore(self.config.max_workers)
        self._client: httpx.AsyncClient | None = None
        self._lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        async with self._lock:
            if self._client is None or self._client.is_closed:
                timeout = httpx.Timeout(
                    connect=self.config.connect_timeout,
                    read=self.config.read_timeout,
                    write=10.0,
                    pool=10.0,
                )
                self._client = httpx.AsyncClient(
                    timeout=timeout,
                    follow_redirects=True,
                    cookies=None,
                    headers={
                        "User-Agent": self.config.user_agent or DEFAULT_USER_AGENT,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "en-US,en;q=0.5",
                        # PayPal's edge cache returns a pre-rendered, CMS-free page when the
                        # client advertises compression support. Requesting identity encoding
                        # ensures the dynamic origin page with the CMS context is served.
                        "Accept-Encoding": "identity",
                        "DNT": "1",
                        "Connection": "keep-alive",
                        "Upgrade-Insecure-Requests": "1",
                    },
                )
            return self._client

    async def close(self) -> None:
        async with self._lock:
            if self._client is not None and not self._client.is_closed:
                await self._client.aclose()
                self._client = None

    async def __aenter__(self) -> HttpClient:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.close()

    def _is_allowed_host(self, url: str) -> bool:
        host = urlparse(url).hostname or ""
        return any(host == allowed or host.endswith(f".{allowed}") for allowed in self.config.allowed_domains)

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ContentSecurityError(f"Unsupported URL scheme: {parsed.scheme}")
        if not parsed.hostname:
            raise ContentSecurityError(f"Missing hostname in URL: {url}")
        if not self._is_allowed_host(url):
            raise ContentSecurityError(f"Host not in allowlist: {parsed.hostname}")

    def _detect_blocking_page(self, response: HttpResponse) -> None:
        """Raise if the response looks like a login, CAPTCHA, or generic error page.

        Detection is structural, not a simple substring search. A generic JavaScript
        dependency whose URL contains the string ``captcha`` must not by itself
        classify the page as a challenge.
        """
        if response.status_code == 429:
            retry_after = response.headers.get("retry-after")
            raise RateLimitError(f"Rate limited (429): {response.url}", retry_after=retry_after)
        if response.status_code >= 500:
            raise TransientNetworkError(f"Server error ({response.status_code}): {response.url}")
        if response.status_code in (401, 403):
            raise AccessChallengeError(f"Access denied ({response.status_code}): {response.url}")
        if response.status_code >= 400:
            raise PermanentHttpError(f"HTTP {response.status_code} for {response.url}", response.status_code)

        title, challenge_score, login_score = self._parse_challenge_signals(response.text)

        # Strong evidence: challenge title phrase plus at least one structural signal.
        if title and self._is_challenge_title(title) and challenge_score >= 1:
            raise AccessChallengeError(f"CAPTCHA/interstitial detected: {response.url}")

        # Multiple structural challenge signals also trigger a block.
        if challenge_score >= 2:
            raise AccessChallengeError(f"CAPTCHA/interstitial detected: {response.url}")

        # Login pages are classified separately.
        if login_score >= 2 and self._looks_like_login_page(response.text):
            raise AccessChallengeError(f"Login page detected: {response.url}")

    def _parse_challenge_signals(self, text: str) -> tuple[str | None, int, int]:
        """Parse HTML and return (title, challenge_score, login_score).

        The challenge score counts independent structural signals:
        - a known CAPTCHA form, iframe, or container
        - a challenge-specific page identifier (class/id)
        - a missing CMS context while the page title is suspicious
        """
        try:
            tree = html.fromstring(text)
        except Exception:
            return None, 0, 0

        title = None
        title_node = tree.find(".//title")
        if title_node is not None and title_node.text:
            title = title_node.text.strip()

        challenge_score = 0
        login_score = 0

        # Known challenge containers and form identifiers.
        challenge_selectors = [
            "//form[contains(@action, 'captcha') or contains(@id, 'captcha') or contains(@class, 'captcha')]",
            "//form[contains(@action, 'challenge') or contains(@id, 'challenge') or contains(@class, 'challenge')]",
            "//iframe[contains(@src, 'captcha') or contains(@name, 'captcha')]",
            "//iframe[contains(@src, 'challenge') or contains(@name, 'challenge')]",
            "//*[contains(@class, 'cf-turnstile') or contains(@class, 'g-recaptcha') or contains(@class, 'h-captcha')]",
            "//*[contains(@id, 'rc-anchor-container') or contains(@class, 'recaptcha')]",
            "//input[contains(@name, 'captcha') or contains(@id, 'captcha')]",
            "//div[contains(@class, 'security-check') or contains(@id, 'security-check')]",
            "//div[contains(@class, 'human-verification') or contains(@id, 'human-verification')]",
        ]
        for selector in challenge_selectors:
            if tree.xpath(selector):
                challenge_score += 1
                break

        # A known challenge page identifier in the URL path or page id.
        body_id = tree.xpath("//body/@id")
        body_classes = tree.xpath("//body/@class")
        body_attrs = " ".join(body_id + body_classes).lower()
        if any(token in body_attrs for token in ("captcha", "challenge", "security-check")):
            challenge_score += 1

        # The expected PayPal CMS context is absent.
        scripts = tree.xpath("//script/text()")
        has_cms_context = any("window.__CMS_ENGINE_RENDER_CONTEXT__" in (s or "") for s in scripts)
        if not has_cms_context:
            challenge_score += 1
            # No CMS context is also a login/error signal.
            login_score += 1

        # Login signals.
        login_form = tree.xpath(
            "//form[.//input[@type='password'] or contains(@action, 'signin') or contains(@action, 'login')]"
        )
        if login_form:
            login_score += 1
        if title and self._is_login_title(title):
            login_score += 1

        return title, challenge_score, login_score

    def _is_challenge_title(self, title: str) -> bool:
        lower = title.lower()
        return any(
            phrase in lower
            for phrase in (
                "captcha",
                "security check",
                "verify you are human",
                "robot check",
                "challenge",
                "are you human",
                "verify your identity",
            )
        )

    def _is_login_title(self, title: str) -> bool:
        lower = title.lower()
        return any(
            phrase in lower
            for phrase in (
                "log in",
                "login",
                "sign in",
                "signin",
                "account login",
            )
        )

    def _looks_like_login_page(self, text: str) -> bool:
        """Return True if the page contains a PayPal login form."""
        try:
            tree = html.fromstring(text)
        except Exception:
            return False
        return bool(
            tree.xpath(
                "//form[.//input[@type='password']]"
                "[contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), 'paypal')]"
            )
        )

    def _calculate_backoff(self, attempt: int) -> float:
        base = 2.0**attempt
        jitter = random.uniform(0, 1)  # noqa: S311 # nosec B311
        return base + jitter

    async def _request(
        self,
        method: str,
        url: str,
        cached: CachedSource | None = None,
        **kwargs: Any,
    ) -> HttpResponse:
        self._validate_url(url)
        headers: dict[str, str] = {}
        if cached:
            if cached.etag:
                headers["If-None-Match"] = cached.etag
            if cached.last_modified:
                headers["If-Modified-Since"] = cached.last_modified

        client = await self._get_client()
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                async with self._semaphore:
                    logger.debug("%s %s (attempt %d)", method, _sanitize_url(url), attempt + 1)
                    response = await client.request(
                        method,
                        url,
                        headers=headers,
                        **kwargs,
                    )
                    # Check final URL after redirects
                    final_url = str(response.url)
                    if final_url != url:
                        self._validate_url(final_url)
                    if len(response.content) > self.config.max_response_size:
                        raise ContentSecurityError(
                            f"Response size {len(response.content)} exceeds limit for {final_url}"
                        )
                    if response.status_code == 304:
                        return HttpResponse(
                            url=final_url,
                            status_code=304,
                            content=b"",
                            text="",
                            headers=dict(response.headers),
                            etag=response.headers.get("etag"),
                            last_modified=response.headers.get("last-modified"),
                            from_cache=True,
                        )
                    # Convert to HttpResponse early so we can inspect headers/text
                    http_response = HttpResponse(
                        url=final_url,
                        status_code=response.status_code,
                        content=response.content,
                        text=response.text,
                        headers=dict(response.headers),
                        etag=response.headers.get("etag"),
                        last_modified=response.headers.get("last-modified"),
                    )
                    self._detect_blocking_page(http_response)
                    if self.config.request_delay > 0:
                        await asyncio.sleep(self.config.request_delay)
                    return http_response
            except (TransientNetworkError, httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    raise NetworkError(f"Failed after {attempt + 1} attempts: {url}") from exc
                retry_after = 0.0
                if isinstance(exc, TransientNetworkError) and exc.retry_after is not None:
                    try:
                        retry_after = float(exc.retry_after)
                    except ValueError:
                        retry_after = 0.0
                delay = max(retry_after, self._calculate_backoff(attempt))
                logger.warning("Transient error for %s, retrying in %.2fs: %s", _sanitize_url(url), delay, exc)
                await asyncio.sleep(delay)
            except (ContentSecurityError, PermanentNetworkError):
                raise
        if last_error:
            raise NetworkError(f"Failed to request {url}") from last_error
        raise NetworkError(f"Unexpected end of retries for {url}")

    async def get(self, url: str, cached: CachedSource | None = None) -> HttpResponse:
        return await self._request("GET", url, cached=cached)

    async def head(self, url: str) -> HttpResponse:
        return await self._request("HEAD", url)
