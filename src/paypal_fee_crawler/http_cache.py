"""Persistent 24-hour HTTP response cache for the PayPal fee crawler.

The cache stores complete successful HTTP responses on disk, keyed by the
effective request identity (URL, market, locale, content-negotiation headers and
a crawler-specific cache version).  It supports freshness checks, conditional
revalidation, atomic writes, and per-key file locking so multiple workers never
download the same unchanged resource twice.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import httpx

try:
    from filelock import FileLock
except ImportError:  # pragma: no cover - filelock is a declared dependency
    FileLock = None  # type: ignore[misc,assignment]

from .models import CacheStats, CrawlConfiguration

logger = logging.getLogger(__name__)

CACHE_VERSION = "1"

# Content-negotiation headers that can change which market/language is served.
_NEGOTIATION_HEADERS = {"accept", "accept-language", "accept-encoding", "accept-charset"}

# Query parameters that are volatile or sensitive and must not affect the key.
_VOLATILE_PARAMS = {
    "token",
    "session",
    "sid",
    "auth",
    "nonce",
    "csrf",
    "request_id",
    "correlation_id",
    "visitor_id",
    "utm_source",
    "utm_medium",
    "utm_campaign",
}

# Response headers considered relevant to store and replay.
_CACHED_RESPONSE_HEADERS = {
    "content-type",
    "content-language",
    "content-length",
    "etag",
    "last-modified",
    "cache-control",
    "expires",
}


def _normalize_url(url: str) -> str:
    """Return a stable, normalized URL with sorted query parameters."""
    parsed = urlparse(url)
    pairs = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() not in _VOLATILE_PARAMS]
    normalized_query = urlencode(sorted(pairs))
    return urlunparse(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path,
            "",
            normalized_query,
            "",
        )
    )


def _market_from_url(url: str) -> str | None:
    """Return the first path segment of a PayPal URL as a market code."""
    parsed = urlparse(url)
    segments = [s for s in parsed.path.strip("/").split("/") if s]
    if segments:
        return segments[0].upper()
    return None


def _cache_key(method: str, url: str, market: str | None, locale: str | None, headers: dict[str, str]) -> str:
    """Return a stable SHA-256 hash representing the cache key."""
    headers_lower = {k.lower(): v for k, v in headers.items()}
    identity: dict[str, Any] = {
        "v": CACHE_VERSION,
        "method": method.upper(),
        "url": _normalize_url(url),
        "accept": headers_lower.get("accept"),
        "accept_language": headers_lower.get("accept-language"),
        "accept_encoding": headers_lower.get("accept-encoding"),
        "accept_charset": headers_lower.get("accept-charset"),
    }
    identity["market"] = market or _market_from_url(url)
    if locale:
        identity["locale"] = locale

    serialized = json.dumps(identity, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _filter_response_headers(headers: httpx.Headers | dict[str, str]) -> dict[str, str]:
    """Return only the response headers we want to replay from the cache."""
    if isinstance(headers, httpx.Headers):
        headers = dict(headers)
    return {k.lower(): v for k, v in headers.items() if k.lower() in _CACHED_RESPONSE_HEADERS and v is not None}


def _is_valid_cacheable_response(response: httpx.Response) -> bool:
    """Return True if a 200 response has a complete, cacheable body."""
    if response.status_code != 200:
        return False
    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            expected = int(content_length)
        except ValueError:
            return False
        if len(response.content) != expected:
            logger.debug("Response content-length mismatch (%s vs %s); not caching", expected, len(response.content))
            return False
    return True


@dataclass
class _CacheEntry:
    """On-disk cache entry (serialized to JSON)."""

    key: str
    url: str
    final_url: str | None
    status_code: int
    headers: dict[str, str]
    content: bytes
    etag: str | None
    last_modified: str | None
    fetched_at: float
    cache_version: str
    market: str | None
    locale: str | None
    cache_control: str | None

    def to_json(self) -> dict[str, Any]:
        return {
            "v": self.cache_version,
            "key": self.key,
            "url": self.url,
            "final_url": self.final_url,
            "status_code": self.status_code,
            "headers": self.headers,
            "content": base64.b64encode(self.content).decode("ascii"),
            "etag": self.etag,
            "last_modified": self.last_modified,
            "fetched_at": self.fetched_at,
            "market": self.market,
            "locale": self.locale,
            "cache_control": self.cache_control,
        }

    def to_httpx_response(self, method: str) -> httpx.Response:
        """Replay this entry as an ``httpx.Response``."""
        return httpx.Response(
            self.status_code,
            content=self.content,
            headers=self.headers,
            request=httpx.Request(method, self.final_url or self.url, headers={}),
        )

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> _CacheEntry | None:
        try:
            version = data.get("v")
            if version != CACHE_VERSION:
                return None
            content = base64.b64decode(data.get("content", ""))
            return cls(
                key=data.get("key", ""),
                url=data["url"],
                final_url=data.get("final_url"),
                status_code=int(data["status_code"]),
                headers=data.get("headers", {}),
                content=content,
                etag=data.get("etag"),
                last_modified=data.get("last_modified"),
                fetched_at=float(data["fetched_at"]),
                cache_version=version,
                market=data.get("market"),
                locale=data.get("locale"),
                cache_control=data.get("cache_control"),
            )
        except Exception:
            return None


class HttpCache:
    """Persistent on-disk HTTP response cache."""

    def __init__(self, config: CrawlConfiguration) -> None:
        self.config = config
        self.stats = CacheStats()
        if config.cache_dir is not None and not config.no_cache:
            self._enabled = True
            self._cache_dir: Path | None = Path(config.cache_dir)
        else:
            self._enabled = False
            self._cache_dir = None
        self._ttl_seconds = config.cache_ttl_hours * 3600.0
        self._refresh = config.refresh_cache
        self._key_locks: dict[str, asyncio.Lock] = {}
        self._file_locks: dict[str, Any] = {}

        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, method: str, url: str, market: str | None, locale: str | None, headers: dict[str, str]) -> str:
        return _cache_key(method, url, market, locale, headers)

    def _entry_path(self, key: str) -> Path:
        if self._cache_dir is None:
            raise RuntimeError("Cache directory is not configured")
        return self._cache_dir / "entries" / key[:2] / f"{key}.json"

    def _lock_path(self, key: str) -> Path:
        if self._cache_dir is None:
            raise RuntimeError("Cache directory is not configured")
        return self._cache_dir / "locks" / key[:2] / f"{key}.lock"

    def _key_lock(self, key: str) -> asyncio.Lock:
        if key not in self._key_locks:
            self._key_locks[key] = asyncio.Lock()
        return self._key_locks[key]

    def _file_lock(self, key: str) -> Any:
        if key not in self._file_locks:
            lock_path = self._lock_path(key)
            self._file_locks[key] = FileLock(lock_path, thread_local=False) if FileLock else None
        return self._file_locks[key]

    def _is_fresh(self, entry: _CacheEntry) -> bool:
        return (time.time() - entry.fetched_at) < self._ttl_seconds

    def _read_entry(self, key: str) -> _CacheEntry | None:
        if self._cache_dir is None:
            return None
        path = self._entry_path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.debug("Corrupt cache entry at %s; ignoring", path)
            self.stats.cache_errors += 1
            return None
        entry = _CacheEntry.from_json(data)
        if entry is None:
            logger.debug("Stale or invalid cache entry at %s; ignoring", path)
            self.stats.cache_errors += 1
            with contextlib.suppress(OSError):
                path.unlink(missing_ok=True)
            return None
        return entry

    def _write_entry(self, entry: _CacheEntry) -> None:
        if self._cache_dir is None:
            return
        path = self._entry_path(entry.key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".tmp.{path.name}.{os.getpid()}.{time.monotonic_ns()}")
        try:
            tmp.write_text(json.dumps(entry.to_json(), ensure_ascii=False, sort_keys=True), encoding="utf-8")
            os.replace(tmp, path)
        except Exception as exc:
            logger.warning("Failed to write cache entry %s: %s", path, exc)
            self.stats.cache_errors += 1
            with contextlib.suppress(OSError):
                tmp.unlink(missing_ok=True)

    def _remove_entry(self, key: str) -> None:
        if self._cache_dir is None:
            return
        path = self._entry_path(key)
        with contextlib.suppress(OSError):
            path.unlink(missing_ok=True)

    async def _acquire_file_lock(self, key: str) -> Any:
        lock = self._file_lock(key)
        if lock is None:
            return None
        try:
            await asyncio.to_thread(lock.acquire)
        except Exception as exc:
            logger.debug("Could not acquire cache lock for %s: %s", key, exc)
            return None
        return lock

    def _release_file_lock(self, lock: Any) -> None:
        if lock is None:
            return
        try:
            lock.release()
        except Exception as exc:
            logger.debug("Could not release cache lock: %s", exc)

    @staticmethod
    def _revalidation_headers(entry: _CacheEntry | None, cached: Any | None) -> dict[str, str]:
        headers: dict[str, str] = {}
        source = entry if entry is not None else cached
        if source is None:
            return headers
        etag = getattr(source, "etag", None)
        last_modified = getattr(source, "last_modified", None)
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified
        return headers

    async def fetch(
        self,
        method: str,
        url: str,
        headers: dict[str, str],
        network: Any,
        *,
        market: str | None = None,
        locale: str | None = None,
        cached: Any | None = None,
    ) -> tuple[httpx.Response, bool]:
        """Return a response and a flag indicating whether it came from the cache.

        ``network`` must be an awaitable callable that accepts the full request
        headers dict and a dict of conditional request headers and returns an
        ``httpx.Response``.
        """
        request_headers = {**headers}
        if locale:
            request_headers["Accept-Language"] = locale

        if method.upper() != "GET":
            reval = self._revalidation_headers(None, cached)
            response = await network(request_headers, reval)
            return response, False

        if not self._enabled:
            reval = self._revalidation_headers(None, cached)
            response = await network(request_headers, reval)
            return response, False

        key = self._key(method, url, market, locale, request_headers)
        async with self._key_lock(key):
            lock = await self._acquire_file_lock(key)
            try:
                entry = self._read_entry(key)

                if entry is not None and not self._refresh and self._is_fresh(entry):
                    self.stats.cache_hits += 1
                    self.stats.bytes_avoided += len(entry.content)
                    logger.debug("Cache hit for %s", _normalize_url(url))
                    return entry.to_httpx_response(method), True

                reval = self._revalidation_headers(entry, cached)
                if reval:
                    self.stats.cache_revalidations += 1
                    logger.debug("Cache revalidation for %s", _normalize_url(url))

                response = await network(request_headers, reval)

                if response.status_code == 304:
                    if entry is not None:
                        entry.fetched_at = time.time()
                        self._write_entry(entry)
                        self.stats.cache_304_responses += 1
                        self.stats.bytes_avoided += len(entry.content)
                        return entry.to_httpx_response(method), True
                    # No stored body for this 304; return the upstream response.
                    return response, False

                if _is_valid_cacheable_response(response):
                    final_url = str(response.url)
                    entry = _CacheEntry(
                        key=key,
                        url=url,
                        final_url=final_url,
                        status_code=response.status_code,
                        headers=_filter_response_headers(response.headers),
                        content=response.content,
                        etag=response.headers.get("etag"),
                        last_modified=response.headers.get("last-modified"),
                        fetched_at=time.time(),
                        cache_version=CACHE_VERSION,
                        market=market or _market_from_url(url),
                        locale=locale,
                        cache_control=response.headers.get("cache-control"),
                    )
                    self._write_entry(entry)
                    self.stats.cache_writes += 1
                    self.stats.cache_misses += 1
                    return response, False

                # Not a cacheable 200; pass through without writing.
                self.stats.cache_misses += 1
                return response, False
            finally:
                self._release_file_lock(lock)
