"""Tests for the HTTP client."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from paypal_fee_crawler.exceptions import ContentSecurityError, PermanentNetworkError
from paypal_fee_crawler.http import CachedSource, HttpClient, HttpResponse, _sanitize_url
from paypal_fee_crawler.models import CrawlConfiguration


def test_sanitize_url_redacts_tokens() -> None:
    url = "https://www.paypal.com/foo?token=secret&session=abc&other=value"
    sanitized = _sanitize_url(url)
    assert "secret" not in sanitized
    assert "abc" not in sanitized
    assert "other=value" in sanitized


def test_sanitize_url_no_query() -> None:
    assert _sanitize_url("https://www.paypal.com/foo") == "https://www.paypal.com/foo"


async def _run_get(
    handler: Any,
    url: str = "https://www.paypal.com/de/business/paypal-business-fees",
    cached: CachedSource | None = None,
    config: CrawlConfiguration | None = None,
) -> HttpResponse:
    cfg = config or CrawlConfiguration(max_workers=1, request_delay=0)
    async with HttpClient(cfg) as client:
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return await client.get(url, cached=cached)


def test_get_success() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>ok</html>", headers={"etag": '"abc"'})

    response = asyncio.run(_run_get(handler))
    assert response.status_code == 200
    assert response.text == "<html>ok</html>"
    assert response.etag == '"abc"'


def test_get_disallowed_domain() -> None:
    config = CrawlConfiguration(allowed_domains=["www.paypal.com"])
    with pytest.raises(ContentSecurityError, match="Host not in allowlist"):
        asyncio.run(_run_get(lambda r: httpx.Response(200), url="https://evil.com/foo", config=config))


def test_get_blocking_page() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<title>Security check</title><div class='captcha'></div>")

    with pytest.raises(PermanentNetworkError, match="CAPTCHA"):
        asyncio.run(_run_get(handler))


def test_get_retry_after() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, text="rate limited", headers={"retry-after": "0"})
        return httpx.Response(200, text="ok")

    config = CrawlConfiguration(max_workers=1, request_delay=0, max_retries=1)
    response = asyncio.run(_run_get(handler, config=config))
    assert response.status_code == 200
    assert len(calls) == 2


def test_get_4xx_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="not found")

    config = CrawlConfiguration(max_workers=1, request_delay=0, max_retries=0)
    with pytest.raises(PermanentNetworkError):
        asyncio.run(_run_get(handler, config=config))


def test_get_too_large() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"x" * (20 * 1024 * 1024))

    config = CrawlConfiguration(max_workers=1, request_delay=0, max_response_size=1024)
    with pytest.raises(ContentSecurityError, match="exceeds"):
        asyncio.run(_run_get(handler, config=config))


def test_get_conditional_not_modified() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["if-none-match"] == '"abc"'
        return httpx.Response(304, content=b"", headers={"content-type": "text/html"})

    config = CrawlConfiguration(max_workers=1, request_delay=0)
    response = asyncio.run(_run_get(handler, cached=CachedSource(etag='"abc"'), config=config))
    assert response.status_code == 304
