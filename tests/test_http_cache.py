"""Tests for the persistent HTTP response cache."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from paypal_fee_crawler.http import HttpClient, HttpResponse
from paypal_fee_crawler.http_cache import _cache_key, _CacheEntry
from paypal_fee_crawler.models import CrawlConfiguration


async def _run_get(
    handler: Any,
    url: str = "https://www.paypal.com/de/business/paypal-business-fees",
    market: str | None = None,
    locale: str | None = None,
    config: CrawlConfiguration | None = None,
) -> HttpResponse:
    cfg = config or CrawlConfiguration(max_workers=1, request_delay=0)
    async with HttpClient(cfg) as client:
        client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        return await client.get(url, market=market, locale=locale)


def _entry_path(cache_dir: Path, method: str, url: str, market: str | None, locale: str | None) -> Path:
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "identity",
    }
    key = _cache_key(method, url, market, locale, headers)
    return cache_dir / "entries" / key[:2] / f"{key}.json"


def test_fresh_cache_hit_performs_no_network_request(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, text="<html>ok</html>", headers={"etag": '"abc"'})

    config = CrawlConfiguration(max_workers=1, request_delay=0, cache_dir=str(tmp_path / "cache"))
    response1 = asyncio.run(_run_get(handler, config=config))
    assert response1.status_code == 200
    assert response1.text == "<html>ok</html>"
    assert not response1.from_cache
    assert len(calls) == 1

    response2 = asyncio.run(_run_get(handler, config=config))
    assert response2.status_code == 200
    assert response2.text == "<html>ok</html>"
    assert response2.from_cache
    assert len(calls) == 1


def test_expired_cache_entry_triggers_revalidation(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    url = "https://www.paypal.com/de/business/paypal-business-fees"
    config = CrawlConfiguration(max_workers=1, request_delay=0, cache_dir=str(cache_dir))
    entry = _CacheEntry(
        key="",
        url=url,
        final_url=url,
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
        content=b"<html>old</html>",
        etag='"abc"',
        last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
        fetched_at=time.time() - 48 * 3600,
        cache_version="1",
        market="DE",
        locale=None,
        cache_control=None,
    )
    entry.key = _cache_key(
        "GET",
        url,
        "DE",
        None,
        {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "identity",
        },
    )
    path = _entry_path(cache_dir, "GET", url, "DE", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry.to_json(), ensure_ascii=False, sort_keys=True), encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["if-none-match"] == '"abc"'
        return httpx.Response(304, headers={"etag": '"abc"'})

    response = asyncio.run(_run_get(handler, url=url, market="DE", config=config))
    assert response.status_code == 200
    assert response.text == "<html>old</html>"
    assert response.from_cache


def test_304_reuses_cached_body(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    url = "https://www.paypal.com/de/business/paypal-business-fees"
    config = CrawlConfiguration(max_workers=1, request_delay=0, cache_dir=str(cache_dir))
    entry = _CacheEntry(
        key="",
        url=url,
        final_url=url,
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
        content=b"<html>cached</html>",
        etag='"abc"',
        last_modified=None,
        fetched_at=time.time() - 48 * 3600,
        cache_version="1",
        market="DE",
        locale=None,
        cache_control=None,
    )
    entry.key = _cache_key(
        "GET",
        url,
        "DE",
        None,
        {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "identity",
        },
    )
    path = _entry_path(cache_dir, "GET", url, "DE", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry.to_json(), ensure_ascii=False, sort_keys=True), encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(304, headers={"etag": '"abc"'})

    response = asyncio.run(_run_get(handler, url=url, market="DE", config=config))
    assert response.status_code == 200
    assert response.text == "<html>cached</html>"
    assert response.from_cache


def test_changed_200_response_replaces_entry(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    url = "https://www.paypal.com/de/business/paypal-business-fees"
    config = CrawlConfiguration(max_workers=1, request_delay=0, cache_dir=str(cache_dir))
    entry = _CacheEntry(
        key="",
        url=url,
        final_url=url,
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
        content=b"<html>old</html>",
        etag='"abc"',
        last_modified=None,
        fetched_at=time.time() - 48 * 3600,
        cache_version="1",
        market="DE",
        locale=None,
        cache_control=None,
    )
    entry.key = _cache_key(
        "GET",
        url,
        "DE",
        None,
        {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "identity",
        },
    )
    path = _entry_path(cache_dir, "GET", url, "DE", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry.to_json(), ensure_ascii=False, sort_keys=True), encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>new</html>", headers={"etag": '"def"'})

    response = asyncio.run(_run_get(handler, url=url, market="DE", config=config))
    assert response.status_code == 200
    assert response.text == "<html>new</html>"
    assert not response.from_cache

    # A fresh fetch must now come from the cache with the new body.
    calls: list[httpx.Request] = []

    def cached_handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, text="<html>newer</html>")

    response2 = asyncio.run(_run_get(cached_handler, url=url, market="DE", config=config))
    assert response2.text == "<html>new</html>"
    assert response2.from_cache
    assert not calls


def test_different_markets_do_not_share_entries(tmp_path: Path) -> None:
    responses = iter(["<html>DE</html>", "<html>FR</html>"])
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, text=next(responses), headers={"etag": '"x"'})

    url = "https://www.paypal.com/de/business/paypal-business-fees"
    config = CrawlConfiguration(max_workers=1, request_delay=0, cache_dir=str(tmp_path / "cache"))

    de = asyncio.run(_run_get(handler, url=url, market="DE", config=config))
    fr = asyncio.run(_run_get(handler, url=url, market="FR", config=config))
    de_again = asyncio.run(_run_get(handler, url=url, market="DE", config=config))

    assert de.text == "<html>DE</html>"
    assert fr.text == "<html>FR</html>"
    assert de_again.text == "<html>DE</html>"
    assert de_again.from_cache
    assert len(calls) == 2


def test_query_params_are_part_of_cache_key(tmp_path: Path) -> None:
    responses = iter(["<html>a</html>", "<html>b</html>"])
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, text=next(responses))

    config = CrawlConfiguration(max_workers=1, request_delay=0, cache_dir=str(tmp_path / "cache"))
    a = asyncio.run(_run_get(handler, url="https://www.paypal.com/de/page?a=1", config=config))
    b = asyncio.run(_run_get(handler, url="https://www.paypal.com/de/page?b=2", config=config))

    assert a.text == "<html>a</html>"
    assert b.text == "<html>b</html>"
    assert len(calls) == 2


def test_concurrent_requests_download_one_resource_only_once(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, text="<html>ok</html>")

    config = CrawlConfiguration(max_workers=1, request_delay=0, cache_dir=str(tmp_path / "cache"))

    async def _run() -> tuple[HttpResponse, HttpResponse]:
        async with HttpClient(config) as client:
            client._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            return await asyncio.gather(
                client.get("https://www.paypal.com/de/page"),
                client.get("https://www.paypal.com/de/page"),
            )

    r1, r2 = asyncio.run(_run())
    assert r1.text == "<html>ok</html>"
    assert r2.text == "<html>ok</html>"
    assert len(calls) == 1


def test_corrupt_cache_entry_is_ignored_and_replaced(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    url = "https://www.paypal.com/de/business/paypal-business-fees"
    path = _entry_path(cache_dir, "GET", url, "DE", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("not valid json", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>fresh</html>", headers={"etag": '"x"'})

    config = CrawlConfiguration(max_workers=1, request_delay=0, cache_dir=str(cache_dir))
    response = asyncio.run(_run_get(handler, url=url, market="DE", config=config))
    assert response.text == "<html>fresh</html>"
    assert not response.from_cache
    data = json.loads(path.read_text(encoding="utf-8"))
    assert base64.b64decode(data["content"]).decode("utf-8") == "<html>fresh</html>"


def test_failed_responses_are_not_cached(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(500, text="error")

    config = CrawlConfiguration(max_workers=1, request_delay=0, cache_dir=str(tmp_path / "cache"), max_retries=0)

    from paypal_fee_crawler.exceptions import NetworkError, TransientNetworkError

    with pytest.raises(NetworkError) as exc_info:
        asyncio.run(_run_get(handler, config=config))

    assert isinstance(exc_info.value.__cause__, TransientNetworkError)
    entries = list((tmp_path / "cache" / "entries").rglob("*.json"))
    assert not entries


def test_no_cache_bypasses_cache(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    url = "https://www.paypal.com/de/business/paypal-business-fees"
    config = CrawlConfiguration(
        max_workers=1,
        request_delay=0,
        cache_dir=str(cache_dir),
        no_cache=True,
    )

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, text="<html>network</html>")

    response = asyncio.run(_run_get(handler, url=url, market="DE", config=config))
    assert response.text == "<html>network</html>"
    assert not response.from_cache
    assert len(calls) == 1
    entries = list((cache_dir / "entries").rglob("*.json"))
    assert not entries


def test_refresh_cache_forces_revalidation(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    url = "https://www.paypal.com/de/business/paypal-business-fees"
    config = CrawlConfiguration(
        max_workers=1,
        request_delay=0,
        cache_dir=str(cache_dir),
        refresh_cache=True,
    )

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["if-none-match"] == '"abc"'
        return httpx.Response(304, headers={"etag": '"abc"'})

    # Seed a fresh cache entry.
    entry = _CacheEntry(
        key="",
        url=url,
        final_url=url,
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
        content=b"<html>cached</html>",
        etag='"abc"',
        last_modified=None,
        fetched_at=time.time(),
        cache_version="1",
        market="DE",
        locale=None,
        cache_control=None,
    )
    entry.key = _cache_key(
        "GET",
        url,
        "DE",
        None,
        {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "identity",
        },
    )
    path = _entry_path(cache_dir, "GET", url, "DE", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry.to_json(), ensure_ascii=False, sort_keys=True), encoding="utf-8")

    response = asyncio.run(_run_get(handler, url=url, market="DE", config=config))
    assert response.text == "<html>cached</html>"
    assert response.from_cache
    assert len(calls) == 1


def test_output_is_identical_with_and_without_cache(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="<html>ok</html>",
            headers={"content-type": "text/html; charset=utf-8", "etag": '"abc"'},
        )

    cached_config = CrawlConfiguration(max_workers=1, request_delay=0, cache_dir=str(tmp_path / "cache"))
    uncached_config = CrawlConfiguration(
        max_workers=1, request_delay=0, cache_dir=str(tmp_path / "cache"), no_cache=True
    )

    cached_response = asyncio.run(_run_get(handler, config=cached_config))
    uncached_response = asyncio.run(_run_get(handler, config=uncached_config))

    assert cached_response.text == uncached_response.text
    assert cached_response.content == uncached_response.content
    assert cached_response.etag == uncached_response.etag
    assert cached_response.headers == uncached_response.headers


def test_cookies_are_not_stored_or_shared_between_markets(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            text="<html>ok</html>",
            headers={"set-cookie": "session=secret; Path=/", "etag": '"x"'},
        )

    config = CrawlConfiguration(max_workers=1, request_delay=0, cache_dir=str(tmp_path / "cache"))

    async def _run() -> tuple[HttpResponse, HttpResponse]:
        async with HttpClient(config) as client:
            # Force the client to create a fresh httpx client per request, just
            # like production does when no shared client is injected.
            client._client = None
            client._create_client = lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler))
            de = await client.get("https://www.paypal.com/de/page", market="DE")
            fr = await client.get("https://www.paypal.com/fr/page", market="FR")
            return de, fr

    de, fr = asyncio.run(_run())
    assert de.headers.get("set-cookie") == "session=secret; Path=/"
    assert fr.headers.get("set-cookie") == "session=secret; Path=/"
    # No Cookie header should be sent on the second request.
    assert "cookie" not in requests[1].headers

    # Cache must not store Set-Cookie either.
    entry_files = list((tmp_path / "cache" / "entries").rglob("*.json"))
    assert entry_files
    data = json.loads(entry_files[0].read_text(encoding="utf-8"))
    assert "set-cookie" not in {k.lower() for k in data["headers"]}


def test_sensitive_query_params_produce_different_cache_keys() -> None:
    """Sensitive or content-affecting query parameters must not share a cache key."""
    headers = {"Accept": "text/html", "Accept-Language": "en", "Accept-Encoding": "identity"}
    base = "https://www.paypal.com/de/page"
    for param in ("token", "session", "auth"):
        key_a = _cache_key("GET", f"{base}?{param}=A", "DE", None, headers)
        key_b = _cache_key("GET", f"{base}?{param}=B", "DE", None, headers)
        assert key_a != key_b, f"{param} values should not collide"

    # Safe UTM parameters are stripped from the key and may collide.
    key_a = _cache_key("GET", f"{base}?utm_source=a", "DE", None, headers)
    key_b = _cache_key("GET", f"{base}?utm_source=b", "DE", None, headers)
    assert key_a == key_b


def test_no_store_responses_are_not_cached(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            text=f"<html>call {len(calls)}</html>",
            headers={"cache-control": "no-store"},
        )

    config = CrawlConfiguration(max_workers=1, request_delay=0, cache_dir=str(tmp_path / "cache"))
    r1 = asyncio.run(_run_get(handler, url="https://www.paypal.com/de/page", config=config))
    r2 = asyncio.run(_run_get(handler, url="https://www.paypal.com/de/page", config=config))

    assert r1.text == "<html>call 1</html>"
    assert r2.text == "<html>call 2</html>"
    assert len(calls) == 2
    assert not list((tmp_path / "cache" / "entries").rglob("*.json"))


def test_private_responses_are_not_cached(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            text=f"<html>call {len(calls)}</html>",
            headers={"cache-control": "private"},
        )

    config = CrawlConfiguration(max_workers=1, request_delay=0, cache_dir=str(tmp_path / "cache"))
    r1 = asyncio.run(_run_get(handler, url="https://www.paypal.com/de/page", config=config))
    r2 = asyncio.run(_run_get(handler, url="https://www.paypal.com/de/page", config=config))

    assert r1.text == "<html>call 1</html>"
    assert r2.text == "<html>call 2</html>"
    assert len(calls) == 2
    assert not list((tmp_path / "cache" / "entries").rglob("*.json"))


def test_no_cache_triggers_revalidation(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    url = "https://www.paypal.com/de/business/paypal-business-fees"
    config = CrawlConfiguration(max_workers=1, request_delay=0, cache_dir=str(cache_dir))
    entry = _CacheEntry(
        key="",
        url=url,
        final_url=url,
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
        content=b"<html>cached</html>",
        etag='"abc"',
        last_modified=None,
        fetched_at=time.time(),
        cache_version="1",
        market="DE",
        locale=None,
        cache_control="no-cache",
    )
    entry.key = _cache_key(
        "GET",
        url,
        "DE",
        None,
        {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "identity",
        },
    )
    path = _entry_path(cache_dir, "GET", url, "DE", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry.to_json(), ensure_ascii=False, sort_keys=True), encoding="utf-8")

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["if-none-match"] == '"abc"'
        return httpx.Response(304, headers={"etag": '"abc"'})

    response = asyncio.run(_run_get(handler, url=url, market="DE", config=config))
    assert response.text == "<html>cached</html>"
    assert response.from_cache
    assert len(calls) == 1


def test_max_age_zero_triggers_revalidation(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    url = "https://www.paypal.com/de/business/paypal-business-fees"
    config = CrawlConfiguration(max_workers=1, request_delay=0, cache_dir=str(cache_dir))
    entry = _CacheEntry(
        key="",
        url=url,
        final_url=url,
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
        content=b"<html>cached</html>",
        etag='"abc"',
        last_modified=None,
        fetched_at=time.time(),
        cache_version="1",
        market="DE",
        locale=None,
        cache_control="max-age=0",
    )
    entry.key = _cache_key(
        "GET",
        url,
        "DE",
        None,
        {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "identity",
        },
    )
    path = _entry_path(cache_dir, "GET", url, "DE", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry.to_json(), ensure_ascii=False, sort_keys=True), encoding="utf-8")

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["if-none-match"] == '"abc"'
        return httpx.Response(304, headers={"etag": '"abc"'})

    response = asyncio.run(_run_get(handler, url=url, market="DE", config=config))
    assert response.text == "<html>cached</html>"
    assert response.from_cache
    assert len(calls) == 1


def test_normal_responses_use_24_hour_ttl(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    url = "https://www.paypal.com/de/business/paypal-business-fees"
    config = CrawlConfiguration(max_workers=1, request_delay=0, cache_dir=str(cache_dir))
    entry = _CacheEntry(
        key="",
        url=url,
        final_url=url,
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
        content=b"<html>cached</html>",
        etag='"abc"',
        last_modified=None,
        fetched_at=time.time() - 3600,
        cache_version="1",
        market="DE",
        locale=None,
        cache_control=None,
    )
    entry.key = _cache_key(
        "GET",
        url,
        "DE",
        None,
        {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "identity",
        },
    )
    path = _entry_path(cache_dir, "GET", url, "DE", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry.to_json(), ensure_ascii=False, sort_keys=True), encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Fresh cache entry should not trigger network request")

    response = asyncio.run(_run_get(handler, url=url, market="DE", config=config))
    assert response.text == "<html>cached</html>"
    assert response.from_cache


def test_max_age_shorter_than_ttl_is_respected(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    url = "https://www.paypal.com/de/business/paypal-business-fees"
    config = CrawlConfiguration(max_workers=1, request_delay=0, cache_dir=str(cache_dir))
    entry = _CacheEntry(
        key="",
        url=url,
        final_url=url,
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
        content=b"<html>cached</html>",
        etag='"abc"',
        last_modified=None,
        fetched_at=time.time() - 3,
        cache_version="1",
        market="DE",
        locale=None,
        cache_control="max-age=2",
    )
    entry.key = _cache_key(
        "GET",
        url,
        "DE",
        None,
        {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "identity",
        },
    )
    path = _entry_path(cache_dir, "GET", url, "DE", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry.to_json(), ensure_ascii=False, sort_keys=True), encoding="utf-8")

    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.headers["if-none-match"] == '"abc"'
        return httpx.Response(304, headers={"etag": '"abc"'})

    response = asyncio.run(_run_get(handler, url=url, market="DE", config=config))
    assert response.text == "<html>cached</html>"
    assert response.from_cache
    assert len(calls) == 1


def test_failed_revalidation_does_not_corrupt_entry(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    url = "https://www.paypal.com/de/business/paypal-business-fees"
    config = CrawlConfiguration(max_workers=1, request_delay=0, cache_dir=str(cache_dir), max_retries=0)
    entry = _CacheEntry(
        key="",
        url=url,
        final_url=url,
        status_code=200,
        headers={"content-type": "text/html; charset=utf-8"},
        content=b"<html>cached</html>",
        etag='"abc"',
        last_modified=None,
        fetched_at=time.time() - 48 * 3600,
        cache_version="1",
        market="DE",
        locale=None,
        cache_control=None,
    )
    entry.key = _cache_key(
        "GET",
        url,
        "DE",
        None,
        {
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "identity",
        },
    )
    path = _entry_path(cache_dir, "GET", url, "DE", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    original_text = json.dumps(entry.to_json(), ensure_ascii=False, sort_keys=True)
    path.write_text(original_text, encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="error")

    from paypal_fee_crawler.exceptions import NetworkError

    with pytest.raises(NetworkError):
        asyncio.run(_run_get(handler, url=url, market="DE", config=config))

    assert path.read_text(encoding="utf-8") == original_text
