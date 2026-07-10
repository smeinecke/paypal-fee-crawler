"""Tests for the end-to-end crawler using mocked network responses."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from paypal_fee_crawler.cms_context import extract_cms_context
from paypal_fee_crawler.crawler import Crawler
from paypal_fee_crawler.exceptions import CountryDiscoveryError
from paypal_fee_crawler.http import HttpClient, HttpResponse
from paypal_fee_crawler.models import CrawlConfiguration


def _load_fixture(name: str) -> str:
    return (Path(__file__).parent / "fixtures" / name).read_text(encoding="utf-8")


async def _fake_get(self: HttpClient, url: str, **kwargs: Any) -> HttpResponse:
    # Return the homepage or a fee page fixture based on the URL.
    if url.endswith("/de") or url.endswith("/de/"):
        html = _load_fixture("de.html")
    elif url.endswith("/us/business/paypal-business-fees"):
        html = _load_fixture("us.html")
    elif url.endswith("/gb/business/paypal-business-fees"):
        html = _load_fixture("gb.html")
    elif url.endswith("/de/business/paypal-business-fees"):
        html = _load_fixture("de.html")
    else:
        html = _load_fixture("de.html")
    return HttpResponse(
        url=url,
        status_code=200,
        content=html.encode("utf-8"),
        text=html,
        headers={"content-type": "text/html; charset=utf-8"},
    )


def _run_crawl(config: CrawlConfiguration) -> Any:
    async def _run() -> Any:
        async with Crawler(config) as crawler:
            with patch.object(HttpClient, "get", _fake_get):
                return await crawler.crawl()

    return asyncio.run(_run())


def test_crawler_discover_with_bootstrap(tmp_path: Path) -> None:
    config = CrawlConfiguration(output_dir=str(tmp_path), countries=["DE"], request_delay=0, max_workers=1)
    report = _run_crawl(config)
    assert report.countries_processed == 1
    assert not report.countries_failed
    assert (tmp_path / "json" / "de.json").exists()


def test_crawler_crawl_multiple(tmp_path: Path) -> None:
    config = CrawlConfiguration(output_dir=str(tmp_path), countries=["DE", "US", "GB"], request_delay=0, max_workers=1)
    report = _run_crawl(config)
    assert report.countries_processed == 3
    assert (tmp_path / "json" / "index.json").exists()
    assert (tmp_path / "json" / "core-fees.json").exists()


def test_crawler_crawl_no_countries(tmp_path: Path) -> None:
    config = CrawlConfiguration(output_dir=str(tmp_path), countries=["XX"], request_delay=0, max_workers=1)
    with pytest.raises(CountryDiscoveryError):
        _run_crawl(config)


def test_crawler_extracts_metadata_from_real_cms() -> None:
    html = _load_fixture("paypal-de-real.html")
    cms = extract_cms_context(html)
    crawler = Crawler(CrawlConfiguration())
    assert crawler._extract_page_title("<html></html>", cms) is not None
    assert crawler._extract_cms_updated_at(cms) is not None
    assert crawler._extract_locale(cms) is not None
    assert crawler._extract_update_date(cms, []) is not None


def test_crawler_extracts_title_from_html() -> None:
    crawler = Crawler(CrawlConfiguration())
    assert crawler._extract_page_title("<title>Custom Title</title>", {}) == "Custom Title"
