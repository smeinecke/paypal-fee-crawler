"""Tests for country discovery and fee-page discovery."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from paypal_fee_crawler.cms_context import extract_cms_context
from paypal_fee_crawler.discovery import (
    _find_country_selectors,
    _is_fee_page,
    _normalize_markets,
    discover_countries,
    discover_fee_page,
    get_bootstrap_markets,
    get_canonical_page_id,
)
from paypal_fee_crawler.exceptions import UnsupportedCountryError
from paypal_fee_crawler.http import HttpClient, HttpResponse
from paypal_fee_crawler.models import CrawlConfiguration, Market


def test_find_country_selector_nested() -> None:
    data = {"nav": {"items": [{"componentType": "CountrySelector", "regions": []}]}}
    selectors = _find_country_selectors(data)
    assert len(selectors) == 1


def test_normalize_markets_basic() -> None:
    selector = {
        "regions": [
            {
                "region": "Europe",
                "countries": [
                    {
                        "countryCode": "DE",
                        "countryName": "Germany",
                        "defaultLocale": "de_DE",
                        "languages": [{"language": "de", "languageName": "Deutsch"}],
                    }
                ],
            }
        ]
    }
    markets = _normalize_markets([selector])
    assert len(markets) == 1
    assert markets[0].country_code == "DE"
    assert markets[0].locale == "de_DE"
    assert markets[0].url_prefix == "https://www.paypal.com/de"


def test_bootstrap_markets() -> None:
    markets = get_bootstrap_markets()
    codes = {m.country_code for m in markets}
    assert {"DE", "US", "GB"}.issubset(codes)


def test_is_fee_page_valid(de_html: str) -> None:
    cms = extract_cms_context(de_html)
    resp = HttpResponse(
        url="https://www.paypal.com/de/business/paypal-business-fees",
        status_code=200,
        content=de_html.encode(),
        text=de_html,
        headers={"content-type": "text/html"},
    )
    assert _is_fee_page(cms, resp) is True


def test_is_fee_page_bad_content_type() -> None:
    resp = HttpResponse(
        url="https://www.paypal.com/de/business/paypal-business-fees",
        status_code=200,
        content=b"",
        text="",
        headers={"content-type": "application/json"},
    )
    assert _is_fee_page({}, resp) is False


def test_is_fee_page_bad_status() -> None:
    resp = HttpResponse(
        url="https://www.paypal.com/de/business/paypal-business-fees",
        status_code=404,
        content=b"",
        text="",
        headers={"content-type": "text/html"},
    )
    assert _is_fee_page({}, resp) is False


def _load_fixture(name: str) -> str:
    return (Path(__file__).parent / "fixtures" / name).read_text(encoding="utf-8")


async def _fake_homepage(self: HttpClient, url: str, **kwargs: Any) -> HttpResponse:
    html = _load_fixture("de.html")
    return HttpResponse(
        url=url,
        status_code=200,
        content=html.encode("utf-8"),
        text=html,
        headers={"content-type": "text/html"},
    )


def test_discover_countries_from_homepage() -> None:
    async def _run() -> list[Market]:
        async with HttpClient(CrawlConfiguration(max_workers=1, request_delay=0)) as client:
            with patch.object(HttpClient, "get", _fake_homepage):
                return await discover_countries(client, CrawlConfiguration())

    markets = asyncio.run(_run())
    codes = {m.country_code for m in markets}
    assert {"DE", "US", "GB"}.issubset(codes)


def test_discover_fee_page_for_de() -> None:
    async def _run() -> str:
        async with HttpClient(CrawlConfiguration(max_workers=1, request_delay=0)) as client:
            with patch.object(HttpClient, "get", _fake_homepage):
                return await discover_fee_page(
                    client,
                    Market(paypal_market_code="DE", iso_country_code="DE", country_name="Germany"),
                    CrawlConfiguration(),
                )

    url = asyncio.run(_run())
    assert "paypal.com/de" in url


def test_discover_fee_page_unsupported() -> None:
    html = _load_fixture("de.html")

    async def _fake_404(self: HttpClient, url: str, **kwargs: Any) -> HttpResponse:
        return HttpResponse(
            url=url,
            status_code=404,
            content=html.encode("utf-8"),
            text=html,
            headers={"content-type": "text/html"},
        )

    async def _run() -> str:
        async with HttpClient(CrawlConfiguration(max_workers=1, request_delay=0)) as client:
            with patch.object(HttpClient, "get", _fake_404):
                return await discover_fee_page(
                    client,
                    Market(paypal_market_code="XX", iso_country_code=None, country_name="Unknown"),
                    CrawlConfiguration(),
                )

    with pytest.raises(UnsupportedCountryError):
        asyncio.run(_run())


def test_get_canonical_page_id_from_real_cms() -> None:
    html = _load_fixture("paypal-de-real.html")
    cms = extract_cms_context(html)
    page_id = get_canonical_page_id(cms)
    assert page_id == "business/paypal-business-fees"


def test_get_canonical_page_id_legacy_top_level() -> None:
    assert get_canonical_page_id({"pageId": "test/page"}) == "test/page"
    assert get_canonical_page_id({"pageName": "test/page"}) == "test/page"


def test_get_canonical_page_id_missing() -> None:
    assert get_canonical_page_id({}) is None
    assert get_canonical_page_id(None) is None  # type: ignore[arg-type]


def test_normalize_markets_locale_fallback() -> None:
    selector = {
        "regions": [
            {
                "region": "Europe",
                "countries": [
                    {
                        "countryCode": "FR",
                        "countryName": "France",
                        "languages": [{"language": "fr", "languageName": "French"}],
                    }
                ],
            }
        ]
    }
    markets = _normalize_markets([selector])
    assert len(markets) == 1
    assert markets[0].country_code == "FR"
    assert markets[0].locale is None
    assert markets[0].preferred_language == "fr"
