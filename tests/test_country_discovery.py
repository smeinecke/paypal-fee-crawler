"""Tests for country discovery and market manifest handling."""

from __future__ import annotations

from paypal_fee_crawler.discovery import _find_country_selectors, _normalize_markets, get_bootstrap_markets


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


def test_bootstrap_markets() -> None:
    markets = get_bootstrap_markets()
    codes = {m.country_code for m in markets}
    assert {"DE", "US", "GB"}.issubset(codes)
