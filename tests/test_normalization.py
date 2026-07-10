"""Tests for normalization utilities."""

from __future__ import annotations

import pytest

from paypal_fee_crawler.normalize import (
    clean_text,
    normalize_country_code,
    normalize_currency_code,
    normalize_decimal_string,
)


def test_normalize_country_code() -> None:
    assert normalize_country_code("de") == "DE"
    assert normalize_country_code("DE") == "DE"
    with pytest.raises(ValueError):
        normalize_country_code("")


def test_normalize_currency_code() -> None:
    assert normalize_currency_code("eur") == "EUR"
    assert normalize_currency_code("EUR") == "EUR"
    with pytest.raises(ValueError):
        normalize_currency_code("XYZ")


def test_normalize_decimal_string() -> None:
    assert normalize_decimal_string("2,99") == "2.99"
    assert normalize_decimal_string("1,234.56") == "1234.56"
    assert normalize_decimal_string("1.234,56") == "1234.56"
    with pytest.raises(ValueError):
        normalize_decimal_string("abc")


def test_clean_text() -> None:
    assert clean_text("  Hello\u00a0World  ") == "Hello World"
