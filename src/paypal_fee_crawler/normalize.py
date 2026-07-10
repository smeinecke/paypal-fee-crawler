"""Normalization helpers for PayPal fee data."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any

from .pricing_tokens import CURRENCY_CODES, _normalize_decimal, _to_canonical_string


def normalize_country_code(value: Any) -> str:
    """Return a validated ISO 3166-1 alpha-2 country code."""
    if not isinstance(value, str) or len(value) != 2:
        raise ValueError(f"Invalid country code: {value!r}")
    return value.upper()


def normalize_currency_code(value: Any) -> str:
    """Return a validated ISO 4217 currency code."""
    if not isinstance(value, str) or len(value) != 3 or value.upper() not in CURRENCY_CODES:
        raise ValueError(f"Invalid or unsupported currency code: {value!r}")
    return value.upper()


def normalize_decimal_string(value: Any) -> str:
    """Normalize a decimal string and return its canonical form."""
    if isinstance(value, Decimal):
        return _to_canonical_string(value)
    if not isinstance(value, str):
        value = str(value)
    try:
        return _to_canonical_string(_normalize_decimal(value))
    except InvalidOperation as exc:
        raise ValueError(f"Invalid decimal value: {value!r}") from exc


def clean_text(value: Any) -> str:
    """Return a clean human-readable string with normalized whitespace."""
    if value is None:
        return ""
    text = str(value)
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    return " ".join(text.split())
