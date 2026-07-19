"""Normalization helpers for PayPal fee data."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

# ISO 4217 currency codes (selected common set; not exhaustive).
CURRENCY_CODES: frozenset[str] = frozenset(
    {
        "AED",
        "AFN",
        "ALL",
        "AMD",
        "ANG",
        "AOA",
        "ARS",
        "AUD",
        "AWG",
        "AZN",
        "BAM",
        "BBD",
        "BDT",
        "BGN",
        "BHD",
        "BIF",
        "BMD",
        "BND",
        "BOB",
        "BRL",
        "BSD",
        "BTN",
        "BWP",
        "BYN",
        "BZD",
        "CAD",
        "CDF",
        "CHF",
        "CLP",
        "CNY",
        "COP",
        "CRC",
        "CUP",
        "CVE",
        "CZK",
        "DJF",
        "DKK",
        "DOP",
        "DZD",
        "EGP",
        "ERN",
        "ETB",
        "EUR",
        "FJD",
        "FKP",
        "FOK",
        "GBP",
        "GEL",
        "GGP",
        "GHS",
        "GIP",
        "GMD",
        "GNF",
        "GTQ",
        "GYD",
        "HKD",
        "HNL",
        "HRK",
        "HTG",
        "HUF",
        "IDR",
        "ILS",
        "IMP",
        "INR",
        "IQD",
        "IRR",
        "ISK",
        "JEP",
        "JMD",
        "JOD",
        "JPY",
        "KES",
        "KGS",
        "KHR",
        "KID",
        "KMF",
        "KRW",
        "KWD",
        "KYD",
        "KZT",
        "LAK",
        "LBP",
        "LKR",
        "LRD",
        "LSL",
        "LYD",
        "MAD",
        "MDL",
        "MGA",
        "MKD",
        "MMK",
        "MNT",
        "MOP",
        "MRU",
        "MUR",
        "MVR",
        "MWK",
        "MXN",
        "MYR",
        "MZN",
        "NAD",
        "NGN",
        "NIO",
        "NOK",
        "NPR",
        "NZD",
        "OMR",
        "PAB",
        "PEN",
        "PGK",
        "PHP",
        "PKR",
        "PLN",
        "PYG",
        "QAR",
        "RON",
        "RSD",
        "RUB",
        "RWF",
        "SAR",
        "SBD",
        "SCR",
        "SDG",
        "SEK",
        "SGD",
        "SHP",
        "SLE",
        "SLL",
        "SOS",
        "SRD",
        "SSP",
        "STN",
        "SYP",
        "SZL",
        "THB",
        "TJS",
        "TMT",
        "TND",
        "TOP",
        "TRY",
        "TTD",
        "TVD",
        "TWD",
        "TZS",
        "UAH",
        "UGX",
        "USD",
        "UYU",
        "UZS",
        "VED",
        "VES",
        "VND",
        "VUV",
        "WST",
        "XAF",
        "XCD",
        "XOF",
        "XPF",
        "YER",
        "ZAR",
        "ZMW",
        "ZWL",
    }
)


def _normalize_decimal(value: str) -> Decimal:
    """Parse a decimal string that may use comma or point as decimal separator."""
    cleaned = value.replace("\u00a0", "").replace(" ", "").replace("\u202f", "")
    # Strip currency symbols, percent signs and other decoration while keeping
    # digits, separators and a leading sign.
    cleaned = re.sub(r"[^\d+.,\-]", "", cleaned)
    has_dot = "." in cleaned
    has_comma = "," in cleaned
    if not has_dot and not has_comma:
        return Decimal(cleaned)

    if has_dot and not has_comma:
        # "1234.56" or "1.234.56" (unusual). Use dot as decimal; remove thousands dots.
        if cleaned.count(".") > 1:
            cleaned = cleaned.replace(".", "")
        return Decimal(cleaned)

    if has_comma and not has_dot:
        if cleaned.count(",") == 1:
            cleaned = cleaned.replace(",", ".")
        else:
            # Keep the last comma as decimal separator, remove the rest.
            parts = cleaned.split(",")
            cleaned = "".join(parts[:-1]) + "." + parts[-1]
        return Decimal(cleaned)

    # Both comma and dot present. Use the last separator as the decimal marker.
    last_dot = cleaned.rfind(".")
    last_comma = cleaned.rfind(",")
    cleaned = cleaned.replace(",", "") if last_dot > last_comma else cleaned.replace(".", "").replace(",", ".")
    return Decimal(cleaned)


def _to_canonical_string(value: Decimal) -> str:
    """Return a canonical decimal string without exponent notation."""
    normalized = value.normalize()
    return f"{normalized:f}"


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
