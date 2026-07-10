"""Mapping between PayPal market codes and ISO country codes.

PayPal uses internal/legacy market codes (e.g. ``C2`` for China) that are not
valid ISO 3166-1 alpha-2 codes. This module keeps the raw PayPal market code
separate from any derived ISO country code and provides safe, deterministic
filename generation.
"""

from __future__ import annotations

import re
from typing import Any

# Known PayPal-specific or legacy market codes that do not map directly to ISO
# 3166-1 alpha-2. The mapping is conservative: only include codes where the
# ISO derivation is well documented.
PAYPAL_MARKET_TO_ISO: dict[str, str] = {
    "C2": "CN",  # PayPal uses C2 for China mainland; ISO code is CN.
    "AN": "NL",  # Netherlands Antilles (deprecated); mapped to NL for legacy compatibility.
}


# PayPal market codes that are intentionally treated as their own market even
# though they look like ISO codes. Most PayPal market codes are already ISO
# alpha-2 codes.
PAYPAL_SPECIFIC_MARKETS: set[str] = {"C2"}


# Pattern for a safe PayPal market code (alphanumeric, 2 characters typically).
_MARKET_CODE_RE = re.compile(r"^[A-Z0-9]{2}$")


# Dangerous filename characters and path traversal patterns.
_UNSAFE_PATH_RE = re.compile(r"[\\/<>|:\"*?\x00-\x1f]")


class MarketIdentity:
    """A normalized market identity that preserves the raw PayPal market code."""

    def __init__(self, paypal_market_code: str, country_name: str | None = None) -> None:
        self.paypal_market_code = normalize_paypal_market_code(paypal_market_code)
        self.country_name = country_name or self.paypal_market_code

    @property
    def iso_country_code(self) -> str | None:
        return iso_country_code_for(self.paypal_market_code)

    @property
    def url_slug(self) -> str:
        return self.paypal_market_code.lower()

    @property
    def is_paypal_specific(self) -> bool:
        return self.paypal_market_code in PAYPAL_SPECIFIC_MARKETS

    def safe_filename(self, suffix: str = ".json") -> str:
        return f"{self.url_slug}{suffix}"

    def model_dump(self) -> dict[str, Any]:
        return {
            "paypal_market_code": self.paypal_market_code,
            "iso_country_code": self.iso_country_code,
            "url_slug": self.url_slug,
            "country_name": self.country_name,
            "is_paypal_specific": self.is_paypal_specific,
        }


def normalize_paypal_market_code(value: Any) -> str:
    """Return a normalized PayPal market code.

    Raises:
        ValueError: if the code is not a plausible PayPal market code.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Invalid PayPal market code: {value!r}")
    code = value.strip().upper()
    if not _MARKET_CODE_RE.match(code):
        raise ValueError(f"Invalid PayPal market code: {value!r}")
    return code


def iso_country_code_for(paypal_market_code: str) -> str | None:
    """Return the ISO 3166-1 alpha-2 country code for a PayPal market code.

    Returns the input itself if it already looks like an ISO alpha-2 code,
    unless the code is explicitly known as a PayPal-specific code.
    """
    code = normalize_paypal_market_code(paypal_market_code)
    if code in PAYPAL_MARKET_TO_ISO:
        return PAYPAL_MARKET_TO_ISO[code]
    # If it looks like a regular ISO alpha-2 code, use it directly.
    if code.isalpha() and len(code) == 2:
        return code
    return None


def is_safe_filename(value: str) -> bool:
    """Return whether *value* is safe to use as a filename component."""
    if not value or value.strip() != value:
        return False
    if _UNSAFE_PATH_RE.search(value):
        return False
    return not (value in {".", ".."} or ".." in value)


def safe_filename(value: str, suffix: str = ".json") -> str:
    """Return a deterministic, safe filename for a market code.

    Raises:
        ValueError: if the input would produce an unsafe filename.
    """
    code = normalize_paypal_market_code(value)
    filename = f"{code.lower()}{suffix}"
    if not is_safe_filename(filename):
        raise ValueError(f"Unsafe filename derived from {value!r}")
    return filename
