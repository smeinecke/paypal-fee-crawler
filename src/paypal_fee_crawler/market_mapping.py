"""Mapping between PayPal market codes and ISO country codes.

PayPal uses internal/legacy market codes (e.g. ``C2`` for China) that are not
valid ISO 3166-1 alpha-2 codes. This module keeps the raw PayPal market code
separate from any derived ISO country code.
"""

from __future__ import annotations

import re

# Known PayPal-specific or legacy market codes that do not map directly to ISO
# 3166-1 alpha-2. The mapping is conservative: only include codes where the
# ISO derivation is well documented.
PAYPAL_MARKET_TO_ISO: dict[str, str] = {
    "C2": "CN",  # PayPal uses C2 for China mainland; ISO code is CN.
    "AN": "NL",  # Netherlands Antilles (deprecated); mapped to NL for legacy compatibility.
}


# Pattern for a safe PayPal market code (alphanumeric, 2 characters typically).
_MARKET_CODE_RE = re.compile(r"^[A-Z0-9]{2}$")


def normalize_paypal_market_code(value: str) -> str:
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
