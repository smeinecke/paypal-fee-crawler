"""Shared crawler constants that multiple modules need without creating cycles."""

from __future__ import annotations

# Output roots that the crawler owns and may modify.
MANAGED_ROOTS = ("json", "meta", "schemas", "change-report.json")

# Full set of managed paths, including generated README updates.
MANAGED_PATHS = (*MANAGED_ROOTS, "README.md")

# PayPal web properties used by the crawler.
PAYPAL_BASE_URL = "https://www.paypal.com"
PAYPAL_HOST_ALLOWLIST = ("www.paypal.com", "www.paypalobjects.com")

# Default country/market code used for discovery.
DEFAULT_DISCOVERY_MARKET = "de"

# Default fee page path template and legacy alternatives.
FEE_PAGE_PATH_TEMPLATE = "{base}/{market}/business/paypal-business-fees"
LEGACY_FEE_PAGE_PATHS = (
    "business/paypal-business-fees",
    "merchant/paypal-merchant-fees",
    "business/fees",
    "seller-fees",
)

# Default discovery entry point.  This is the DE fee page which hosts the market
# selector used to discover all supported countries.
DEFAULT_DISCOVERY_URL = FEE_PAGE_PATH_TEMPLATE.format(base=PAYPAL_BASE_URL, market=DEFAULT_DISCOVERY_MARKET)

# Classifier metadata.  The mode is deprecated and kept only for compatibility
# with existing output sidecars; the version is the authoritative classifier
# label.
CLASSIFIER_MODE = "rules"
CLASSIFIER_VERSION = "rules-v1"
CLASSIFIER_METADATA = (CLASSIFIER_MODE, CLASSIFIER_VERSION)
