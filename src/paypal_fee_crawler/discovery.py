"""Country and fee-page discovery for PayPal markets."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urljoin

from .cms_context import extract_cms_context, find_global_json_assignments
from .exceptions import (
    CountryDiscoveryError,
    FeePageError,
    ParserError,
    UnsupportedCountryError,
)
from .http import HttpClient, HttpResponse
from .models import CrawlConfiguration, Language, Market

logger = logging.getLogger(__name__)

BOOTSTRAP_MARKETS: list[Market] = [
    Market(
        country_code="DE",
        country_name="Germany",
        region="europe",
        locale="de_DE",
        languages=[Language(code="de", name="Deutsch")],
        url_prefix="https://www.paypal.com/de",
        preferred_language="de",
    ),
    Market(
        country_code="US",
        country_name="United States",
        region="north_america",
        locale="en_US",
        languages=[Language(code="en", name="English")],
        url_prefix="https://www.paypal.com/us",
        preferred_language="en",
    ),
    Market(
        country_code="GB",
        country_name="United Kingdom",
        region="europe",
        locale="en_GB",
        languages=[Language(code="en", name="English")],
        url_prefix="https://www.paypal.com/gb",
        preferred_language="en",
    ),
]


def _find_country_selectors(data: Any) -> list[dict[str, Any]]:
    """Recursively search parsed JSON for CountrySelector structures."""
    found: list[dict[str, Any]] = []
    if isinstance(data, dict):
        if data.get("componentType") == "CountrySelector" or data.get("type") == "CountrySelector":
            found.append(data)
        for value in data.values():
            found.extend(_find_country_selectors(value))
    elif isinstance(data, list):
        for item in data:
            found.extend(_find_country_selectors(item))
    return found


def _normalize_markets(selectors: list[dict[str, Any]]) -> list[Market]:
    """Convert CountrySelector data into normalized Market objects."""
    markets: list[Market] = []
    seen: set[str] = set()
    for selector in selectors:
        regions = selector.get("regions") or selector.get("regionList") or []
        for region in regions:
            region_name = region.get("region") or region.get("name") or "unknown"
            countries = region.get("countries") or region.get("countryList") or []
            for country in countries:
                code = country.get("countryCode") or country.get("code") or country.get("country")
                if not code:
                    continue
                code = code.upper()
                if code in seen:
                    continue
                seen.add(code)
                languages: list[Language] = []
                langs = country.get("languages") or country.get("languageList") or []
                default_locale = country.get("defaultLocale") or country.get("preferredLocale")
                preferred = None
                for lang in langs:
                    lang_code = lang.get("language") or lang.get("code") or lang.get("languageCode")
                    if not lang_code:
                        continue
                    lang_name = lang.get("languageName") or lang.get("name")
                    languages.append(Language(code=lang_code, name=lang_name))
                    if default_locale and default_locale.startswith(lang_code):
                        preferred = lang_code
                if not default_locale and languages:
                    default_locale = f"{code.lower()}_{code.upper()}"
                markets.append(
                    Market(
                        country_code=code,
                        country_name=country.get("countryName") or country.get("name") or code,
                        region=str(region_name).lower().replace(" ", "_"),
                        locale=default_locale,
                        languages=languages,
                        url_prefix=f"https://www.paypal.com/{code.lower()}",
                        preferred_language=preferred,
                    )
                )
    return markets


async def discover_countries(
    http_client: HttpClient,
    config: CrawlConfiguration,
    homepage_url: str = "https://www.paypal.com/de",
) -> list[Market]:
    """Discover PayPal markets from the country selector embedded in a homepage.

    Falls back to the bootstrap list if discovery fails and a fallback is allowed
    by the caller (the caller decides whether to treat fallback as an error).
    """
    try:
        response = await http_client.get(homepage_url)
    except Exception as exc:
        logger.warning("Country discovery request failed: %s", exc)
        raise CountryDiscoveryError(f"Failed to retrieve PayPal homepage: {exc}") from exc

    try:
        cms = extract_cms_context(response.text)
    except ParserError as exc:
        logger.warning("Could not extract CMS context from homepage: %s", exc)
        # Try to find other global JSON assignments that may contain the selector.
        assignments = find_global_json_assignments(response.text, ["__CONFIG__", "__APP_DATA__", "window.__CONFIG__"])
        cms = None
        for data in assignments.values():
            selectors = _find_country_selectors(data)
            if selectors:
                markets = _normalize_markets(selectors)
                if markets:
                    return markets
        raise CountryDiscoveryError(f"No country selector found in homepage: {exc}") from exc

    selectors = _find_country_selectors(cms)
    if not selectors:
        raise CountryDiscoveryError("No CountrySelector component found in CMS context")
    markets = _normalize_markets(selectors)
    if not markets:
        raise CountryDiscoveryError("CountrySelector found but no markets could be extracted")
    return markets


def get_bootstrap_markets() -> list[Market]:
    """Return a small, conservative bootstrap market list."""
    return list(BOOTSTRAP_MARKETS)


def _is_fee_page(page_data: dict[str, Any], response: HttpResponse) -> bool:
    """Validate that a page is a plausible merchant fee page."""
    if response.status_code != 200:
        return False
    content_type = response.headers.get("content-type", "").lower()
    if "text/html" not in content_type and "application/xhtml" not in content_type:
        return False
    page_id = page_data.get("pageId") or page_data.get("pageName") or page_data.get("pageReference", {}).get("id")
    if page_id and "business" in str(page_id).lower():
        return True
    if page_id and "fee" in str(page_id).lower():
        return True

    # Look for fee-related components.
    def _has_fee_component(data: Any) -> bool:
        if isinstance(data, dict):
            ct = data.get("componentType", "")
            if isinstance(ct, str) and ("Fee" in ct or "fee" in str(data).lower()):
                return True
            for value in data.values():
                if _has_fee_component(value):
                    return True
        elif isinstance(data, list):
            for item in data:
                if _has_fee_component(item):
                    return True
        return False

    return _has_fee_component(page_data)


async def discover_fee_page(
    http_client: HttpClient,
    market: Market,
    config: CrawlConfiguration,
) -> str:
    """Find and validate the canonical merchant fee page URL for a market.

    Returns the confirmed URL. Raises FeePageError or UnsupportedCountryError on failure.
    """
    cc = market.country_code.lower()
    candidates = [f"https://www.paypal.com/{cc}/business/paypal-business-fees"]
    for legacy in config.legacy_fee_paths:
        candidates.append(f"https://www.paypal.com/{cc}/{legacy}")

    tested: list[str] = []
    for url in candidates:
        tested.append(url)
        try:
            response = await http_client.get(url)
        except Exception as exc:
            logger.debug("Fee page candidate failed for %s: %s", market.country_code, exc)
            continue
        try:
            page_data = extract_cms_context(response.text)
        except ParserError:
            # Not a valid CMS page; likely a redirect or error page.
            continue
        if _is_fee_page(page_data, response):
            return str(response.url)

    # If the default path failed, try the homepage and search navigation for fee links.
    homepage = f"https://www.paypal.com/{cc}"
    try:
        response = await http_client.get(homepage)
    except Exception as exc:
        raise UnsupportedCountryError(
            f"No fee page found and homepage failed for {market.country_code}: {exc}"
        ) from exc

    try:
        page_data = extract_cms_context(response.text)
    except ParserError as exc:
        raise FeePageError(f"Homepage for {market.country_code} has no valid CMS context: {exc}") from exc

    # Search structurally for fee/business links in navigation.
    def _find_fee_links(data: Any, base: str = "") -> list[str]:
        links: list[str] = []
        if isinstance(data, dict):
            ct = data.get("componentType", "")
            if isinstance(ct, str) and any(token in ct.lower() for token in ["nav", "link", "button"]):
                href = data.get("href") or data.get("url") or data.get("link")
                if (
                    href
                    and isinstance(href, str)
                    and any(token in href.lower() for token in ["fee", "business", "merchant", "seller"])
                ):
                    links.append(urljoin(homepage, href))
            for value in data.values():
                links.extend(_find_fee_links(value, base))
        elif isinstance(data, list):
            for item in data:
                links.extend(_find_fee_links(item, base))
        return links

    fee_links = _find_fee_links(page_data)
    for url in fee_links:
        tested.append(url)
        try:
            response = await http_client.get(url)
        except Exception:  # nosec B112 # noqa: S112
            continue
        try:
            page_data = extract_cms_context(response.text)
        except ParserError:
            continue
        if _is_fee_page(page_data, response):
            return str(response.url)

    raise UnsupportedCountryError(f"No public merchant fee page found for {market.country_code}")


def get_canonical_page_id(page_data: dict[str, Any]) -> str | None:
    """Return a stable page identifier from the CMS context."""
    if isinstance(page_data, dict):
        page_ref = page_data.get("pageReference")
        if isinstance(page_ref, dict):
            return page_ref.get("id") or page_ref.get("pageId")
        return page_data.get("pageId") or page_data.get("pageName")
    return None
