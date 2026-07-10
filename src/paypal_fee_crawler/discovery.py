"""Country and fee-page discovery for PayPal markets."""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urljoin, urlparse

from .cms_context import (
    ALLOWLISTED_GLOBAL_CONTEXTS,
    extract_all_contexts,
    extract_cms_context,
    find_global_json_assignments,
)
from .exceptions import (
    AccessChallengeError,
    ContentSecurityError,
    CountryDiscoveryError,
    FeePageError,
    FeePageStructureError,
    NetworkError,
    ParserError,
    PermanentHttpError,
    PermanentNetworkError,
    RateLimitError,
    TransientNetworkError,
    UnsupportedCountryError,
)
from .http import HttpClient, HttpResponse
from .market_mapping import iso_country_code_for
from .models import CrawlConfiguration, Language, Market

logger = logging.getLogger(__name__)

# Markets that don't have their own fee page but redirect to another market's
# fee page.  PayPal serves these markets from the alias target's page, so we
# use that URL directly instead of following a redirect to a non-fee page.
FEE_PAGE_ALIASES: dict[str, str] = {
    "GI": "https://www.paypal.com/uk/business/paypal-business-fees",
    "GG": "https://www.paypal.com/uk/business/paypal-business-fees",
    "IM": "https://www.paypal.com/uk/business/paypal-business-fees",
    "JE": "https://www.paypal.com/uk/business/paypal-business-fees",
}

BOOTSTRAP_MARKETS: list[Market] = [
    Market(
        paypal_market_code="DE",
        iso_country_code="DE",
        country_name="Germany",
        region="europe",
        locale="de_DE",
        languages=[Language(code="de", name="Deutsch")],
        url_prefix="https://www.paypal.com/de",
        preferred_language="de",
    ),
    Market(
        paypal_market_code="US",
        iso_country_code="US",
        country_name="United States",
        region="north_america",
        locale="en_US",
        languages=[Language(code="en", name="English")],
        url_prefix="https://www.paypal.com/us",
        preferred_language="en",
    ),
    Market(
        paypal_market_code="GB",
        iso_country_code="GB",
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
    """Convert CountrySelector data into normalized Market objects.

    PayPal market codes are preserved; ISO country codes are derived via the
    conservative mapping table.
    """
    markets: list[Market] = []
    seen: set[str] = set()
    for selector in selectors:
        regions = selector.get("regions") or selector.get("regionList") or []
        for region in regions:
            region_name = region.get("region") or region.get("name") or "unknown"
            countries = region.get("countries") or region.get("countryList") or []
            for country in countries:
                paypal_code = country.get("countryCode") or country.get("code") or country.get("country")
                if not paypal_code:
                    continue
                paypal_code = paypal_code.upper()
                if paypal_code in seen:
                    continue
                seen.add(paypal_code)
                iso_code = iso_country_code_for(paypal_code)
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
                # Avoid invented locale strings; only keep real PayPal locale data.
                if not default_locale and languages:
                    preferred = preferred or languages[0].code
                markets.append(
                    Market(
                        paypal_market_code=paypal_code,
                        iso_country_code=iso_code,
                        country_name=country.get("countryName") or country.get("name") or paypal_code,
                        region=str(region_name).lower().replace(" ", "_"),
                        locale=default_locale,
                        languages=languages,
                        url_prefix=f"https://www.paypal.com/{paypal_code.lower()}",
                        preferred_language=preferred,
                    )
                )
    return markets


async def discover_countries(
    http_client: HttpClient,
    config: CrawlConfiguration,
    discovery_url: str = "https://www.paypal.com/de/business/paypal-business-fees",
) -> list[Market]:
    """Discover PayPal markets from the country selector embedded in a page.

    The PayPal homepage redirect no longer exposes the CMS render context, so
    the country selector is read from a fee page (which still contains the
    CMS context). The selector is searched across all allowlisted global JSON
    contexts (CMS, footer, header) independently. Falls back to the bootstrap
    list if discovery fails and a fallback is allowed by the caller.
    """
    try:
        response = await http_client.get(discovery_url)
    except (NetworkError, ParserError) as exc:
        logger.warning("Country discovery request failed: %s", exc)
        raise CountryDiscoveryError(f"Failed to retrieve PayPal discovery page: {exc}") from exc

    try:
        all_contexts = extract_all_contexts(response.text)
    except ParserError as exc:
        logger.warning("Could not extract CMS context from discovery page: %s", exc)
        # Try to find any global JSON assignments that may contain the selector.
        assignments = find_global_json_assignments(response.text, list(ALLOWLISTED_GLOBAL_CONTEXTS))
        for data in assignments.values():
            selectors = _find_country_selectors(data)
            if selectors:
                markets = _normalize_markets(selectors)
                if markets:
                    return markets
        raise CountryDiscoveryError(f"No country selector found in discovery page: {exc}") from exc

    contexts = all_contexts["contexts"]
    # Search each parsed context independently; the selector may live only in the
    # footer or header context on real pages.
    for _name, context in contexts.items():
        selectors = _find_country_selectors(context)
        if selectors:
            markets = _normalize_markets(selectors)
            if markets:
                return markets

    # Last-resort fallback: try the CMS context itself.
    cms = contexts.get("window.__CMS_ENGINE_RENDER_CONTEXT__")
    if cms is not None:
        selectors = _find_country_selectors(cms)
        if not selectors:
            raise CountryDiscoveryError("No CountrySelector component found in any global context")
        markets = _normalize_markets(selectors)
        if not markets:
            raise CountryDiscoveryError("CountrySelector found but no markets could be extracted")
        return markets

    raise CountryDiscoveryError("No country selector found in any global context")


def get_bootstrap_markets() -> list[Market]:
    """Return a small, conservative bootstrap market list."""
    return list(BOOTSTRAP_MARKETS)


def _is_html_response(response: HttpResponse) -> bool:
    """Return True if the response looks like an HTML document."""
    content_type = response.headers.get("content-type", "").lower()
    return "text/html" in content_type or "application/xhtml" in content_type


def _is_fee_url_path(url: str) -> bool:
    """Return True if the URL path is a known PayPal fee page path.

    PayPal no longer embeds the CMS context on the public fee pages, so the
    canonical fee URLs are accepted as valid when they return HTML even without
    a CMS render context.
    """
    path = urlparse(url).path.lower().rstrip("/")
    return path.endswith("/business/paypal-business-fees") or path.endswith("/business/fees")


def _is_fee_page(page_data: dict[str, Any], response: HttpResponse) -> bool:
    """Validate that a page is a plausible merchant fee page."""
    if response.status_code != 200:
        return False
    if not _is_html_response(response):
        return False

    page_id = get_canonical_page_id(page_data)
    if page_id and "business" in str(page_id).lower():
        return True
    if page_id and "fee" in str(page_id).lower():
        return True

    # Look for fee-related components.
    def _has_fee_component(data: Any) -> bool:
        if isinstance(data, dict):
            ct = data.get("componentType", "")
            if isinstance(ct, str) and "Fee" in ct:
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


def _find_fee_links(data: Any, homepage: str) -> list[str]:
    """Search structurally for fee/business/merchant/seller links in CMS navigation."""
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
            links.extend(_find_fee_links(value, homepage))
    elif isinstance(data, list):
        for item in data:
            links.extend(_find_fee_links(item, homepage))
    return links


async def discover_fee_page(
    http_client: HttpClient,
    market: Market,
    config: CrawlConfiguration,
) -> str:
    """Find and validate the canonical merchant fee page URL for a market.

    Returns the confirmed URL. Raises FeePageError or UnsupportedCountryError on failure.
    Transient failures, access challenges, and parser/structure failures are never
    converted into unsupported-country records.
    """
    code = market.paypal_market_code
    slug = market.url_slug

    # Check if this market uses another market's fee page (redirect alias).
    alias_url = FEE_PAGE_ALIASES.get(code)
    if alias_url:
        try:
            response = await http_client.get(alias_url)
            try:
                page_data = extract_cms_context(response.text)
                if _is_fee_page(page_data, response):
                    return str(response.url)
            except ParserError:
                # No CMS context — check if it's still a valid fee-page URL.
                response_url = str(response.url)
                if _is_html_response(response) and _is_fee_url_path(response_url):
                    return response_url
        except (NetworkError, ParserError) as exc:
            logger.debug("Alias fee page fetch failed for %s: %s", code, exc)
        # Fall through to normal discovery if alias didn't work.

    candidates = [f"https://www.paypal.com/{slug}/business/paypal-business-fees"]
    for legacy in config.legacy_fee_paths:
        candidates.append(f"https://www.paypal.com/{slug}/{legacy}")

    tested: list[str] = []
    transient_failure = False
    parser_failure = False

    for index, url in enumerate(candidates):
        tested.append(url)
        try:
            response = await http_client.get(url)
        except (AccessChallengeError, ContentSecurityError) as exc:
            logger.debug("Access/security failure on candidate for %s: %s", code, exc)
            transient_failure = True
            continue
        except PermanentHttpError as exc:
            if exc.status_code == 404:
                # Confirmed not-found for this candidate path; keep trying others.
                continue
            logger.debug("Permanent HTTP error for candidate %s: %s", code, exc)
            transient_failure = True
            continue
        except PermanentNetworkError as exc:
            logger.debug("Permanent network error for candidate %s: %s", code, exc)
            transient_failure = True
            continue
        except (TransientNetworkError, RateLimitError, NetworkError) as exc:
            logger.debug("Transient candidate failure for %s: %s", code, exc)
            transient_failure = True
            continue
        except ParserError as exc:
            logger.debug("Parser error on candidate for %s: %s", code, exc)
            parser_failure = True
            continue

        try:
            page_data = extract_cms_context(response.text)
        except ParserError as exc:
            # PayPal no longer embeds the CMS context on the public fee pages. If the
            # primary canonical fee-page URL returns HTML but has no CMS context, trust it
            # as long as the page still belongs to this market (no cross-country redirect).
            response_url = str(response.url)
            if (
                index == 0
                and _is_html_response(response)
                and _is_fee_url_path(response_url)
                and urlparse(response_url).path.lower().startswith(f"/{slug}/")
            ):
                return response_url
            logger.debug("Parser error extracting CMS from candidate for %s: %s", code, exc)
            parser_failure = True
            continue
        if _is_fee_page(page_data, response):
            return str(response.url)

    # If the default path failed, try the homepage and search navigation for fee links.
    homepage = f"https://www.paypal.com/{slug}"
    tested.append(homepage)
    try:
        response = await http_client.get(homepage)
    except (AccessChallengeError, ContentSecurityError) as exc:
        raise FeePageError(f"Homepage access challenge for {code}: {exc}") from exc
    except PermanentHttpError as exc:
        raise FeePageError(f"Homepage returned HTTP {exc.status_code} for {code}") from exc
    except PermanentNetworkError as exc:
        raise FeePageError(f"Homepage access denied for {code}: {exc}") from exc
    except (TransientNetworkError, RateLimitError, NetworkError) as exc:
        raise FeePageError(f"Homepage request failed for {code}: {exc}") from exc

    try:
        page_data = extract_cms_context(response.text)
    except ParserError as exc:
        raise FeePageStructureError(f"Homepage for {code} has no valid CMS context: {exc}") from exc

    fee_links = _find_fee_links(page_data, homepage)
    for url in fee_links:
        tested.append(url)
        try:
            response = await http_client.get(url)
        except (AccessChallengeError, ContentSecurityError) as exc:
            logger.debug("Access/security failure on fee link for %s: %s", code, exc)
            transient_failure = True
            continue
        except PermanentHttpError as exc:
            if exc.status_code == 404:
                continue
            logger.debug("Permanent HTTP error on fee link for %s: %s", code, exc)
            transient_failure = True
            continue
        except PermanentNetworkError as exc:
            logger.debug("Permanent network error on fee link for %s: %s", code, exc)
            transient_failure = True
            continue
        except (TransientNetworkError, RateLimitError, NetworkError) as exc:
            logger.debug("Transient failure on fee link for %s: %s", code, exc)
            transient_failure = True
            continue
        except ParserError as exc:
            logger.debug("Parser error on fee link for %s: %s", code, exc)
            parser_failure = True
            continue

        try:
            page_data = extract_cms_context(response.text)
        except ParserError as exc:
            logger.debug("Parser error extracting CMS from fee link for %s: %s", code, exc)
            parser_failure = True
            continue
        if _is_fee_page(page_data, response):
            return str(response.url)

    if transient_failure or parser_failure:
        raise FeePageError(
            f"Could not confirm a fee page for {code}; "
            f"transient_failure={transient_failure}, parser_failure={parser_failure}"
        )

    raise UnsupportedCountryError(
        f"No public merchant fee page found for {code}",
        tested_urls=tested,
    )


def _get_nested(data: Any, *keys: str) -> Any:
    """Safely walk a nested dict path."""
    for key in keys:
        if not isinstance(data, dict):
            return None
        data = data.get(key)
    return data


def get_canonical_page_id(page_data: dict[str, Any]) -> str | None:
    """Return a stable page identifier from the CMS context.

    Real PayPal pages place the page ID inside ``pageModel.pageReference.id``,
    ``pageContext.additionalContext.clientSideContext.pageId``, or the URI path
    inside ``pageContext.cmsEngineContext.environment.pageURI``.
    """
    if not isinstance(page_data, dict):
        return None

    # Top-level pageId/pageName (legacy/generated fixtures).
    page_id = page_data.get("pageId") or page_data.get("pageName")
    if page_id:
        return str(page_id)

    # pageModel.metadata.pageId/pageId
    page_model = page_data.get("pageModel")
    if isinstance(page_model, dict):
        metadata = page_model.get("metadata")
        if isinstance(metadata, dict):
            page_id = metadata.get("pageId")
            if page_id:
                return str(page_id)
        page_ref = page_model.get("pageReference")
        if isinstance(page_ref, dict):
            page_id = page_ref.get("pageId") or page_ref.get("id")
            if page_id:
                return str(page_id)

    # pageContext.additionalContext.clientSideContext.pageId
    page_id = _get_nested(page_data, "pageContext", "additionalContext", "clientSideContext", "pageId")
    if page_id:
        return str(page_id)

    # pageContext.cmsEngineContext.environment.pageURI/pagePath
    env = _get_nested(page_data, "pageContext", "cmsEngineContext", "environment")
    if isinstance(env, dict):
        page_uri = env.get("pageURI") or env.get("pagePath")
        if page_uri:
            return str(page_uri)

    # Legacy pageContext.environment.pageURI
    page_context = page_data.get("pageContext")
    if isinstance(page_context, dict):
        env = page_context.get("environment") or {}
        if isinstance(env, dict):
            page_uri = env.get("pageURI") or env.get("pagePath")
            if page_uri:
                return str(page_uri)
    return None
