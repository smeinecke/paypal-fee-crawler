"""Main crawl orchestration for the PayPal fee crawler."""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from .classify import classify_legacy, classify_structural
from .cms_context import extract_cms_context
from .components import ComponentsExtractor, iter_components
from .discovery import discover_countries, discover_fee_page, get_bootstrap_markets, get_canonical_page_id
from .exceptions import (
    CountryDiscoveryError,
    ExitCode,
    FeePageError,
    NetworkError,
    ParserError,
    RegressionError,
    UnsupportedCountryError,
)
from .exceptions import (
    ValidationError as CrawlerValidationError,
)
from .html_tables import extract_html_locale, extract_html_pdf_url, extract_html_tables
from .http import CachedSource, HttpClient
from .models import (
    ClassifierMode,
    CountryOutput,
    CrawlConfiguration,
    CrawlReport,
    Market,
    ParserWarning,
    Source,
    UnsupportedCountry,
)
from .output import OutputPublisher
from .regression import PreviousState, RegressionLimits, check_regression, enforce_regression
from .validation import validate_all_output, validate_country_output

logger = logging.getLogger(__name__)


class Crawler:
    """End-to-end crawler for PayPal merchant fee data."""

    def __init__(self, config: CrawlConfiguration) -> None:
        self.config = config
        self.http_client = HttpClient(config)
        self.warnings: list[ParserWarning] = []
        self._previous_state: PreviousState | None = None

    async def __aenter__(self) -> Crawler:
        return self

    async def __aexit__(self, *_exc: Any) -> None:
        await self.http_client.close()

    async def discover(self) -> list[Market]:
        """Discover PayPal markets."""
        try:
            markets = await discover_countries(
                self.http_client,
                self.config,
                discovery_url="https://www.paypal.com/de/business/paypal-business-fees",
            )
            if not markets:
                raise CountryDiscoveryError("No markets discovered")
            return markets
        except CountryDiscoveryError as exc:
            # Try to load the previous manifest from the configured or default location.
            manifest_path = self.config.country_manifest_path
            if not manifest_path and self.config.output_dir:
                manifest_path = str(Path(self.config.output_dir) / "meta" / "countries.json")
            if manifest_path:
                previous = self._load_previous_manifest(manifest_path)
                if previous:
                    logger.warning("Discovery failed; using previous manifest: %s", exc)
                    return previous.markets
            logger.warning("Discovery failed; falling back to bootstrap list: %s", exc)
            return get_bootstrap_markets()

    def _load_previous_manifest(self, manifest_path: str | None = None) -> Any | None:
        path = Path(manifest_path) if manifest_path else None
        if not path or not path.exists():
            return None
        try:
            from .models import CountryManifest

            return CountryManifest.model_validate_json(path.read_text())
        except Exception as exc:
            logger.warning("Could not load previous manifest: %s", exc)
            return None

    def _load_previous_country_output(self, market: Market) -> CountryOutput | None:
        """Load the previous country output from disk, if available and valid."""
        if not self.config.output_dir:
            return None
        prev_path = Path(self.config.output_dir) / "json" / f"{market.url_slug}.json"
        if not prev_path.exists():
            return None
        try:
            return CountryOutput.model_validate_json(prev_path.read_text())
        except Exception as exc:
            logger.debug("Could not load previous output for %s: %s", market.paypal_market_code, exc)
            return None

    def _stable_timestamp(self) -> str | None:
        """Return a deterministic timestamp for this run.

        The output timestamp is taken from the configuration when provided; otherwise
        None is returned. This prevents runtime clock values from entering canonical
        output by default.
        """
        return self.config.timestamp or None

    def _extract_update_date(self, cms: dict[str, Any], sections: list[Any]) -> str | None:
        """Find an explicit update date in the CMS sections or metadata."""
        # Try explicit metadata at the top level first.
        for key in ("pageUpdatedAt", "lastModified", "updatedAt", "publishedAt"):
            value = cms.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        # Real PayPal pages use pageModel.update_time.
        page_model = cms.get("pageModel")
        if isinstance(page_model, dict):
            for key in ("update_time", "updatedAt", "lastModified"):
                value = page_model.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        # Search rendered section text for locale-specific update phrases.
        phrases = (
            r"Letzte\s+Aktualisierung\s*:\s*([^\n<]+)",
            r"Last\s+updated\s*:\s*([^\n<]+)",
            r"Updated\s*:\s*([^\n<]+)",
            r"Stand\s*:\s*([^\n<]+)",
            r"(?:Aktualisiert|Updated)\s+(?:am\s+)?([^\n<]+)",
        )
        for section in sections:
            text = ""
            if isinstance(section, dict):
                text = section.get("body") or section.get("heading") or ""
            else:
                text = f"{section.heading or ''} {section.body or ''}"
            if not isinstance(text, str):
                text = str(text)
            for phrase in phrases:
                match = re.search(phrase, text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        return None

    def _extract_pdf_url(self, cms: dict[str, Any]) -> str | None:
        """Find a printable PDF fee schedule link in the CMS components."""
        for component in iter_components(cms):
            if not isinstance(component, dict):
                continue
            ct = component.get("componentType") or component.get("type") or ""
            if ct not in {"Button", "Link", "CTALink", "CTACollection"}:
                continue
            for key in ("url", "mobileUrl", "href", "link"):
                value = component.get(key)
                if isinstance(value, str) and "pdf" in value.lower():
                    return value
            content = component.get("content") or component.get("fields") or {}
            if isinstance(content, dict):
                for key in ("url", "mobileUrl", "href", "link"):
                    value = content.get(key)
                    if isinstance(value, str) and "pdf" in value.lower():
                        return value
        return None

    def _extract_page_title(self, response_text: str, cms: dict[str, Any]) -> str:
        """Return a human-readable page title from the HTML or CMS context."""
        # Real PayPal pages store the title inside pageModel.metadata.page__title.
        page_model = cms.get("pageModel")
        if isinstance(page_model, dict):
            metadata = page_model.get("metadata") or {}
            if isinstance(metadata, dict):
                for key in ("page__title", "pageTitle", "title"):
                    value = metadata.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()
        cms_title = cms.get("pageTitle") or cms.get("pageName")
        if isinstance(cms_title, str) and cms_title.strip():
            return cms_title.strip()
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", response_text, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()
        return "PayPal Merchant and Seller Fees"

    def _extract_cms_updated_at(self, cms: dict[str, Any]) -> str | None:
        """Return the CMS content update timestamp from the page model."""
        page_model = cms.get("pageModel")
        if isinstance(page_model, dict):
            for key in ("update_time", "updatedAt", "lastModified"):
                value = page_model.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        return None

    def _extract_locale(self, cms: dict[str, Any]) -> str | None:
        """Return the page locale from the CMS context."""
        page_context = cms.get("pageContext")
        if isinstance(page_context, dict):
            cms_engine = page_context.get("cmsEngineContext")
            if isinstance(cms_engine, dict):
                requestor = cms_engine.get("requestor")
                if isinstance(requestor, dict):
                    locality = requestor.get("locality")
                    if isinstance(locality, dict):
                        locale = locality.get("locale")
                        if isinstance(locale, str) and locale.strip():
                            return locale.strip()
        page_model = cms.get("pageModel")
        if isinstance(page_model, dict):
            language = page_model.get("language")
            if isinstance(language, list) and language and isinstance(language[0], str) and language[0].strip():
                return language[0].strip()
        return None

    async def _crawl_country(self, market: Market) -> tuple[CountryOutput | None, UnsupportedCountry | None, bool]:
        """Crawl a single country and return its output, unsupported record, or transient flag.

        The third return value is True when a transient (network/parser) failure
        occurred and the caller should decide whether to reuse previous data.
        """
        code = market.paypal_market_code
        previous = self._load_previous_country_output(market)
        try:
            fee_url = await discover_fee_page(self.http_client, market, self.config)
        except UnsupportedCountryError as exc:
            logger.info("Country %s has no fee page: %s", code, exc)
            previous_unsupported = None
            if self._previous_state is not None:
                previous_unsupported = self._previous_state.unsupported_records.get(code)
            if previous_unsupported is not None:
                return None, previous_unsupported, False
            return (
                None,
                UnsupportedCountry(
                    paypal_market_code=code,
                    iso_country_code=market.iso_country_code,
                    country_name=market.country_name,
                    tested_urls=exc.tested_urls,
                    reason=str(exc),
                    first_confirmed_at=self._stable_timestamp(),
                    last_confirmed_at=self._stable_timestamp(),
                    temporary=False,
                ),
                False,
            )
        except FeePageError as exc:
            logger.warning("Fee page discovery failed for %s: %s", code, exc)
            return None, None, True

        # Try to use cached source if we have previous output.
        cached = None
        if previous is not None:
            cached = CachedSource(
                etag=previous.source.etag,
                last_modified=previous.source.last_modified,
                content_sha256=previous.source.content_sha256,
            )

        try:
            response = await self.http_client.get(fee_url, cached=cached)
        except NetworkError as exc:
            logger.warning("Network error for %s: %s", code, exc)
            return None, None, True

        if response.status_code == 304 and cached and cached.content_sha256 and previous is not None:
            # Reuse previous output unchanged.
            return previous, None, False

        try:
            cms = extract_cms_context(response.text)
        except ParserError as exc:
            logger.warning("CMS context missing for %s, falling back to HTML extraction: %s", code, exc)
            cms = None

        if cms is not None:
            extractor = ComponentsExtractor()
            sections, tables, warnings = extractor.extract(cms)
            page_id = get_canonical_page_id(cms) or "unknown"
            page_title = self._extract_page_title(response.text, cms)
            page_updated = self._extract_update_date(cms, sections)
            cms_updated = self._extract_cms_updated_at(cms)
            page_locale = self._extract_locale(cms)
            pdf_url = self._extract_pdf_url(cms)
        else:
            sections, tables, warnings = extract_html_tables(response.text, str(response.url))
            page_id = str(response.url).rstrip("/").split("/")[-1] or "unknown"
            page_title = self._extract_page_title(response.text, {})
            page_updated = self._extract_update_date({}, sections)
            cms_updated = None
            page_locale = extract_html_locale(response.text) or market.locale
            pdf_url = extract_html_pdf_url(response.text)

        self.warnings.extend(warnings)
        if self.config.classifier_mode == ClassifierMode.STRUCTURAL:
            derived = classify_structural(
                tables, market_code=market.paypal_market_code, locale=page_locale
            ).derived
        elif self.config.classifier_mode == ClassifierMode.SHADOW:
            legacy_run = classify_legacy(
                tables, market_code=market.paypal_market_code, locale=page_locale
            )
            structural_run = classify_structural(
                tables, market_code=market.paypal_market_code, locale=page_locale
            )
            if legacy_run.derived != structural_run.derived:
                self.warnings.append(
                    ParserWarning(
                        code="classifier_diff",
                        message="Structural classifier produced different derived fees than legacy",
                    )
                )
            derived = legacy_run.derived
        else:
            derived = classify_legacy(
                tables, market_code=market.paypal_market_code, locale=page_locale
            ).derived
        if page_locale and not market.locale:
            market = market.model_copy(update={"locale": page_locale})

        source = Source(
            requested_url=fee_url,
            canonical_url=str(response.url),
            page_id=str(page_id),
            page_title=page_title,
            page_updated_at=page_updated,
            cms_updated_at=cms_updated,
            pdf_url=pdf_url,
            etag=response.etag,
            last_modified=response.last_modified,
            content_sha256=response.content_sha256,
        )

        output = CountryOutput(
            schema_version=1,
            generated_at=self._stable_timestamp(),
            market=market,
            source=source,
            sections=sections,
            tables=tables,
            derived=derived,
            warnings=warnings,
        )
        return output, None, False

    async def crawl(self) -> CrawlReport:
        """Run the full crawl and publish output."""
        output_dir = Path(self.config.output_dir) if self.config.output_dir else None
        if not output_dir:
            raise CrawlerValidationError("No output directory configured")

        self._previous_state = PreviousState.load(output_dir)
        previous_state = self._previous_state

        markets = await self.discover()
        if self.config.countries:
            selected = {c.upper() for c in self.config.countries}
            markets = [m for m in markets if m.paypal_market_code in selected]

        if not markets:
            raise CountryDiscoveryError("No markets to crawl")

        # Limit concurrency.
        semaphore = asyncio.Semaphore(self.config.max_workers)
        outputs: dict[str, CountryOutput] = {}
        unsupported: list[UnsupportedCountry] = []
        failed: list[str] = []
        reused: list[str] = []

        async def _process(market: Market) -> None:
            async with semaphore:
                cc = market.paypal_market_code
                try:
                    output, unsup, transient = await self._crawl_country(market)
                    if output is not None:
                        outputs[cc] = output
                    elif unsup is not None:
                        unsupported.append(unsup)
                    elif transient:
                        if (
                            self.config.transient_policy == "reuse-previous"
                            and cc in previous_state.supported_countries
                        ):
                            previous_output = self._load_previous_country_output(market)
                            if previous_output is not None:
                                outputs[cc] = previous_output
                                reused.append(cc)
                                return
                        failed.append(cc)
                    else:
                        failed.append(cc)
                except Exception as exc:
                    logger.warning("Unexpected error for %s: %s", cc, exc)
                    failed.append(cc)

        await asyncio.gather(*[_process(m) for m in markets])

        # Validate each output.
        validation_errors: list[str] = []
        for cc, output in list(outputs.items()):
            if cc in reused:
                # Reused previous data is already validated from a previous run.
                continue
            errors = validate_country_output(output.model_dump(mode="json"))
            if errors:
                validation_errors.append(f"{cc}: " + "; ".join(errors))
                logger.error("Validation errors for %s: %s", cc, "; ".join(errors))
                failed.append(cc)
                del outputs[cc]

        if not outputs:
            raise CrawlerValidationError("No country output passed validation:\n" + "\n".join(validation_errors))

        # Default policy: any transient failure blocks publication to avoid mixed freshness.
        if failed and self.config.transient_policy == "fail":
            raise CrawlerValidationError(
                "Crawl failed on transient or validation errors (transient_policy=fail): " + ", ".join(sorted(failed))
            )

        # Regression checks against like-for-like market sets.
        previous = previous_state
        limits = RegressionLimits(
            max_table_count_delta_ratio=0.5,
            max_row_count_delta_ratio=0.5,
            max_country_count_delta_ratio=0.1,
            allow_country_drop=self.config.allow_country_drop,
        )
        current_discovered = {m.paypal_market_code for m in markets}
        current_supported = set(outputs.keys())
        current_unsupported = {u.paypal_market_code for u in unsupported}
        current_transient = set(failed)
        change_report = check_regression(
            previous,
            current_discovered,
            current_supported,
            current_unsupported,
            current_transient,
            outputs,
            limits,
        )
        try:
            enforce_regression(change_report, self.config.fail_on_regression)
        except RegressionError as exc:
            logger.error("Regression guard failed: %s", exc)
            raise

        # Publish atomically.
        publisher = OutputPublisher(
            output_dir=output_dir,
            staging_dir=self.config.staging_dir,
            timestamp=self._stable_timestamp(),
        )
        staging: Path | None = None
        try:
            _, staging = publisher.publish(outputs, markets, unsupported, change_report)
            changed, _changed_files = publisher.commit(staging)
        except Exception as exc:
            if staging is not None:
                publisher.rollback(staging)
            raise CrawlerValidationError(f"Failed to publish output: {exc}") from exc

        # Validate final output on disk.
        final_errors = validate_all_output(output_dir)
        if final_errors:
            raise CrawlerValidationError("Published output failed validation:\n" + "\n".join(final_errors))

        # Determine exit code.
        exit_code = ExitCode.SUCCESS_NO_CHANGE
        if self.config.fail_on_warning and self.warnings:
            exit_code = ExitCode.PARSER_FAILURE
        if failed:
            exit_code = ExitCode.PARSER_FAILURE

        return CrawlReport(
            exit_code=exit_code,
            changed=changed,
            countries_processed=len(outputs),
            countries_failed=failed,
            countries_unsupported=[u.paypal_market_code for u in unsupported],
            countries_reused=reused,
            warnings=self.warnings,
        )
