"""Main crawl orchestration for the PayPal fee crawler."""

from __future__ import annotations

import asyncio
import datetime
import logging
import re
from pathlib import Path
from typing import Any

from .classify import classify_tables
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
from .http import CachedSource, HttpClient
from .models import (
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
                homepage_url="https://www.paypal.com/de",
            )
            if not markets:
                raise CountryDiscoveryError("No markets discovered")
            return markets
        except CountryDiscoveryError as exc:
            if self.config.country_manifest_path:
                # Try to load previous manifest.
                previous = self._load_previous_manifest()
                if previous:
                    logger.warning("Discovery failed; using previous manifest: %s", exc)
                    return previous.markets
            logger.warning("Discovery failed; falling back to bootstrap list: %s", exc)
            return get_bootstrap_markets()

    def _load_previous_manifest(self) -> Any | None:
        path = Path(self.config.country_manifest_path) if self.config.country_manifest_path else None
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

    def _stable_timestamp(self) -> str:
        """Return a deterministic timestamp for this run.

        The output timestamp is taken from the configuration when provided; otherwise
        the current time is used. This keeps the output stable across transient failures.
        """
        if self.config.timestamp:
            return self.config.timestamp
        return datetime.datetime.now(datetime.UTC).isoformat()

    def _extract_update_date(self, cms: dict[str, Any], sections: list[Any]) -> str | None:
        """Find an explicit update date in the CMS sections or metadata."""
        # Try explicit metadata at the top level first.
        for key in ("pageUpdatedAt", "lastModified", "updatedAt", "publishedAt"):
            value = cms.get(key)
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
        # The real CMS context does not always contain a pageTitle; fall back to HTML.
        cms_title = cms.get("pageTitle") or cms.get("pageName")
        if isinstance(cms_title, str) and cms_title.strip():
            return cms_title.strip()
        title_match = re.search(r"<title[^>]*>([^<]+)</title>", response_text, re.IGNORECASE)
        if title_match:
            return title_match.group(1).strip()
        return "PayPal Merchant and Seller Fees"

    async def _crawl_country(self, market: Market) -> tuple[CountryOutput | None, UnsupportedCountry | None]:
        """Crawl a single country and return its output or unsupported record."""
        code = market.paypal_market_code
        previous = self._load_previous_country_output(market)
        try:
            fee_url = await discover_fee_page(self.http_client, market, self.config)
        except UnsupportedCountryError as exc:
            logger.info("Country %s has no fee page: %s", code, exc)
            return None, UnsupportedCountry(
                paypal_market_code=code,
                iso_country_code=market.iso_country_code,
                country_name=market.country_name,
                tested_urls=[],
                reason=str(exc),
                first_confirmed_at=self._stable_timestamp(),
                last_confirmed_at=self._stable_timestamp(),
                temporary=False,
            )
        except FeePageError as exc:
            logger.warning("Fee page discovery failed for %s: %s", code, exc)
            # Preserve previous output if available to avoid data loss on transient failures.
            if previous is not None:
                return previous, None
            return None, None

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
            if previous is not None:
                return previous, None
            return None, None

        if response.status_code == 304 and cached and cached.content_sha256 and previous is not None:
            # Reuse previous output unchanged.
            return previous, None

        try:
            cms = extract_cms_context(response.text)
        except ParserError as exc:
            logger.warning("Parser error for %s: %s", code, exc)
            if previous is not None:
                return previous, None
            return None, None

        extractor = ComponentsExtractor()
        sections, tables, warnings = extractor.extract(cms)
        self.warnings.extend(warnings)

        derived = classify_tables(tables)
        page_id = get_canonical_page_id(cms) or "unknown"
        page_title = self._extract_page_title(response.text, cms)
        page_updated = self._extract_update_date(cms, sections)
        pdf_url = self._extract_pdf_url(cms)

        source = Source(
            requested_url=fee_url,
            canonical_url=str(response.url),
            page_id=str(page_id),
            page_title=page_title,
            page_updated_at=page_updated,
            cms_updated_at=None,
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
        return output, None

    async def crawl(self) -> CrawlReport:
        """Run the full crawl and publish output."""
        output_dir = Path(self.config.output_dir) if self.config.output_dir else None
        if not output_dir:
            raise CrawlerValidationError("No output directory configured")

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

        async def _process(market: Market) -> None:
            async with semaphore:
                try:
                    output, unsup = await self._crawl_country(market)
                    if output:
                        outputs[market.paypal_market_code] = output
                    elif unsup:
                        unsupported.append(unsup)
                    else:
                        failed.append(market.paypal_market_code)
                except Exception as exc:
                    logger.warning("Unexpected error for %s: %s", market.paypal_market_code, exc)
                    failed.append(market.paypal_market_code)

        await asyncio.gather(*[_process(m) for m in markets])

        # Validate each output.
        validation_errors: list[str] = []
        for cc, output in list(outputs.items()):
            errors = validate_country_output(output.model_dump(mode="json"))
            if errors:
                validation_errors.append(f"{cc}: " + "; ".join(errors))
                failed.append(cc)
                del outputs[cc]

        if not outputs:
            raise CrawlerValidationError("No country output passed validation:\n" + "\n".join(validation_errors))

        # Regression checks against separate market sets.
        previous = PreviousState.load(output_dir)
        limits = RegressionLimits(
            max_table_count_delta_ratio=0.5,
            max_row_count_delta_ratio=0.5,
            max_country_count_delta_ratio=0.1,
            allow_country_drop=self.config.allow_country_drop,
        )
        change_report = check_regression(previous, outputs, limits)
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
            publisher.rollback(staging)
        except Exception as exc:
            if staging is not None:
                publisher.rollback(staging)
            raise CrawlerValidationError(f"Failed to publish output: {exc}") from exc

        # Validate final output on disk.
        final_errors = validate_all_output(output_dir)
        if final_errors:
            raise CrawlerValidationError("Published output failed validation:\n" + "\n".join(final_errors))

        # Success always returns 0; warnings are surfaced in the report and CLI
        # decides whether to promote them to a non-zero exit code.
        exit_code = ExitCode.SUCCESS_NO_CHANGE
        if self.config.fail_on_warning and self.warnings:
            exit_code = ExitCode.PARSER_FAILURE
        # A failure to fully process any requested country is a non-zero failure
        # unless the previous data was preserved (in which case it is still a warning).
        if failed:
            exit_code = ExitCode.PARSER_FAILURE

        return CrawlReport(
            exit_code=exit_code,
            changed=changed,
            countries_processed=len(outputs),
            countries_failed=failed,
            countries_unsupported=[u.paypal_market_code for u in unsupported],
            warnings=self.warnings,
        )
