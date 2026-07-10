"""Main crawl orchestration for the PayPal fee crawler."""

from __future__ import annotations

import asyncio
import datetime
import logging
from pathlib import Path
from typing import Any

from .classify import classify_tables
from .cms_context import extract_cms_context
from .components import ComponentsExtractor
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

    async def _crawl_country(self, market: Market) -> tuple[CountryOutput | None, UnsupportedCountry | None]:
        """Crawl a single country and return its output or unsupported record."""
        cc = market.country_code
        try:
            fee_url = await discover_fee_page(self.http_client, market, self.config)
        except UnsupportedCountryError as exc:
            logger.info("Country %s has no fee page: %s", cc, exc)
            return None, UnsupportedCountry(
                country_code=cc,
                country_name=market.country_name,
                tested_urls=[],
                reason=str(exc),
                first_confirmed_at=datetime.datetime.now(datetime.UTC).isoformat(),
                last_confirmed_at=datetime.datetime.now(datetime.UTC).isoformat(),
                temporary=False,
            )
        except FeePageError as exc:
            logger.warning("Fee page discovery failed for %s: %s", cc, exc)
            return None, None

        # Try to use cached source if we have previous output.
        cached = None
        if self.config.output_dir:
            prev_path = Path(self.config.output_dir) / "json" / f"{cc.lower()}.json"
            if prev_path.exists():
                try:
                    prev = CountryOutput.model_validate_json(prev_path.read_text())
                    cached = CachedSource(
                        etag=prev.source.etag,
                        last_modified=prev.source.last_modified,
                        content_sha256=prev.source.content_sha256,
                    )
                except Exception as exc:
                    logger.debug("Could not read cached source for %s: %s", cc, exc)

        try:
            response = await self.http_client.get(fee_url, cached=cached)
        except NetworkError as exc:
            logger.warning("Network error for %s: %s", cc, exc)
            return None, None

        if response.status_code == 304 and cached and cached.content_sha256 and self.config.output_dir:
            # Reuse previous output.
            prev_path = Path(self.config.output_dir) / "json" / f"{cc.lower()}.json"
            if prev_path.exists():
                return CountryOutput.model_validate_json(prev_path.read_text()), None

        try:
            cms = extract_cms_context(response.text)
        except ParserError as exc:
            logger.warning("Parser error for %s: %s", cc, exc)
            return None, None

        extractor = ComponentsExtractor()
        sections, tables, warnings = extractor.extract(cms)
        self.warnings.extend(warnings)

        derived = classify_tables(tables)
        page_id = get_canonical_page_id(cms) or "unknown"
        page_title = cms.get("pageTitle") or cms.get("pageName") or "PayPal Merchant and Seller Fees"
        page_updated = cms.get("pageUpdatedAt") or cms.get("lastModified")

        source = Source(
            requested_url=fee_url,
            canonical_url=str(response.url),
            page_id=str(page_id),
            page_title=str(page_title) if page_title else None,
            page_updated_at=str(page_updated) if page_updated else None,
            cms_updated_at=None,
            pdf_url=None,
            etag=response.etag,
            last_modified=response.last_modified,
            content_sha256=response.content_sha256,
        )

        output = CountryOutput(
            schema_version=1,
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
            markets = [m for m in markets if m.country_code in selected]

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
                        outputs[market.country_code] = output
                    elif unsup:
                        unsupported.append(unsup)
                    else:
                        failed.append(market.country_code)
                except Exception as exc:
                    logger.warning("Unexpected error for %s: %s", market.country_code, exc)
                    failed.append(market.country_code)

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

        # Regression checks.
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
        )
        staging: Path | None = None
        try:
            _, staging = publisher.publish(outputs, markets, unsupported, change_report)
            changed, changed_files = publisher.commit(staging)
            publisher.rollback(staging)
        except Exception as exc:
            if staging is not None:
                publisher.rollback(staging)
            raise CrawlerValidationError(f"Failed to publish output: {exc}") from exc

        # Validate final output on disk.
        final_errors = validate_all_output(output_dir)
        if final_errors:
            raise CrawlerValidationError("Published output failed validation:\n" + "\n".join(final_errors))

        exit_code = ExitCode.SUCCESS_WITH_CHANGES if changed else ExitCode.SUCCESS_NO_CHANGE
        if self.config.fail_on_warning and self.warnings:
            exit_code = ExitCode.PARSER_FAILURE

        return CrawlReport(
            exit_code=exit_code,
            changed=changed,
            countries_processed=len(outputs),
            countries_failed=failed,
            countries_unsupported=[u.country_code for u in unsupported],
            warnings=self.warnings,
        )
