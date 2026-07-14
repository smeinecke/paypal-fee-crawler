"""Main crawl orchestration for the PayPal fee crawler."""

from __future__ import annotations

import asyncio
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
from .html_tables import extract_html_locale, extract_html_pdf_url, extract_html_tables
from .http import CachedSource, HttpClient, HttpResponse
from .models import (
    ChangeReport,
    ClassifierMetadata,
    CountryOutput,
    CrawlConfiguration,
    CrawlReport,
    CrawlState,
    DerivedFeeResult,
    Market,
    ParserWarning,
    PublicCountryOutput,
    Source,
    UnsupportedCountry,
)
from .output import OutputPublisher
from .regression import PreviousState, RegressionLimits, check_regression, enforce_regression
from .validation import validate_all_output, validate_country_output

logger = logging.getLogger(__name__)


def _placeholder_tables(table_count: int, row_count: int) -> list:
    """Return a list of tables with a matching total row count.

    The public v3 country files do not contain the full table structure, but the
    regression guard still needs accurate structural counts when a previous run
    is reused after a transient failure.
    """
    from .models import Row, Table

    if table_count <= 0 or row_count <= 0:
        return []
    base = row_count // table_count
    extra = row_count % table_count
    tables = []
    for i in range(table_count):
        rows = [Row() for _ in range(base + (1 if i < extra else 0))]
        tables.append(Table(rows=rows))
    return tables


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
        """Load the previous country output from disk, if available and valid.

        Public v3 country files no longer contain the full internal structure, so
        we reconstitute a minimal internal ``CountryOutput`` from the public file
        and the compact ``crawl-state.json`` sidecar.
        """
        if not self.config.output_dir:
            return None
        output_dir = Path(self.config.output_dir)
        prev_path = output_dir / "json" / f"{market.url_slug}.json"
        state_path = output_dir / "meta" / "crawl-state.json"
        if not prev_path.exists():
            return None
        try:
            public = PublicCountryOutput.model_validate_json(prev_path.read_text())
        except Exception as exc:
            logger.debug("Could not load previous public output for %s: %s", market.paypal_market_code, exc)
            return None

        state_entry = None
        if state_path.exists():
            try:
                state = CrawlState.model_validate_json(state_path.read_text())
                state_entry = state.markets.get(market.paypal_market_code)
            except Exception as exc:
                logger.debug("Could not load crawl state for %s: %s", market.paypal_market_code, exc)

        source_url = None
        source_updated_at = None
        raw_content_sha256 = None
        table_count = 0
        row_count = 0
        if state_entry is not None:
            source_url = state_entry.source_url
            source_updated_at = state_entry.source_updated_at
            raw_content_sha256 = state_entry.raw_content_sha256
            table_count = state_entry.table_count
            row_count = state_entry.row_count

        source = Source(
            requested_url=source_url or "",
            canonical_url=source_url,
            page_updated_at=source_updated_at,
            content_sha256=raw_content_sha256,
        )

        # Preserve previous structural counts for regression comparison even
        # though the public file does not contain the tables.
        tables = _placeholder_tables(table_count, row_count)

        return CountryOutput(
            schema_version=1,
            generated_at=public.generated_at,
            market=Market.model_validate(public.market.model_dump(mode="json")),
            source=source,
            tables=tables,
            derived=DerivedFeeResult.model_validate(public.derived.model_dump(mode="json")),
        )

    def _load_crawl_cache(self, market: Market) -> CachedSource | None:
        """Load cached HTTP headers for conditional requests from a sidecar file."""
        if not self.config.output_dir:
            return None
        cache_path = Path(self.config.output_dir) / "meta" / "crawl-cache.json"
        if not cache_path.exists():
            return None
        try:
            from .models import CrawlCache

            cache = CrawlCache.model_validate_json(cache_path.read_text())
            entry = cache.markets.get(market.paypal_market_code)
            if entry is None:
                return None
            return CachedSource(
                etag=entry.etag,
                last_modified=entry.last_modified,
                content_sha256=entry.content_sha256,
            )
        except Exception as exc:
            logger.debug("Could not load crawl cache for %s: %s", market.paypal_market_code, exc)
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
            r"Last\s+updated\s*:\s*([^\n<]+)",
            r"Updated\s*:\s*([^\n<]+)",
            r"Letzte\s+Aktualisierung\s*:\s*([^\n<]+)",
            r"(?:Aktualisiert|Stand)\s*:\s*([^\n<]+)",
            r"(?:Aktualisiert|Updated)\s+(?:am\s+)?([^\n<]+)",
            r"Dernière\s+mise\s+à\s+jour\s*:\s*([^\n<]+)",
            r"Mise\s+à\s+jour\s*:\s*([^\n<]+)",
            r"Última\s+actualización\s*:\s*([^\n<]+)",
            r"Actualizado\s+(?:el\s+)?([^\n<]+)",
            r"Ultimo\s+aggiornamento\s*:\s*([^\n<]+)",
            r"Ultima\s+attualizzazione\s*:\s*([^\n<]+)",
            r"Aggiornato\s+(?:il\s+)?([^\n<]+)",
            r"Última\s+atualização\s*:\s*([^\n<]+)",
            r"Atualizado\s+(?:em\s+)?([^\n<]+)",
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

    def _unsupported_country_result(
        self,
        market: Market,
        exc: UnsupportedCountryError,
    ) -> tuple[CountryOutput | None, UnsupportedCountry | None, bool]:
        code = market.paypal_market_code
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

    def _resolve_cached_source(
        self,
        market: Market,
        previous: CountryOutput | None,
    ) -> CachedSource | None:
        cached = self._load_crawl_cache(market)
        if cached is None and previous is not None:
            return CachedSource(
                etag=previous.source.etag,
                last_modified=previous.source.last_modified,
                content_sha256=previous.source.content_sha256,
            )
        return cached

    async def _fetch_fee_page(
        self,
        code: str,
        fee_url: str,
        cached: CachedSource | None,
    ) -> HttpResponse | None:
        try:
            return await self.http_client.get(fee_url, cached=cached)
        except NetworkError as exc:
            logger.warning("Network error for %s: %s", code, exc)
            return None

    def _extract_page_content(
        self,
        response: HttpResponse,
        market: Market,
    ) -> tuple[
        list[Any],
        list[Any],
        list[Any],
        str,
        str,
        str | None,
        str | None,
        str | None,
        str | None,
    ]:
        try:
            cms = extract_cms_context(response.text)
        except ParserError as exc:
            logger.warning(
                "CMS context missing for %s, falling back to HTML extraction: %s",
                market.paypal_market_code,
                exc,
            )
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

        return (
            sections,
            tables,
            warnings,
            page_id,
            page_title,
            page_updated,
            cms_updated,
            page_locale,
            pdf_url,
        )

    def _classify_output(
        self,
        tables: list[Any],
        source: Source,
    ) -> DerivedFeeResult:
        return classify_tables(tables, source=source)

    async def _crawl_country(self, market: Market) -> tuple[CountryOutput | None, UnsupportedCountry | None, bool]:
        """Crawl a single country and return its output, unsupported record, and transient flag."""
        code = market.paypal_market_code
        previous = self._load_previous_country_output(market)
        try:
            fee_url = await discover_fee_page(self.http_client, market, self.config)
        except UnsupportedCountryError as exc:
            logger.info("Country %s has no fee page: %s", code, exc)
            return self._unsupported_country_result(market, exc)
        except FeePageError as exc:
            logger.warning("Fee page discovery failed for %s: %s", code, exc)
            return None, None, True

        cached = self._resolve_cached_source(market, previous)
        response = await self._fetch_fee_page(code, fee_url, cached)
        if response is None:
            return None, None, True
        if response.status_code == 304 and cached and cached.content_sha256 and previous is not None:
            return previous, None, False

        (
            sections,
            tables,
            warnings,
            page_id,
            page_title,
            page_updated,
            cms_updated,
            page_locale,
            pdf_url,
        ) = self._extract_page_content(response, market)
        self.warnings.extend(warnings)
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
        derived = self._classify_output(tables, source)

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

    def _filter_countries(self, markets: list[Market]) -> list[Market]:
        if self.config.countries:
            selected = {c.upper() for c in self.config.countries}
            return [m for m in markets if m.paypal_market_code in selected]
        return markets

    async def _process_market(
        self,
        market: Market,
        semaphore: asyncio.Semaphore,
        previous_state: PreviousState,
    ) -> tuple[str, CountryOutput | None, UnsupportedCountry | None, bool, bool]:
        async with semaphore:
            cc = market.paypal_market_code
            try:
                output, unsup, transient = await self._crawl_country(market)
            except Exception as exc:
                logger.warning("Unexpected error for %s: %s", cc, exc)
                return cc, None, None, False, True
            if output is not None:
                return cc, output, None, False, False
            if unsup is not None:
                return cc, None, unsup, False, False
            if transient:
                if self.config.transient_policy == "reuse-previous" and cc in previous_state.supported_countries:
                    previous_output = self._load_previous_country_output(market)
                    if previous_output is not None:
                        return cc, previous_output, None, True, False
                return cc, None, None, False, True
            return cc, None, None, False, True

    async def _run_crawl(
        self,
        markets: list[Market],
        previous_state: PreviousState,
    ) -> tuple[dict[str, CountryOutput], list[UnsupportedCountry], list[str], list[str]]:
        semaphore = asyncio.Semaphore(self.config.max_workers)
        results = await asyncio.gather(*[self._process_market(m, semaphore, previous_state) for m in markets])
        outputs: dict[str, CountryOutput] = {}
        unsupported: list[UnsupportedCountry] = []
        failed: list[str] = []
        reused: list[str] = []
        for cc, output, unsup, is_reused, is_failed in results:
            if output is not None:
                outputs[cc] = output
                if is_reused:
                    reused.append(cc)
            elif unsup is not None:
                unsupported.append(unsup)
            elif is_failed:
                failed.append(cc)
        return outputs, unsupported, failed, reused

    def _validate_outputs(
        self,
        outputs: dict[str, CountryOutput],
        reused: list[str],
        failed: list[str],
    ) -> list[str]:
        validation_errors: list[str] = []
        for cc, output in list(outputs.items()):
            if cc in reused:
                continue
            errors = validate_country_output(output.model_dump(mode="json"))
            if errors:
                validation_errors.append(f"{cc}: " + "; ".join(errors))
                logger.error("Validation errors for %s: %s", cc, "; ".join(errors))
                failed.append(cc)
                del outputs[cc]
        return validation_errors

    def _build_change_report(
        self,
        previous_state: PreviousState,
        markets: list[Market],
        outputs: dict[str, CountryOutput],
        unsupported: list[UnsupportedCountry],
        failed: list[str],
    ) -> ChangeReport:
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
        classifier_metadata = ClassifierMetadata(
            classifier_mode="rules",
            classifier_version="rules-v1",
        )
        return check_regression(
            previous_state,
            current_discovered,
            current_supported,
            current_unsupported,
            current_transient,
            outputs,
            limits,
            current_classifier_metadata=classifier_metadata,
        )

    async def _publish_outputs(
        self,
        output_dir: Path,
        outputs: dict[str, CountryOutput],
        markets: list[Market],
        unsupported: list[UnsupportedCountry],
        change_report: ChangeReport,
    ) -> bool:
        publisher = OutputPublisher(
            output_dir=output_dir,
            staging_dir=self.config.staging_dir,
            timestamp=self._stable_timestamp(),
            keep_diagnostics=self.config.keep_diagnostics,
        )
        staging: Path | None = None
        try:
            _, staging = publisher.publish(
                outputs,
                markets,
                unsupported,
                change_report,
                classifier_metadata=ClassifierMetadata(
                    classifier_mode="rules",
                    classifier_version="rules-v1",
                ),
            )
            changed, _ = publisher.commit(staging)
        except Exception as exc:
            if staging is not None:
                publisher.rollback(staging)
            raise CrawlerValidationError(f"Failed to publish output: {exc}") from exc
        return changed

    def _determine_exit_code(self, failed: list[str]) -> ExitCode:
        if self.config.fail_on_warning and self.warnings:
            return ExitCode.PARSER_FAILURE
        if failed:
            return ExitCode.PARSER_FAILURE
        return ExitCode.SUCCESS_NO_CHANGE

    def _determine_diagnostics_path(self, output_dir: Path) -> str | None:
        if self.config.keep_diagnostics:
            return str(output_dir / "meta" / "diagnostics")
        return None

    async def crawl(self) -> CrawlReport:
        """Run the full crawl and publish output."""
        output_dir = Path(self.config.output_dir) if self.config.output_dir else None
        if not output_dir:
            raise CrawlerValidationError("No output directory configured")

        self._previous_state = PreviousState.load(output_dir)
        previous_state = self._previous_state

        markets = await self.discover()
        markets = self._filter_countries(markets)
        if not markets:
            raise CountryDiscoveryError("No markets to crawl")

        outputs, unsupported, failed, reused = await self._run_crawl(markets, previous_state)

        validation_errors = self._validate_outputs(outputs, reused, failed)
        if not outputs:
            raise CrawlerValidationError("No country output passed validation:\n" + "\n".join(validation_errors))

        if failed and self.config.transient_policy == "fail":
            raise CrawlerValidationError(
                "Crawl failed on transient or validation errors (transient_policy=fail): " + ", ".join(sorted(failed))
            )

        change_report = self._build_change_report(previous_state, markets, outputs, unsupported, failed)
        try:
            enforce_regression(change_report, self.config.fail_on_regression)
        except RegressionError as exc:
            logger.error("Regression guard failed: %s", exc)
            raise

        changed = await self._publish_outputs(
            output_dir,
            outputs,
            markets,
            unsupported,
            change_report,
        )

        final_errors = validate_all_output(output_dir)
        if final_errors:
            raise CrawlerValidationError("Published output failed validation:\n" + "\n".join(final_errors))

        exit_code = self._determine_exit_code(failed)
        diagnostics_path = self._determine_diagnostics_path(output_dir)
        return CrawlReport(
            exit_code=exit_code,
            changed=changed,
            countries_processed=len(outputs),
            countries_failed=failed,
            countries_unsupported=[u.paypal_market_code for u in unsupported],
            countries_reused=reused,
            warnings=self.warnings,
            change_report_path=str(output_dir / "change-report.json"),
            diagnostics_path=diagnostics_path,
        )
