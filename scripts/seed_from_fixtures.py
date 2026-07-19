#!/usr/bin/env python3
"""Seed the paypal-fee-data repository from fixtures.

This helper regenerates a subset of countries from local HTML fixtures while
preserving existing data for all other markets. It is used when live PayPal
pages block unauthenticated automated requests, but it can also be run
incrementally to refresh fixture-backed markets without deleting the rest of
the dataset.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from paypal_fee_crawler.classify import classify_tables
from paypal_fee_crawler.cms_context import extract_cms_context
from paypal_fee_crawler.components import ComponentsExtractor
from paypal_fee_crawler.constants import FEE_PAGE_PATH_TEMPLATE, PAYPAL_BASE_URL
from paypal_fee_crawler.crawler import Crawler
from paypal_fee_crawler.discovery import get_bootstrap_markets, get_canonical_page_id
from paypal_fee_crawler.models import (
    CountryManifest,
    CountryOutput,
    CrawlCache,
    CrawlConfiguration,
    CrawlReport,
    Market,
    Source,
    UnsupportedCountry,
)
from paypal_fee_crawler.regression import PreviousState
from paypal_fee_crawler.validation import validate_all_output

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent

SEED_MARKETS = ["de", "us", "gb"]


def _fixture_path(code: str) -> Path:
    """Return the real-capture fixture path for a country."""
    return FIXTURES / f"paypal-{code}-real.html"


def _compute_content_sha256(html: str) -> str:
    return hashlib.sha256(html.encode("utf-8")).hexdigest()


def _load_existing_outputs(
    output_dir: Path, crawler: Crawler
) -> tuple[dict[str, CountryOutput], list[Market], list[UnsupportedCountry], list[UnsupportedCountry]]:
    """Load all existing public country outputs and rebuild minimal internal objects.

    Existing metadata such as HTTP cache headers, table counts and source URLs are
    preserved from ``crawl-state.json`` and ``crawl-cache.json`` so the published
    state remains consistent. Also returns the original manifest markets and the
    previously known unsupported/transient-failure entries so a seed run does not
    discard discovery state.
    """
    manifest_path = output_dir / "meta" / "countries.json"
    cache_path = output_dir / "meta" / "crawl-cache.json"

    manifest = None
    if manifest_path.exists():
        try:
            manifest = CountryManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Warning: could not load manifest: {exc}", file=sys.stderr)

    cache = CrawlCache()
    if cache_path.exists():
        try:
            cache = CrawlCache.model_validate_json(cache_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Warning: could not load crawl cache: {exc}", file=sys.stderr)

    outputs: dict[str, CountryOutput] = {}
    if manifest is None:
        return outputs

    for market in manifest.markets:
        cc = market.paypal_market_code
        output = crawler.load_previous_country_output(market)
        if output is None:
            continue

        cache_entry = cache.markets.get(cc)
        if cache_entry is not None:
            output = output.model_copy(
                update={
                    "source": output.source.model_copy(
                        update={
                            "etag": cache_entry.etag,
                            "last_modified": cache_entry.last_modified,
                            "content_sha256": cache_entry.content_sha256,
                        }
                    )
                }
            )

        outputs[cc] = output

    markets = manifest.markets if manifest is not None else []
    unsupported = manifest.unsupported if manifest is not None else []
    transient_failures = manifest.transient_failures if manifest is not None else []
    return outputs, markets, unsupported, transient_failures


def _build_output(code: str, html: str, existing: CountryOutput | None, crawler: Crawler) -> CountryOutput:
    """Classify a fixture and produce a CountryOutput, preserving existing metadata."""
    cms = extract_cms_context(html)
    extractor = ComponentsExtractor()
    sections, tables, warnings = extractor.extract(cms)

    bootstrap = {m.paypal_market_code: m for m in get_bootstrap_markets()}
    market = bootstrap.get(code.upper())
    if market is None:
        market = Market(paypal_market_code=code.upper(), iso_country_code=code.upper(), country_name=code.upper())

    page_id = get_canonical_page_id(cms) or "unknown"
    page_title = crawler.extract_page_title(html, cms)
    page_updated_at = crawler.extract_update_date(cms, sections)
    cms_updated_at = crawler.extract_cms_updated_at(cms)
    content_sha256 = _compute_content_sha256(html)
    requested_url = FEE_PAGE_PATH_TEMPLATE.format(base=PAYPAL_BASE_URL, market=code.lower())
    canonical_url = requested_url

    if existing is not None:
        existing_source = existing.source
        requested_url = existing_source.requested_url or requested_url
        canonical_url = existing_source.canonical_url or requested_url
        if not page_updated_at:
            page_updated_at = existing_source.page_updated_at
        if not cms_updated_at:
            cms_updated_at = existing_source.cms_updated_at
        if page_id == "unknown":
            page_id = existing_source.page_id or page_id
        if page_title == "PayPal Merchant and Seller Fees":
            page_title = existing_source.page_title or page_title

    source = Source(
        requested_url=requested_url,
        canonical_url=canonical_url,
        page_id=page_id,
        page_title=page_title,
        page_updated_at=page_updated_at,
        cms_updated_at=cms_updated_at,
        content_sha256=content_sha256,
    )

    derived = classify_tables(tables, source=source)

    generated_at = existing.generated_at if existing is not None else None

    return CountryOutput(
        schema_version=1,
        generated_at=generated_at,
        market=market,
        source=source,
        sections=sections,
        tables=tables,
        derived=derived,
        warnings=warnings,
    )


def main() -> int:
    crawler = Crawler(CrawlConfiguration(output_dir=str(OUTPUT_DIR)))
    outputs, markets, unsupported, transient_failures = _load_existing_outputs(OUTPUT_DIR, crawler)

    for code in SEED_MARKETS:
        fixture = _fixture_path(code)
        if not fixture.exists():
            print(f"Fixture not found: {fixture}", file=sys.stderr)
            return 1

        html = fixture.read_text(encoding="utf-8")
        existing = outputs.get(code.upper())
        outputs[code.upper()] = _build_output(code, html, existing, crawler)

    previous = PreviousState.load(OUTPUT_DIR)
    change_report = crawler.build_change_report(
        previous,
        markets,
        outputs,
        unsupported,
        [u.paypal_market_code for u in transient_failures],
    )

    existing_report: CrawlReport | None = None
    report_path = OUTPUT_DIR / "meta" / "crawl-report.json"
    if report_path.exists():
        try:
            existing_report = CrawlReport.model_validate_json(report_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Warning: could not load crawl report: {exc}", file=sys.stderr)

    changed, changed_files, publisher = crawler.publish_outputs(
        OUTPUT_DIR,
        outputs,
        markets,
        unsupported,
        change_report,
        [u.paypal_market_code for u in transient_failures],
    )

    report_kwargs = {
        "exit_code": 0,
        "changed": changed,
        "countries_processed": len(outputs),
    }
    if existing_report is not None:
        report = existing_report.model_copy(update=report_kwargs)
    else:
        report = CrawlReport(
            **report_kwargs,
            change_report_path="change-report.json",
        )
    publisher.write_crawl_report(OUTPUT_DIR, report)

    errors = validate_all_output(OUTPUT_DIR)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"Seeded {len(SEED_MARKETS)} countries. Changed files: {changed_files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
