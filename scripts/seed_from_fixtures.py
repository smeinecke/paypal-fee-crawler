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
from paypal_fee_crawler.crawler import Crawler
from paypal_fee_crawler.discovery import get_bootstrap_markets
from paypal_fee_crawler.models import (
    ChangeReport,
    ClassifierMetadata,
    CountryManifest,
    CountryOutput,
    CrawlCache,
    CrawlConfiguration,
    CrawlState,
    Market,
    Source,
)
from paypal_fee_crawler.output import OutputPublisher
from paypal_fee_crawler.regression import PreviousState, RegressionLimits, check_regression
from paypal_fee_crawler.validation import validate_all_output

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent

SEED_MARKETS = ["de", "us", "gb"]


def _fixture_path(code: str) -> Path:
    """Return the real-capture fixture path for a country."""
    return FIXTURES / f"paypal-{code}-real.html"


def _compute_content_sha256(html: str) -> str:
    return hashlib.sha256(html.encode("utf-8")).hexdigest()


def _load_existing_outputs(output_dir: Path) -> dict[str, CountryOutput]:
    """Load all existing public country outputs and rebuild minimal internal objects.

    Existing metadata such as HTTP cache headers, table counts and source URLs are
    preserved from ``crawl-state.json`` and ``crawl-cache.json`` so the published
    state remains consistent.
    """
    manifest_path = output_dir / "meta" / "countries.json"
    cache_path = output_dir / "meta" / "crawl-cache.json"
    state_path = output_dir / "meta" / "crawl-state.json"

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

    state = None
    if state_path.exists():
        try:
            state = CrawlState.model_validate_json(state_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"Warning: could not load crawl state: {exc}", file=sys.stderr)

    crawler = Crawler(CrawlConfiguration(output_dir=str(output_dir)))

    outputs: dict[str, CountryOutput] = {}
    if manifest is None:
        return outputs

    for market in manifest.markets:
        cc = market.paypal_market_code
        output = crawler._load_previous_country_output(market)
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

        state_entry = state.markets.get(cc) if state is not None else None
        if state_entry is not None and output.generated_at is None:
            output = output.model_copy(update={"generated_at": state_entry.source_updated_at})

        outputs[cc] = output

    return outputs


def _build_output(code: str, html: str, existing: CountryOutput | None) -> CountryOutput:
    """Classify a fixture and produce a CountryOutput, preserving existing metadata."""
    cms = extract_cms_context(html)
    extractor = ComponentsExtractor()
    sections, tables, warnings = extractor.extract(cms)
    derived = classify_tables(tables)

    bootstrap = {m.paypal_market_code: m for m in get_bootstrap_markets()}
    market = bootstrap.get(code.upper())
    if market is None:
        market = Market(paypal_market_code=code.upper(), iso_country_code=code.upper(), country_name=code.upper())

    page_id = cms.get("pageId")
    page_title = cms.get("pageTitle")
    cms_updated_at = cms.get("cmsUpdatedAt")
    page_updated_at = cms.get("pageUpdatedAt")
    content_sha256 = _compute_content_sha256(html)
    requested_url = f"https://www.paypal.com/{code.lower()}/business/paypal-business-fees"
    canonical_url = requested_url

    if existing is not None:
        existing_source = existing.source
        requested_url = existing_source.requested_url or requested_url
        canonical_url = existing_source.canonical_url or requested_url
        if not page_updated_at:
            page_updated_at = existing_source.page_updated_at
        if not cms_updated_at:
            cms_updated_at = existing_source.cms_updated_at
        if not page_id:
            page_id = existing_source.page_id
        if not page_title:
            page_title = existing_source.page_title

    source = Source(
        requested_url=requested_url,
        canonical_url=canonical_url,
        page_id=page_id,
        page_title=page_title,
        page_updated_at=page_updated_at,
        cms_updated_at=cms_updated_at,
        content_sha256=content_sha256,
    )

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


def _stable_timestamp(outputs: dict[str, CountryOutput]) -> str:
    """Return the latest generated_at timestamp so existing files remain stable."""
    timestamps = [o.generated_at for o in outputs.values() if o.generated_at]
    if not timestamps:
        return "2025-01-01T00:00:00+00:00"
    return max(timestamps)


def _build_change_report(
    outputs: dict[str, CountryOutput],
    output_dir: Path,
) -> ChangeReport:
    """Compute a change report against the previous committed state."""
    previous = PreviousState.load(output_dir)
    current_discovered = set(outputs.keys()) | previous.discovered_countries
    current_supported = set(outputs.keys())
    current_unsupported = previous.unsupported_countries
    current_transient: set[str] = set()

    return check_regression(
        previous,
        current_discovered,
        current_supported,
        current_unsupported,
        current_transient,
        outputs,
        RegressionLimits(),
        current_classifier_metadata=ClassifierMetadata(
            classifier_mode="rules",
            classifier_version="rules-v1",
        ),
    )


def main() -> int:
    outputs = _load_existing_outputs(OUTPUT_DIR)

    for code in SEED_MARKETS:
        fixture = _fixture_path(code)
        if not fixture.exists():
            print(f"Fixture not found: {fixture}", file=sys.stderr)
            return 1

        html = fixture.read_text(encoding="utf-8")
        existing = outputs.get(code.upper())
        outputs[code.upper()] = _build_output(code, html, existing)

    timestamp = _stable_timestamp(outputs)

    for cc in list(outputs.keys()):
        if outputs[cc].generated_at is None:
            outputs[cc] = outputs[cc].model_copy(update={"generated_at": timestamp})

    markets = [output.market for output in outputs.values()]
    unsupported: list = []

    change_report = _build_change_report(outputs, OUTPUT_DIR)

    publisher = OutputPublisher(OUTPUT_DIR, timestamp=timestamp)
    _, staging = publisher.publish(
        outputs,
        markets=markets,
        unsupported=unsupported,
        change_report=change_report,
        classifier_metadata=ClassifierMetadata(
            classifier_mode="rules",
            classifier_version="rules-v1",
        ),
    )
    changed, changed_files = publisher.commit(staging)
    publisher.rollback(staging)

    errors = validate_all_output(OUTPUT_DIR)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"Seeded {len(SEED_MARKETS)} countries. Changed files: {changed_files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
