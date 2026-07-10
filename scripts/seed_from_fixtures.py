#!/usr/bin/env python3
"""Seed the paypal-fee-data repository from synthetic fixtures.

This is a one-off helper used to generate the initial data set while live
PayPal pages block unauthenticated automated requests. It is not part of the
regular crawler pipeline, which remains fully live-fetch capable.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from paypal_fee_crawler.classify import classify_tables
from paypal_fee_crawler.cms_context import extract_cms_context
from paypal_fee_crawler.components import ComponentsExtractor
from paypal_fee_crawler.discovery import get_bootstrap_markets
from paypal_fee_crawler.models import CountryOutput, Market, Source
from paypal_fee_crawler.output import OutputPublisher
from paypal_fee_crawler.validation import validate_all_output

FIXTURES = Path(__file__).parent.parent / "tests" / "fixtures"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "paypal-fee-data"

BOOTSTRAP = {m.country_code: m for m in get_bootstrap_markets()}


def _build_output(code: str, html: str) -> CountryOutput:
    cms = extract_cms_context(html)
    extractor = ComponentsExtractor()
    sections, tables, warnings = extractor.extract(cms)
    derived = classify_tables(tables)

    market = BOOTSTRAP.get(code.upper())
    if market is None:
        market = Market(country_code=code.upper(), country_name=code.upper())

    source = Source(
        requested_url=f"https://www.paypal.com/{code.lower()}/business/paypal-business-fees",
        canonical_url=f"https://www.paypal.com/{code.lower()}/business/paypal-business-fees",
        page_id=cms.get("pageId"),
        page_title=cms.get("pageTitle"),
        page_updated_at=cms.get("pageUpdatedAt"),
        cms_updated_at=None,
    )

    return CountryOutput(
        schema_version=1,
        market=market,
        source=source,
        sections=sections,
        tables=tables,
        derived=derived,
        warnings=warnings,
    )


def main() -> int:
    outputs: dict[str, CountryOutput] = {}
    for code in ["de", "us", "gb"]:
        html = (FIXTURES / f"{code}.html").read_text(encoding="utf-8")
        outputs[code.upper()] = _build_output(code, html)

    publisher = OutputPublisher(OUTPUT_DIR)
    _, staging = publisher.publish(
        outputs,
        markets=[outputs[cc].market for cc in outputs],
        unsupported=[],
    )
    changed, changed_files = publisher.commit(staging)
    publisher.rollback(staging)

    errors = validate_all_output(OUTPUT_DIR)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"Seeded {len(outputs)} countries. Changed files: {changed_files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
