#!/usr/bin/env python3
"""Generate minimal gold and synthetic corpus fixtures for PR 3 promotion gate.

Gold fixtures are created from simple synthetic CountryOutput objects where the
legacy and structural classifiers already agree.  Synthetic fixtures are
examples where they may differ (intentionally kept for future review).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from paypal_fee_crawler.classify import classify_legacy
from paypal_fee_crawler.models import CountryOutput, Market, Row, Source, Table
from paypal_fee_crawler.pricing_tokens import render_rich_text_node


def _make_table(caption: str, document_id: str, rows: list[list[str]]) -> Table:
    return Table(
        caption=caption,
        section_path=[caption],
        document_id=document_id,
        rows=[Row(cells=[render_rich_text_node(cell) for cell in row]) for row in rows],
    )


def _make_country(market_code: str, country_name: str, tables: list[Table]) -> CountryOutput:
    market = Market(
        paypal_market_code=market_code,
        country_name=country_name,
        country_code=market_code.lower(),
        locale="en",
    )
    derived = classify_legacy(tables, market_code=market_code).derived
    source = Source(
        requested_url="https://example.com",
        canonical_url="https://example.com",
        page_id="test",
        page_title="test",
    )
    return CountryOutput(
        schema_version=1,
        generated_at=datetime.now(UTC).isoformat(),
        market=market,
        source=source,
        sections=[],
        tables=tables,
        derived=derived,
        warnings=[],
    )


def _write_fixture(path: Path, country: CountryOutput) -> None:
    path.write_text(json.dumps(country.model_dump(mode="json"), indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    root = Path(__file__).parent
    gold_dir = root / "gold"
    synthetic_dir = root / "synthetic"
    gold_dir.mkdir(exist_ok=True)
    synthetic_dir.mkdir(exist_ok=True)

    fixed = _make_country(
        "DE",
        "Germany",
        [
            _make_table(
                "Fixed fee by currency",
                "FEETB18",
                [["EUR", "0.39 EUR"], ["USD", "0.49 USD"]],
            )
        ],
    )
    _write_fixture(gold_dir / "de-fixed.json", fixed)

    standard = _make_country(
        "GB",
        "United Kingdom",
        [
            _make_table(
                "Standard commercial transaction fees",
                "FEETB16",
                [["Commercial transactions", "2.99% + 0.39 EUR"]],
            )
        ],
    )
    _write_fixture(gold_dir / "gb-standard.json", standard)

    # A synthetic fixture where the legacy classifier is the source of truth but
    # the structural classifier may see an observation.
    ambiguous = _make_country(
        "US",
        "United States",
        [
            _make_table(
                "Commercial and fixed fee schedule",
                "FEETB999",
                [["Commercial", "2.99% + 0.39 USD"], ["USD", "0.39 USD"]],
            )
        ],
    )
    _write_fixture(synthetic_dir / "us-ambiguous.json", ambiguous)

    print(f"Wrote fixtures to {gold_dir} and {synthetic_dir}")


if __name__ == "__main__":
    main()
