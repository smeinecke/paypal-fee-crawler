"""Tests for core-fee classification."""

from __future__ import annotations

from paypal_fee_crawler.classify import classify_tables
from paypal_fee_crawler.models import Row, Table
from paypal_fee_crawler.pricing_tokens import render_rich_text_node


def _table(caption: str, rows: list[list[str]], section_path: list[str] | None = None) -> Table:
    return Table(
        caption=caption,
        section_path=section_path or [caption],
        rows=[Row(cells=[render_rich_text_node(cell) for cell in row]) for row in rows],
    )


def test_classify_standard_commercial() -> None:
    tables = [
        _table("Commercial transaction fees", [["Commercial transactions", "2.99% + 0.39 EUR"]]),
        _table("Fixed fee by received currency", [["EUR", "0.39 EUR"], ["USD", "0.49 USD"]]),
    ]
    derived = classify_tables(tables)
    assert derived.status == "complete"
    assert derived.standard_commercial.percentage == "2.99"
    assert any(fee.currency == "EUR" for fee in derived.commercial_fixed_fees)


def test_classify_international_surcharge() -> None:
    tables = [
        _table("International surcharge", [["EEA", "0%"], ["GB", "+1.29%"], ["Other", "+1.99%"]]),
    ]
    derived = classify_tables(tables)
    assert derived.status == "partial"
    regions = {s.region: s.percentage_points for s in derived.international_surcharges}
    assert regions.get("GB") == "1.29"
    assert regions.get("OTHER") == "1.99"


def test_classify_unclassified_when_uncertain() -> None:
    tables = [
        _table("Random table", [["A", "B"], ["C", "D"]]),
    ]
    derived = classify_tables(tables)
    assert derived.status == "unclassified"
