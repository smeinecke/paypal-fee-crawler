"""Tests for schema-driven extraction helpers."""

from __future__ import annotations

from paypal_fee_crawler.extraction import (
    ExtractionDecision,
    ObservationKind,
    _column_roles,
    extract_conversion_spread,
    extract_fixed_fees,
    extract_international_surcharges,
    extract_standard_percentage,
)
from paypal_fee_crawler.models import FixedFees, Row, Table
from paypal_fee_crawler.pricing_tokens import render_rich_text_node
from paypal_fee_crawler.profiles import build_table_profile


def _table(rows: list[list[str]]) -> Table:
    return Table(
        component_id="c-1",
        document_id="FEETB001",
        source_order=1,
        rows=[Row(cells=[render_rich_text_node(cell) for cell in row]) for row in rows],
    )


def test_standard_percentage_from_mixed_row() -> None:
    table = _table([["Commercial", "2.99% + 0.39 EUR"]])
    profile = build_table_profile(table)
    decision = extract_standard_percentage(table, profile)
    assert decision.value == "2.99"
    assert decision.selected_rows == (0,)


def test_standard_percentage_ignores_excluded_labels() -> None:
    table = _table([["Charity donation", "2.9%"], ["Commercial", "2.99%"]])
    profile = build_table_profile(table)
    decision = extract_standard_percentage(table, profile)
    assert decision.value == "2.99"


def test_standard_percentage_from_structural_row() -> None:
    table = _table([["Some row", "2.99%"]])
    profile = build_table_profile(table)
    decision = extract_standard_percentage(table, profile)
    assert decision.value == "2.99"


def test_standard_percentage_ambiguous() -> None:
    table = _table([["Some row", "no percentage"]])
    profile = build_table_profile(table)
    decision = extract_standard_percentage(table, profile)
    assert decision.value is None
    assert all(o.kind == ObservationKind.EXTRACTION_CONFLICT for o in decision.observations)


def test_fixed_fee_extraction() -> None:
    table = _table([["EUR", "0.39 EUR"], ["USD", "0.49 USD"]])
    profile = build_table_profile(table)
    decision = extract_fixed_fees(table, profile)
    assert decision.value is not None
    assert set(decision.value) == {FixedFees(currency="EUR", amount="0.39"), FixedFees(currency="USD", amount="0.49")}


def test_fixed_fee_skips_notes() -> None:
    table = _table([["Currency", "Amount"], ["EUR", "0.39 EUR"], ["* footnote"]])
    profile = build_table_profile(table)
    decision = extract_fixed_fees(table, profile)
    assert decision.value is not None
    assert len(decision.value) == 1
    assert decision.value[0].currency == "EUR"


def test_fixed_fee_detects_conflict() -> None:
    table = _table([["EUR", "0.39 EUR"], ["EUR", "0.49 EUR"]])
    profile = build_table_profile(table)
    decision = extract_fixed_fees(table, profile)
    assert decision.value is not None
    assert len(decision.value) == 1
    assert any(o.kind == ObservationKind.EXTRACTION_CONFLICT for o in decision.observations)


def test_column_roles_label_and_value() -> None:
    table = _table([["Region", "Surcharge"], ["EEA", "+0.49%"]])
    profile = build_table_profile(table)
    roles = _column_roles(profile, table)
    assert roles.label_column == 0
    assert roles.percentage_columns == (1,)
    assert roles.money_columns == ()
    assert roles.confidence == 100


def test_international_surcharge_extraction() -> None:
    table = _table([["Region", "Surcharge"], ["EEA", "+0.49%"], ["GB", "+1.29%"]])
    profile = build_table_profile(table)
    decision = extract_international_surcharges(table, profile, market_code="US")
    assert decision.value is not None
    assert len(decision.value) == 2
    regions = {s.region for s in decision.value}
    assert regions == {"EEA", "GB"}


def test_conversion_spread_rejects_unapproved() -> None:
    table = _table([["2.99%"]])
    profile = build_table_profile(table)
    decision = extract_conversion_spread(table, profile, has_approved_evidence=False)
    assert decision.value is None
    assert any(o.kind == ObservationKind.UNKNOWN_FINGERPRINT for o in decision.observations)


def test_conversion_spread_with_approved_evidence() -> None:
    table = _table([["Currency conversion spread", "2.99%"]])
    profile = build_table_profile(table)
    decision = extract_conversion_spread(table, profile, has_approved_evidence=True)
    assert decision.value == "2.99"


def test_extraction_decision_typing() -> None:
    decision = ExtractionDecision(value="1.99", selected_rows=(0,), evidence=(), observations=())
    assert decision.value == "1.99"
