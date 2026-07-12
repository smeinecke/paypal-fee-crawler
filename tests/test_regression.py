"""Regression tests for the four release-blocking defects."""

from __future__ import annotations

from paypal_fee_crawler import scoring
from paypal_fee_crawler.classify import classify_structural
from paypal_fee_crawler.extraction import (
    ObservationKind,
    extract_conversion_spread,
    extract_fixed_fees,
    extract_standard_percentage,
)
from paypal_fee_crawler.models import FixedFees, Row, Table
from paypal_fee_crawler.pricing_tokens import render_rich_text_node
from paypal_fee_crawler.profiles import build_table_profile
from paypal_fee_crawler.scoring import BlockerCode, EvidenceSignal, ScoreResult


def _table(caption: str, rows: list[list[str]], document_id: str = "FEETB001") -> Table:
    return Table(
        caption=caption,
        section_path=[caption],
        document_id=document_id,
        source_order=1,
        rows=[Row(cells=[render_rich_text_node(cell) for cell in row]) for row in rows],
    )


def _cell(text: str) -> object:
    """Return a Cell with pricing tokens tokenized from *text*."""
    return render_rich_text_node(text)


def test_select_category_uses_selected_score_not_ranked_top() -> None:
    """Blocked top-ranked score must not leak into the selected candidate."""
    standard = ScoreResult(
        category=scoring.FeeCategory.STANDARD_COMMERCIAL,
        score=75,
        signals=(EvidenceSignal(code=scoring.EvidenceCode.HAS_PERCENTAGE_COLUMN, source=scoring.EvidenceSource.STRUCTURAL, weight=40),),
        blockers=(BlockerCode.ONLY_MONEY_FOR_PERCENTAGE_CATEGORY,),
    )
    fixed = ScoreResult(
        category=scoring.FeeCategory.FIXED_FEE,
        score=62,
        signals=(EvidenceSignal(code=scoring.EvidenceCode.HAS_MONEY_COLUMN, source=scoring.EvidenceSource.STRUCTURAL, weight=40),),
        blockers=(),
    )
    conversion = ScoreResult(
        category=scoring.FeeCategory.CURRENCY_CONVERSION,
        score=30,
        signals=(),
        blockers=(BlockerCode.ONLY_MONEY_FOR_PERCENTAGE_CATEGORY,),
    )
    international = ScoreResult(
        category=scoring.FeeCategory.INTERNATIONAL_SURCHARGE,
        score=25,
        signals=(),
        blockers=(BlockerCode.ONLY_MONEY_FOR_PERCENTAGE_CATEGORY,),
    )

    decision = scoring.select_category((standard, fixed, conversion, international))
    assert decision.status == "selected"
    assert decision.selected_category is scoring.FeeCategory.FIXED_FEE
    assert decision.selected_score is not None
    assert decision.selected_score.score == 62
    assert decision.selected_score.category == scoring.FeeCategory.FIXED_FEE
    assert decision.winner_margin is None


def test_standard_percentage_excludes_personal_row_order_independent() -> None:
    """Personal/friends-and-family rows must not win, regardless of row order."""
    forward = _table("Standard fees", [["Commercial payments", "2.99%"], ["Personal payments", "5.00%"]])
    reverse = _table("Standard fees", [["Personal payments", "5.00%"], ["Commercial payments", "2.99%"]])

    for table in (forward, reverse):
        profile = build_table_profile(table)
        decision = extract_standard_percentage(table, profile)
        assert decision.value == "2.99", f"failed for {table.caption}"


def test_standard_percentage_reports_conflict_for_equally_supported_values() -> None:
    """Two structurally identical commercial rows with different values are a conflict."""
    table = _table("Standard fees", [["Commercial payments A", "2.99%"], ["Commercial payments B", "3.49%"]])
    profile = build_table_profile(table)
    decision = extract_standard_percentage(table, profile)
    assert decision.value is None
    assert any(
        o.kind == ObservationKind.EXTRACTION_CONFLICT and "2.99" in o.message and "3.49" in o.message
        for o in decision.observations
    )


def test_fixed_fees_currency_label_column_and_no_duplicate() -> None:
    """EUR | 0.39 style tables extract and skip duplicate (currency, amount) pairs."""
    table = _table("Fixed fee by currency", [["EUR", "0.39"], ["USD", "0.49"], ["EUR", "0.39"]])
    profile = build_table_profile(table)
    decision = extract_fixed_fees(table, profile)
    assert decision.value is not None
    assert len(decision.value) == 2
    assert set(decision.value) == {FixedFees(currency="EUR", amount="0.39"), FixedFees(currency="USD", amount="0.49")}


def test_fixed_fees_currency_label_conflicting_value_is_conflict() -> None:
    """Conflicting amounts for the same currency label are reported as a conflict."""
    table = _table("Fixed fee by currency", [["EUR", "0.39"], ["EUR", "0.49"]])
    profile = build_table_profile(table)
    decision = extract_fixed_fees(table, profile)
    assert any(o.kind == ObservationKind.EXTRACTION_CONFLICT for o in decision.observations)


def test_conversion_spread_conflicting_rows_is_conflict() -> None:
    """A single conversion table with two distinct spreads must fail closed."""
    table = _table("Currency conversion spread", [["3.25%"], ["4.50%"]], document_id="FEETB539")
    profile = build_table_profile(table)
    decision = extract_conversion_spread(table, profile, has_approved_evidence=True)
    assert decision.value is None
    assert any(o.kind == ObservationKind.EXTRACTION_CONFLICT and "3.25" in o.message for o in decision.observations)


def test_conversion_spread_conflicting_tables_is_conflict() -> None:
    """Two conversion tables with different spreads must not silently choose one."""
    table_3 = _table("Currency conversion spread", [["3.25%"]], document_id="FEETB539")
    table_4 = _table("Currency conversion spread", [["4.50%"]], document_id="FEETB539")
    run = classify_structural([table_3, table_4])
    assert any(
        o.kind == ObservationKind.EXTRACTION_CONFLICT and "3.25" in o.message and "4.5" in o.message
        for o in run.observations
    )
    assert run.derived.currency_conversion is None
