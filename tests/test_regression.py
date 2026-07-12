"""Regression tests for the structural classifier merge gate."""

from __future__ import annotations

import pytest

from paypal_fee_crawler import scoring
from paypal_fee_crawler.classify import TableDecision, classify_structural
from paypal_fee_crawler.comparison import compare_runs
from paypal_fee_crawler.exceptions import RegistryValidationError
from paypal_fee_crawler.extraction import (
    ObservationKind,
    extract_conversion_spread,
    extract_fixed_fees,
    extract_standard_percentage,
)
from paypal_fee_crawler.models import FixedFees, Market, Row, Table
from paypal_fee_crawler.pricing_tokens import render_rich_text_node
from paypal_fee_crawler.profiles import NormalizedTableRecord, TableContext, build_table_profile
from paypal_fee_crawler.registry import ClusterRecord, ClusterStatus, FingerprintBuilder, FingerprintRegistry
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
        signals=(
            EvidenceSignal(
                code=scoring.EvidenceCode.HAS_PERCENTAGE_COLUMN, source=scoring.EvidenceSource.STRUCTURAL, weight=40
            ),
        ),
        blockers=(BlockerCode.ONLY_MONEY_FOR_PERCENTAGE_CATEGORY,),
    )
    fixed = ScoreResult(
        category=scoring.FeeCategory.FIXED_FEE,
        score=62,
        signals=(
            EvidenceSignal(
                code=scoring.EvidenceCode.HAS_MONEY_COLUMN, source=scoring.EvidenceSource.STRUCTURAL, weight=40
            ),
        ),
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


def test_preserved_contexts_influence_scoring() -> None:
    """A reference context must provide both lexical and relationship evidence."""
    table = _table("", [["2.99% + 0.39 EUR"]])
    record = NormalizedTableRecord(
        table=table,
        contexts=(
            TableContext(
                caption="Commercial transaction fees",
                reference_id="ref-1",
                section_path=("Fees",),
                parent_path=("Commercial",),
            ),
        ),
    )
    run = classify_structural([record])
    decision = run.table_decisions[0]
    assert decision.selected_category == scoring.FeeCategory.STANDARD_COMMERCIAL
    assert decision.status == "selected"
    assert any(s.code == scoring.EvidenceCode.POSITIVE_LEXICAL_HINT for s in decision.ranked_scores[0].signals)
    assert any(s.code == scoring.EvidenceCode.REFERENCE_CONTEXT_MATCH for s in decision.ranked_scores[0].signals)


def test_generic_structural_shapes_require_contextual_support() -> None:
    """Generic money/percentage shapes without context must not be classified."""
    products = _table("Products", [["EUR", "10.00 EUR"], ["USD", "12.00 USD"]])
    pricing = _table("Pricing", [["Plan A", "10% + 5.00 EUR"]])
    run = classify_structural([products, pricing])
    assert run.derived.status == "unclassified"
    assert run.derived.commercial_fixed_fees == []
    assert run.derived.standard_commercial is None


def _conversion_registry_for(table: Table) -> FingerprintRegistry:
    """Return a registry with an approved conversion cluster for *table*."""
    profile = build_table_profile(table)
    fingerprint = str(FingerprintBuilder.build(profile, table))
    cluster = ClusterRecord(
        name="approved-conversion",
        category="currency_conversion",
        fingerprints=frozenset({fingerprint}),
        document_ids=frozenset(),
        required_features=frozenset(),
        reviewed_examples=frozenset(),
        status=ClusterStatus.APPROVED,
    )
    return FingerprintRegistry({"approved-conversion": cluster})


def test_approved_conversion_fingerprint_works_end_to_end() -> None:
    """An opaque percentage table with an approved conversion fingerprint extracts a spread."""
    table = _table("Currency conversion spread", [["3.25%"]])
    registry = _conversion_registry_for(table)
    run = classify_structural([table], registry=registry)
    assert run.derived.currency_conversion is not None
    assert run.derived.currency_conversion.spread_percentage == "3.25"


def test_registry_feature_validation_fails_closed() -> None:
    """A registry with an unknown required feature must not silently approve."""
    table = _table("Currency conversion spread", [["3.25%"]])
    profile = build_table_profile(table)
    fingerprint = str(FingerprintBuilder.build(profile, table))
    cluster = ClusterRecord(
        name="bad-conversion",
        category="currency_conversion",
        fingerprints=frozenset({fingerprint}),
        document_ids=frozenset(),
        required_features=frozenset({"has_moneey_typo"}),
        reviewed_examples=frozenset(),
        status=ClusterStatus.APPROVED,
    )
    registry = FingerprintRegistry({"bad-conversion": cluster})
    with pytest.raises(RegistryValidationError):
        classify_structural([table], registry=registry)


def test_cross_table_fixed_fees_deduplicate() -> None:
    """Identical fixed-fee tables across a page must not duplicate output values."""
    table = _table("Fixed fee by currency", [["EUR", "0.39 EUR"], ["USD", "0.49 USD"]], document_id="FEETB18")
    run = classify_structural([table, table])
    assert len(run.derived.commercial_fixed_fees) == 2
    currencies = {f.currency for f in run.derived.commercial_fixed_fees}
    assert currencies == {"EUR", "USD"}


def test_cross_table_surcharges_deduplicate() -> None:
    """Identical international-surcharge tables must not duplicate output values."""
    table = _table("International surcharge", [["EEA", "0%"], ["Other", "+1.99%"]], document_id="FEETB91")
    run = classify_structural([table, table])
    regions = {s.region for s in run.derived.international_surcharges}
    assert regions == {"EEA", "OTHER"}


def test_table_decision_preserves_diagnostics() -> None:
    """Unclassified and ambiguous decisions must retain ranked scores and reasons."""
    table = _table("Random table", [["A", "B"], ["C", "D"]])
    run = classify_structural([table])
    decision = run.table_decisions[0]
    assert decision.status == "unclassified"
    assert decision.ambiguity_reason is not None
    assert len(decision.ranked_scores) == 4

    ambiguous = _table("Ambiguous", [["10% + 5.00 EUR"]])
    run = classify_structural([ambiguous])
    decision = run.table_decisions[0]
    assert decision.status in ("selected", "ambiguous", "unclassified")
    assert decision.ranked_scores


def test_low_margin_uses_winner_margin() -> None:
    """LOW_MARGIN must reflect the distance to the runner-up, not the absolute score."""
    # A close runner-up should trigger a low-margin warning.
    table = Table(caption="Standard fees", rows=[Row(cells=[render_rich_text_node("2.99% + 1.50%")])])
    record = NormalizedTableRecord(table=table, contexts=(TableContext(reference_id="ref-1"),))
    run = classify_structural([record])
    assert run.derived.status == "partial"
    assert any(o.kind == ObservationKind.LOW_MARGIN for o in run.observations)

    # A single eligible category with no runner-up should not trigger LOW_MARGIN, even if score < 75.
    fixed = Table(
        caption="Fixed fee",
        rows=[
            Row(cells=[render_rich_text_node("EUR"), render_rich_text_node("0.39 EUR")]),
            Row(cells=[render_rich_text_node("USD"), render_rich_text_node("0.49 USD")]),
        ],
    )
    fixed_record = NormalizedTableRecord(table=fixed, contexts=(TableContext(reference_id="ref-2"),))
    run = classify_structural([fixed_record])
    assert run.derived.status == "partial"
    assert run.derived.commercial_fixed_fees
    assert not any(o.kind == ObservationKind.LOW_MARGIN for o in run.observations)


def test_comparison_selected_categories_come_from_table_decisions() -> None:
    """A derived result with only fixed fees must not be reported as standard-commercial."""
    table = _table("Fixed fee", [["EUR", "0.39"]], document_id="FEETB18")
    run = classify_structural([table])
    # Compare the run against itself; the only selected category should be fixed_fee.
    comparison = compare_runs(run, run, Market(paypal_market_code="DE", country_code="DE", country_name="Germany"))
    assert comparison.selected_categories_match
    assert comparison.legacy_selected_categories == ("fixed_fee",)
    assert "standard_commercial" not in comparison.legacy_selected_categories


def test_comparison_table_decision_key_uses_priority_identity() -> None:
    """Both-selected-None decisions are ignored and stable identity is used."""
    from paypal_fee_crawler.comparison import _compare_table_decisions

    decision = TableDecision(
        table_id="table-1",
        document_id=None,
        component_id=None,
        fingerprint="sha256:abc",
        selected_category=None,
        selected_score=None,
        status="unclassified",
        ambiguity_reason="no eligible category",
        winner_margin=None,
        ranked_scores=(),
        blockers=(),
        evidence_codes=(),
        evidence_sources=(),
    )
    assert _compare_table_decisions((decision,), (decision,)) == ()
    assert _compare_table_decisions((decision,), ()) == ()
