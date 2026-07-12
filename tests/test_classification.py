"""Tests for core-fee classification."""

from __future__ import annotations

from paypal_fee_crawler import scoring
from paypal_fee_crawler.classify import classify_legacy, classify_structural, classify_tables
from paypal_fee_crawler.models import Row, Table
from paypal_fee_crawler.pricing_tokens import render_rich_text_node
from paypal_fee_crawler.registry import ClusterRecord, ClusterStatus, FingerprintRegistry


def _table(
    caption: str, rows: list[list[str]], section_path: list[str] | None = None, document_id: str | None = None
) -> Table:
    return Table(
        caption=caption,
        section_path=section_path or [caption],
        document_id=document_id,
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


def test_classify_structural_standard_commercial() -> None:
    tables = [
        _table("Commercial transaction fees", [["Commercial transactions", "2.99% + 0.39 EUR"]], document_id="FEETB16"),
        _table("Fixed fee by received currency", [["EUR", "0.39 EUR"], ["USD", "0.49 USD"]], document_id="FEETB18"),
    ]
    run = classify_structural(tables)
    assert run.derived.status == "complete"
    assert run.derived.standard_commercial is not None
    assert run.derived.standard_commercial.percentage == "2.99"
    assert any(fee.currency == "EUR" for fee in run.derived.commercial_fixed_fees)
    assert run.classifier_version == "structural-1"


def test_classify_structural_international_surcharge() -> None:
    tables = [
        _table(
            "International surcharge", [["EEA", "0%"], ["GB", "+1.29%"], ["Other", "+1.99%"]], document_id="FEETB91"
        ),
    ]
    run = classify_structural(tables)
    assert run.derived.status == "partial"
    regions = {s.region: s.percentage_points for s in run.derived.international_surcharges}
    assert regions.get("GB") == "1.29"
    assert regions.get("OTHER") == "1.99"


def test_classify_structural_unclassified_when_uncertain() -> None:
    tables = [_table("Random table", [["A", "B"], ["C", "D"]])]
    run = classify_structural(tables)
    assert run.derived.status == "unclassified"


def test_classify_legacy_returns_run() -> None:
    tables = [
        _table("Commercial transaction fees", [["Commercial transactions", "2.99% + 0.39 EUR"]]),
        _table("Fixed fee by received currency", [["EUR", "0.39 EUR"], ["USD", "0.49 USD"]]),
    ]
    run = classify_legacy(tables)
    assert run.classifier_version == "legacy"
    assert run.derived.status == "complete"


def test_score_standard_commercial_vector() -> None:
    table = _table(
        "Commercial transaction fees", [["Commercial transactions", "2.99% + 0.39 EUR"]], document_id="FEETB16"
    )
    result = scoring.score_standard_commercial(table)
    assert result.category == scoring.FeeCategory.STANDARD_COMMERCIAL
    assert result.score >= scoring.MINIMUM_SCORE
    assert result.eligible
    assert any(s.code == scoring.EvidenceCode.HAS_PERCENTAGE_COLUMN for s in result.signals)
    assert any(s.code == scoring.EvidenceCode.HAS_MIXED_PERCENT_MONEY_ROW for s in result.signals)


def test_score_fixed_fee_vector() -> None:
    table = _table("Fixed fee by received currency", [["EUR", "0.39 EUR"], ["USD", "0.49 USD"]], document_id="FEETB18")
    result = scoring.score_fixed_fee(table)
    assert result.category == scoring.FeeCategory.FIXED_FEE
    assert result.score >= scoring.MINIMUM_SCORE
    assert result.eligible
    assert any(s.code == scoring.EvidenceCode.HAS_MONEY_COLUMN for s in result.signals)
    assert any(s.code == scoring.EvidenceCode.HAS_MULTIPLE_CURRENCIES for s in result.signals)


def test_score_international_surcharge_vector() -> None:
    table = _table(
        "International surcharge", [["EEA", "0%"], ["GB", "+1.29%"], ["Other", "+1.99%"]], document_id="FEETB91"
    )
    result = scoring.score_international_surcharge(table)
    assert result.category == scoring.FeeCategory.INTERNATIONAL_SURCHARGE
    assert result.score >= scoring.MINIMUM_SCORE
    assert result.eligible
    assert any(s.code == scoring.EvidenceCode.HAS_PERCENTAGE_COLUMN for s in result.signals)


def test_select_category_ambiguous() -> None:
    results = (
        scoring.ScoreResult(scoring.FeeCategory.STANDARD_COMMERCIAL, 55, (), ()),
        scoring.ScoreResult(scoring.FeeCategory.FIXED_FEE, 50, (), ()),
        scoring.ScoreResult(scoring.FeeCategory.INTERNATIONAL_SURCHARGE, 0, (), ()),
        scoring.ScoreResult(scoring.FeeCategory.CURRENCY_CONVERSION, 0, (), ()),
    )
    decision = scoring.select_category(results)
    assert decision.status == "unclassified"
    assert decision.selected_category is None


def test_select_category_margin_ambiguous() -> None:
    results = (
        scoring.ScoreResult(scoring.FeeCategory.STANDARD_COMMERCIAL, 80, (), ()),
        scoring.ScoreResult(scoring.FeeCategory.FIXED_FEE, 70, (), ()),
        scoring.ScoreResult(scoring.FeeCategory.INTERNATIONAL_SURCHARGE, 0, (), ()),
        scoring.ScoreResult(scoring.FeeCategory.CURRENCY_CONVERSION, 0, (), ()),
    )
    decision = scoring.select_category(results)
    assert decision.status == "ambiguous"
    assert decision.selected_category is None


def test_select_category_selected() -> None:
    results = (
        scoring.ScoreResult(scoring.FeeCategory.STANDARD_COMMERCIAL, 80, (), ()),
        scoring.ScoreResult(scoring.FeeCategory.FIXED_FEE, 50, (), ()),
        scoring.ScoreResult(scoring.FeeCategory.INTERNATIONAL_SURCHARGE, 0, (), ()),
        scoring.ScoreResult(scoring.FeeCategory.CURRENCY_CONVERSION, 0, (), ()),
    )
    decision = scoring.select_category(results)
    assert decision.status == "selected"
    assert decision.selected_category == scoring.FeeCategory.STANDARD_COMMERCIAL
    assert decision.winner_margin == 30


def test_market_code_matches_respects_boundaries() -> None:
    assert scoring.market_code_matches("US", "US")
    assert scoring.market_code_matches("US-CA", "US")
    assert scoring.market_code_matches("US_CA", "US")
    assert scoring.market_code_matches("US$", "US")
    assert not scoring.market_code_matches("business", "US")
    assert not scoring.market_code_matches("usaus", "US")


def test_market_code_matches_aliases() -> None:
    assert scoring.market_code_matches("UK", "GB")
    assert scoring.market_code_matches("uk", "GB")
    assert not scoring.market_code_matches("united kingdom", "GB")


def test_region_from_text_grouped_regions() -> None:
    assert scoring.region_from_text("US") == "US_CA"
    assert scoring.region_from_text("Canada") == "US_CA"
    assert scoring.region_from_text("USA") == "US_CA"
    assert scoring.region_from_text("GB") == "GB"
    assert scoring.region_from_text("EEA") == "EEA"
    assert scoring.region_from_text("Other") == "OTHER"


def test_registry_approved_fingerprint_boosts_score() -> None:
    from paypal_fee_crawler.profiles import build_table_profile
    from paypal_fee_crawler.registry import FingerprintBuilder

    table = _table("Commercial transaction fees", [["Commercial transactions", "2.99% + 0.39 EUR"]])
    scores = scoring.score_all_categories(table)
    standard_score = next(s for s in scores if s.category == scoring.FeeCategory.STANDARD_COMMERCIAL)
    assert standard_score.signals[0].code == scoring.EvidenceCode.HAS_PERCENTAGE_COLUMN

    profile = build_table_profile(table)
    fingerprint = str(FingerprintBuilder.build(profile, table))
    registry = FingerprintRegistry(
        {
            "commercial": ClusterRecord(
                name="commercial",
                category="standard_commercial",
                fingerprints=frozenset({fingerprint}),
                document_ids=frozenset(),
                required_features=frozenset(),
                reviewed_examples=frozenset(),
                status=ClusterStatus.APPROVED,
            )
        }
    )
    scores = scoring.score_all_categories(table, registry=registry)
    standard_score = next(s for s in scores if s.category == scoring.FeeCategory.STANDARD_COMMERCIAL)
    assert scoring.EvidenceCode.KNOWN_FINGERPRINT in [s.code for s in standard_score.signals]


def test_registry_document_id_blocks_incompatible_category() -> None:
    table = _table("Commercial transaction fees", [["Commercial transactions", "2.99% + 0.39 EUR"]])
    table = table.model_copy(update={"document_id": "FEETB18"})
    registry = FingerprintRegistry(
        {
            "fixed": ClusterRecord(
                name="fixed",
                category="fixed_fee",
                fingerprints=frozenset(),
                document_ids=frozenset({"FEETB18"}),
                required_features=frozenset(),
                reviewed_examples=frozenset(),
                status=ClusterStatus.APPROVED,
            )
        }
    )
    scores = scoring.score_all_categories(table, registry=registry)
    standard_score = next(s for s in scores if s.category == scoring.FeeCategory.STANDARD_COMMERCIAL)
    assert scoring.BlockerCode.INCOMPATIBLE_FINGERPRINT in standard_score.blockers

    fixed_score = next(s for s in scores if s.category == scoring.FeeCategory.FIXED_FEE)
    assert scoring.EvidenceCode.KNOWN_DOCUMENT_ID in [s.code for s in fixed_score.signals]
