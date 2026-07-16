"""Unit tests for small classifier helpers introduced in recent tuning."""

from __future__ import annotations

from paypal_fee_crawler.classify import (
    _condition_score,
    _conditions_for_row,
    _conditions_match_for_reference,
    _detect_reference,
    _extract_amount_condition,
    _market_code_from_url,
    _pricing_plan_for_label,
    _resolve_reference,
    _variant_for_withdrawals,
)
from paypal_fee_crawler.comparison import ComparisonResult, compare_runs
from paypal_fee_crawler.models import Cell, Row, Source, TransactionFeeRule


def test_pricing_plan_for_label_detects_interchange_plus_plus() -> None:
    assert _pricing_plan_for_label("Interchange Plus Plus Fee") == "interchange_plus_plus"


def test_pricing_plan_for_label_detects_interchange_plus() -> None:
    assert _pricing_plan_for_label("Interchange Plus Fee Structure") == "interchange_plus"


def test_pricing_plan_for_label_detects_blended_italian() -> None:
    assert _pricing_plan_for_label("Tariffe secondo il Piano tariffario misto") == "blended"


def test_pricing_plan_for_label_detects_flat_rate() -> None:
    assert _pricing_plan_for_label("Flat Rate Pricing") == "blended"


def test_variant_for_withdrawals_matches_wire_transfer() -> None:
    assert _variant_for_withdrawals("", "wire transfer", "", "", [], False, False) == "wire_transfer"


def test_variant_for_withdrawals_falls_back_to_standard() -> None:
    assert _variant_for_withdrawals("", "some withdrawal", "", "", [], False, False) == "standard"


def test_conditions_for_row_eterminal_sets_pricing_plan() -> None:
    conditions = _conditions_for_row(
        "advanced_card_payments",
        "eterminal",
        "Virtual Terminal - Blended Pricing: Visa, MasterCard",
        methods=[],
        table=None,
    )
    assert conditions["authorization_channel"] == "terminal"
    assert conditions["point_of_sale"] is True
    assert conditions["pricing_plan"] == "blended"


def test_conditions_for_row_fx_service_spread() -> None:
    conditions = _conditions_for_row(
        "advanced_card_payments",
        "fx_service",
        "Foreign Exchange Spread",
        methods=[],
        table=None,
    )
    assert conditions["service"] == "fx_spread"


def test_conditions_for_row_fx_service_as_a_service() -> None:
    conditions = _conditions_for_row(
        "advanced_card_payments",
        "fx_service",
        "Foreign Exchange as a Service",
        methods=[],
        table=None,
    )
    assert conditions["service"] == "fx_as_a_service"


def test_conditions_for_row_withdrawals_sets_method() -> None:
    conditions = _conditions_for_row(
        "withdrawals",
        "bank_account",
        "Withdraw to a bank account",
        methods=[],
        table=None,
    )
    assert conditions["withdrawal_method"] == "bank_account"


def test_extract_amount_condition_with_currency() -> None:
    condition = _extract_amount_condition("Below 10.00 EUR")
    assert condition == {"operator": "lt", "value": "10", "currency": "EUR"}


def test_extract_amount_condition_without_currency() -> None:
    condition = _extract_amount_condition("Above 1000")
    assert condition == {"operator": "gt", "value": "1000"}


def test_conditions_match_for_reference_superset_markets() -> None:
    rule = {"applies_to_markets": ["MY", "SG"], "transaction_region": "domestic"}
    source = {"applies_to_markets": ["SG"], "transaction_region": "domestic"}
    assert _conditions_match_for_reference(rule, source) is True


def test_conditions_match_for_reference_payment_method_subset() -> None:
    rule = {"payment_methods": ["american_express"], "transaction_region": "domestic"}
    source = {"payment_methods": ["american_express"], "transaction_region": "domestic"}
    assert _conditions_match_for_reference(rule, source) is True


def test_conditions_match_for_reference_rejects_region_mismatch() -> None:
    rule = {"transaction_region": "domestic"}
    source = {"transaction_region": "international"}
    assert _conditions_match_for_reference(rule, source) is False


def test_detect_reference_ignores_single_cell_header() -> None:
    row = Row(cells=[Cell(text="Reembolsos de transacciones comerciales")])
    assert _detect_reference(row, "other_commercial") is None


def test_market_code_from_url_extracts_country_code() -> None:
    assert _market_code_from_url("https://www.paypal.com/de/business/paypal-business-fees") == "DE"
    assert _market_code_from_url("https://www.paypal.com/c2/business/paypal-business-fees") == "C2"
    assert _market_code_from_url("https://www.paypal.com/") is None


def test_conditions_match_for_reference_rejects_unsolicited_methods() -> None:
    rule = {"payment_methods": ["american_express"], "transaction_region": "domestic"}
    source = {"transaction_region": "domestic"}
    assert _conditions_match_for_reference(rule, source) is False


def test_condition_score_prefers_matching_keys_and_penalises_extras() -> None:
    standard = TransactionFeeRule(
        id="advanced_card_payments",
        variant_id="standard",
        conditions={"transaction_region": "domestic"},
    )
    terminal = TransactionFeeRule(
        id="advanced_card_payments",
        variant_id="eterminal",
        conditions={"transaction_region": "domestic", "authorization_channel": "terminal", "point_of_sale": True},
    )
    source = {"transaction_region": "domestic"}
    assert _condition_score(standard, source) > _condition_score(terminal, source)


def test_resolve_reference_prefers_specific_market_match() -> None:
    rules = [
        TransactionFeeRule(
            id="other_commercial",
            variant_id="standard",
            percentage="3.4",
            conditions={"applies_to_markets": ["SG"], "transaction_region": "domestic"},
        ),
        TransactionFeeRule(
            id="other_commercial",
            variant_id="standard",
            percentage="3.9",
            conditions={"applies_to_markets": ["MY", "SG"], "transaction_region": "domestic"},
        ),
    ]
    source = Source(requested_url="https://www.paypal.com/sg/business/paypal-business-fees")
    resolved, ambiguous = _resolve_reference(
        "other_commercial",
        rules,
        source_variant_id="standard",
        source_conditions={"applies_to_markets": ["SG"], "transaction_region": "domestic"},
        source=source,
    )
    assert ambiguous is False
    assert resolved is not None
    assert resolved.percentage == "3.4"


def test_resolve_reference_relaxes_region_for_payment_method_match() -> None:
    rules = [
        TransactionFeeRule(
            id="advanced_card_payments",
            variant_id="american_express",
            percentage="3.5",
            conditions={"payment_methods": ["american_express"], "transaction_region": "domestic"},
        ),
    ]
    source = Source(requested_url="https://www.paypal.com/ca/business/paypal-business-fees")
    resolved, ambiguous = _resolve_reference(
        "advanced_card_payments",
        rules,
        source_variant_id="american_express",
        source_conditions={"payment_methods": ["american_express"], "transaction_region": "international"},
        source=source,
    )
    assert ambiguous is False
    assert resolved is not None
    assert resolved.percentage == "3.5"


def test_resolve_reference_prefers_exact_market_over_all_other() -> None:
    rules = [
        TransactionFeeRule(
            id="other_commercial",
            variant_id="standard",
            percentage="2.9",
            conditions={"applies_to_markets": ["all_other_markets"], "transaction_region": "domestic"},
        ),
        TransactionFeeRule(
            id="other_commercial",
            variant_id="standard",
            percentage="1.9",
            conditions={"applies_to_markets": ["ES"], "transaction_region": "domestic"},
        ),
    ]
    source = Source(requested_url="https://www.paypal.com/es/business/paypal-business-fees")
    resolved, ambiguous = _resolve_reference(
        "other_commercial",
        rules,
        source_variant_id="standard",
        source_conditions={"transaction_region": "domestic"},
        source=source,
    )
    assert ambiguous is False
    assert resolved is not None
    assert resolved.percentage == "1.9"


def test_condition_score_penalises_all_other_markets() -> None:
    all_other = TransactionFeeRule(
        id="other_commercial",
        variant_id="standard",
        conditions={"applies_to_markets": ["all_other_markets"], "transaction_region": "domestic"},
    )
    specific = TransactionFeeRule(
        id="other_commercial",
        variant_id="standard",
        conditions={"applies_to_markets": ["ES"], "transaction_region": "domestic"},
    )
    source = {"applies_to_markets": ["ES"], "transaction_region": "domestic"}
    assert _condition_score(specific, source) > _condition_score(all_other, source)


def test_compare_runs_returns_empty_legacy_result() -> None:
    assert compare_runs() == ComparisonResult()
