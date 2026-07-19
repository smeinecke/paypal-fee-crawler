"""Unit tests for small classifier helpers introduced in recent tuning."""

from __future__ import annotations

from typing import Any

from paypal_fee_crawler.classify import (
    _build_direct_fixed_rules,
    _cell_looks_like_fee_cell,
    _classify_product_or_apm,
    _condition_score,
    _conditions_for_row,
    _conditions_match_for_reference,
    _detect_reference,
    _extract_amount_condition,
    _extract_direct_fixed_amounts,
    _handle_unusable_rate_row,
    _has_likely_numeric_fee_candidate,
    _is_apm_special_label,
    _market_code_from_url,
    _parse_canonical_amount,
    _pricing_plan_for_label,
    _resolve_reference,
    _variant_for_withdrawals,
)
from paypal_fee_crawler.models import Cell, Row, Source, Table, TableHeader, TransactionFeeRule
from paypal_fee_crawler.pricing_tokens import tokenize_text


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


def _row(cells: list[str]) -> Row:
    return Row(cells=[Cell(text=c, tokens=tokenize_text(c)) for c in cells])


def _table(caption: str, headers: list[str], rows: list[list[str]]) -> Table:
    return Table(
        document_id="DOC-1",
        caption=caption,
        headers=[TableHeader(text=h) for h in headers],
        rows=[_row(r) for r in rows],
    )


def test_is_apm_special_label_not_misled_by_withdrawal_return() -> None:
    """A withdrawal/return row must not be mistaken for an APM special method."""
    label = "Bank Return on Withdrawal/Transfer out of PayPal"
    assert _is_apm_special_label(label) is False
    product, _ = _classify_product_or_apm(label)
    assert product == "withdrawals"


def test_extract_direct_fixed_amounts_handles_zero_fee() -> None:
    """Explicit zero-fee rows for direct fixed-fee products become amount 0."""
    table = _table("Withdrawals", ["Product", "Fee"], [["Bank account", "No Fee"]])
    source = Source(requested_url="https://www.paypal.com/in/business/paypal-business-fees")
    amounts = _extract_direct_fixed_amounts(_row(["Bank account", "No Fee"]), "withdrawals", table, source)
    assert amounts == [("0", "INR", "bank_account")]


def test_numeric_fee_candidate_detected_in_last_cell() -> None:
    """A numeric value in the last fee cell is counted as a numeric fee candidate."""
    table = _table("Other Fees", ["Product", "Fee"], [["Bank Return", "250.00 INR"]])
    row = _row(["Bank Return", "250.00 INR"])
    assert _has_likely_numeric_fee_candidate(row, table) is True
    assert _cell_looks_like_fee_cell("Fee", table) is True


def test_numeric_fee_candidate_ignored_for_non_fee_header() -> None:
    """A number under a non-fee header should not be treated as a fee candidate."""
    table = _table("Market Codes", ["Market", "Code"], [["Germany", "DE"]])
    row = _row(["Germany", "DE"])
    assert _has_likely_numeric_fee_candidate(row, table) is False


def test_parse_canonical_amount_understands_separators() -> None:
    assert _parse_canonical_amount("50,000.00") == "50000"
    assert _parse_canonical_amount("50.000,00") == "50000"
    assert _parse_canonical_amount("1,000") == "1000"
    assert _parse_canonical_amount("0,35") == "0.35"
    assert _parse_canonical_amount("10,00") == "10"
    assert _parse_canonical_amount("1000000") == "1000000"
    assert _parse_canonical_amount("no number") is None


def test_extract_direct_fixed_amounts_parses_idr_bank_return() -> None:
    """A thousands-separated IDR bank-return amount is parsed as one fixed fee."""
    table = _table(
        "Bank Return on Withdrawal/Transfer out of PayPal",
        ["Market/Region", "Rate"],
        [["ID", "50,000.00 IDR"]],
    )
    amounts = _extract_direct_fixed_amounts(table.rows[0], "withdrawals", table, None)
    assert amounts == [("50000", "IDR", "bank_return")]


def test_extract_direct_fixed_amounts_records_request_multi_currency() -> None:
    """Records Request carrying GBP and EUR yields two standard variant amounts."""
    table = _table(
        "Other Fees",
        ["Activity", "Rate"],
        [["Records Request", "10,00 GBP or 12,00 EUR (per item)"]],
    )
    amounts = _extract_direct_fixed_amounts(table.rows[0], "records_request", table, None)
    assert amounts == [("10", "GBP", "standard"), ("12", "EUR", "standard")]


def test_extract_direct_fixed_amounts_sepa_italian_variants() -> None:
    """An Italian SEPA cell with standard and instant settlement yields two variant amounts."""
    table = _table(
        "Ricezione",
        ["Tipo di pagamento", "Tariffa"],
        [
            [
                "Addebito diretto SEPA",
                "0,35 EUR/transazione (liquidazione standard)0,40 EUR/transazione (liquidazione istantanea)",
            ]
        ],
    )
    amounts = _extract_direct_fixed_amounts(table.rows[0], "sepa_direct_debit", table, None)
    assert amounts == [
        ("0.35", "EUR", "standard_settlement"),
        ("0.4", "EUR", "instant_settlement"),
    ]


def test_build_direct_fixed_rules_adds_fee_currency_when_multi_currency_same_variant() -> None:
    """Multiple currencies for the same variant receive a fee_currency condition."""
    table = _table(
        "Other Fees",
        ["Activity", "Rate"],
        [["Records Request", "10,00 GBP or 12,00 EUR"]],
    )
    row = table.rows[0]
    direct_amounts = [("10", "GBP", "standard"), ("12", "EUR", "standard")]
    rules = _build_direct_fixed_rules(
        row=row,
        row_index=0,
        product_id="records_request",
        fallback_variant_id="standard",
        label="Records Request",
        methods=[],
        table=table,
        source=None,
        direct_amounts=direct_amounts,
    )
    assert len(rules) == 2
    gbp_rule = [r for r in rules if r.fee_components[0].currency == "GBP"][0]
    eur_rule = [r for r in rules if r.fee_components[0].currency == "EUR"][0]
    assert gbp_rule.conditions["fee_currency"] == "GBP"
    assert eur_rule.conditions["fee_currency"] == "EUR"


def test_build_direct_fixed_rules_no_incomplete_placeholder_for_sepa_variants() -> None:
    """SEPA standard/instant split produces two complete, distinct rules."""
    table = _table(
        "SEPA",
        ["Product", "Rate"],
        [["SEPA Direct Debit", "0.35 EUR (standard)0.40 EUR (instant)"]],
    )
    row = table.rows[0]
    direct_amounts = [("0.35", "EUR", "standard_settlement"), ("0.4", "EUR", "instant_settlement")]
    rules = _build_direct_fixed_rules(
        row=row,
        row_index=0,
        product_id="sepa_direct_debit",
        fallback_variant_id="standard",
        label="SEPA Direct Debit",
        methods=[],
        table=table,
        source=None,
        direct_amounts=direct_amounts,
    )
    assert len(rules) == 2
    assert all(r.fee_components[0].amount in {"0.35", "0.4"} for r in rules)
    assert {r.variant_id for r in rules} == {"standard_settlement", "instant_settlement"}


def test_handle_unusable_rate_row_buckets_numeric_candidates() -> None:
    """Rows with a numeric candidate and no usable rate become unclassified."""
    table = _table("Other Fees", ["Product", "Fee"], [["Bank Return", "250.00 PHP"]])
    row = table.rows[0]
    unclassified: list[Any] = []
    ignored: list[Any] = []
    _handle_unusable_rate_row(row, 0, "Bank Return", "withdrawals", table, None, unclassified, ignored)
    assert len(unclassified) == 1
    assert unclassified[0].reason == "unsupported_fee_shape"


def test_variant_for_withdrawals_finds_bank_return_from_table_context() -> None:
    """A bare row label in a bank-return table still resolves the bank_return variant."""
    variant = _variant_for_withdrawals(
        "ID",
        "id",
        "bank return on withdrawal transfer out of paypal",
        "id bank return on withdrawal transfer out of paypal",
        [],
        False,
        False,
    )
    assert variant == "bank_return"
