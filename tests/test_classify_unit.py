"""Unit tests for small classifier helpers introduced in recent tuning."""

from __future__ import annotations

from paypal_fee_crawler.classify import (
    _conditions_for_row,
    _extract_amount_condition,
    _pricing_plan_for_label,
    _variant_for_withdrawals,
)


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
