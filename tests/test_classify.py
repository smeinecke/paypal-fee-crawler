"""Tests for the rule-based transaction fee classifier."""

from __future__ import annotations

from pathlib import Path

from paypal_fee_crawler.classify import (
    classify_tables,
)
from paypal_fee_crawler.cms_context import extract_cms_context
from paypal_fee_crawler.components import ComponentsExtractor
from paypal_fee_crawler.models import Cell, DerivedFeeResult, Row, Table, TableHeader
from paypal_fee_crawler.pricing_tokens import tokenize_text


def _table(caption: str, rows: list[list[str]]) -> Table:
    return Table(
        document_id="DOC-1",
        caption=caption,
        headers=[TableHeader(text=c) for c in ["Product", "Fee"]],
        rows=[
            Row(
                cells=[Cell(text=cell, tokens=tokenize_text(cell)) for cell in row],
            )
            for row in rows
        ],
    )


def test_classify_commercial_rate_table_extracts_rules() -> None:
    table = _table(
        "Standardgebühr beim Empfang von Inlandstransaktionen",
        [
            ["PayPal Checkout", "2.99% + 0.39 EUR"],
            ["Alle anderen geschäftlichen Transaktionen", "2.99% + 0.39 EUR"],
        ],
    )
    result = classify_tables([table])
    ids = {r.id for r in result.transaction_fee_rules}
    assert "paypal_checkout" in ids
    assert "other_commercial" in ids


def test_classify_fixed_fee_table_extracts_schedule() -> None:
    table = _table(
        "Festgebühr bei geschäftlichen Transaktionen",
        [
            ["EUR", "0.39 EUR"],
            ["USD", "0.49 USD"],
        ],
    )
    result = classify_tables([table])
    schedule = result.fixed_fee_schedules.get("commercial")
    assert schedule is not None
    assert schedule.entries.get("EUR") == "0.39"
    assert schedule.entries.get("USD") == "0.49"


def test_classify_resolves_reference_to_online_card_schedule() -> None:
    commercial = _table(
        "Standardgebühr beim Empfang von Inlandstransaktionen",
        [
            [
                "Zahlungen mit Kredit- und Debitkarten mit erweiterten Funktionen",
                "Es gelten die Gebühren für Online-Kartenzahlungen",
            ],
            ["PayPal Checkout", "2.99% + 0.39 EUR"],
        ],
    )
    online_card = _table(
        "Empfang von Inlandstransaktionen über die PayPal-Dienste für Online-Zahlungen",
        [
            ["Zahlungen mit Kredit- und Debitkarten mit erweiterten Funktionen", "2.99% + 0.39 EUR"],
        ],
    )
    result = classify_tables([commercial, online_card])
    rule = next(r for r in result.transaction_fee_rules if r.id == "advanced_card_payments")
    assert rule.rate_reference is not None
    assert rule.rate_reference.reference == "online_card_payments.advanced"
    assert rule.rate_reference.resolved_rate is not None
    assert rule.rate_reference.resolved_rate.percentage == "2.99"


def test_classify_detects_ambiguous_product() -> None:
    table = _table(
        "Standardgebühr beim Empfang von Inlandstransaktionen",
        [
            ["Rückbuchung oder Rückzahlung", "0.30 EUR"],
        ],
    )
    result = classify_tables([table])
    assert len(result.ambiguous_rows) == 1
    assert "chargebacks" in result.ambiguous_rows[0].candidates
    assert "refunds" in result.ambiguous_rows[0].candidates


def _germany_result() -> DerivedFeeResult:
    text = Path("tests/fixtures/paypal-de-real.html").read_text(encoding="utf-8")
    cms = extract_cms_context(text)
    sections, tables, warnings = ComponentsExtractor().extract(cms)
    return classify_tables(tables)


def test_germany_core_rules() -> None:
    result = _germany_result()
    ids = {r.id for r in result.transaction_fee_rules}
    assert "paypal_checkout" in ids
    assert "goods_and_services" in ids
    assert "advanced_card_payments" in ids
    assert result.fixed_fee_schedules
    assert result.international_surcharge_schedules
    assert result.status in {"complete", "partial"}


def test_germany_advanced_card_variants() -> None:
    result = _germany_result()
    advanced_rules = [r for r in result.transaction_fee_rules if r.id == "advanced_card_payments"]
    assert advanced_rules
    standard = next(r for r in advanced_rules if r.variant_id == "standard")
    assert standard.percentage == "2.99"
    assert standard.fixed_fee_schedule == "advanced_card_payments"
    assert standard.international_surcharge_schedule == "advanced_card_payments"
    assert standard.rate_reference is not None
    assert standard.rate_reference.resolved_rate is not None
    assert standard.rate_reference.resolved_rate.percentage == "2.99"

    assert {r.variant_id for r in advanced_rules} >= {"standard", "donations", "eterminal"}
    donations = next(r for r in advanced_rules if r.variant_id == "donations")
    assert donations.percentage == "2.49"
    assert donations.conditions.get("transaction_purpose") == "donation"
    eterminal = next(r for r in advanced_rules if r.variant_id == "eterminal")
    assert eterminal.percentage == "3.39"
    assert eterminal.conditions.get("authorization_channel") == "terminal"


def test_germany_apm_variants() -> None:
    result = _germany_result()
    apm_rules = {r.variant_id: r for r in result.transaction_fee_rules if r.id == "alternative_payment_methods"}
    assert "default" in apm_rules
    assert "special" in apm_rules
    assert apm_rules["default"].percentage == "2.99"
    assert apm_rules["special"].percentage == "5.49"
    assert apm_rules["special"].conditions.get("payment_methods") == [
        "gopay",
        "latvian_online_bank_transfer",
        "lithuanian_online_bank_transfer",
        "ovo_premium",
        "skrill",
        "thai_online_bank_transfer",
    ]


def test_schedule_inheritance_from_commercial_when_allowed() -> None:
    """Product-specific schedules may be explicitly inherited from commercial."""
    commercial = _table(
        "Standardgebühr beim Empfang von Inlandstransaktionen",
        [
            ["PayPal Checkout", "2.99% + 0.39 EUR"],
            ["Alle anderen geschäftlichen Transaktionen", "2.99% + 0.39 EUR"],
        ],
    )
    fixed_fee = _table(
        "Festgebühr bei geschäftlichen Transaktionen",
        [
            ["EUR", "0.39 EUR"],
        ],
    )
    intl_surcharge = _table(
        "Prozentuale Zusatzgebühr für internationale geschäftliche Transaktionen",
        [
            ["EUR", "0.00%"],
        ],
    )
    result = classify_tables([commercial, fixed_fee, intl_surcharge])
    assert result.coverage_summary.inherited_schedules == 4
    inherited = [d for d in result.diagnostics if d.type == "inherited_schedule"]
    assert len(inherited) == 4
    assert all(d.inherited_from == "commercial" for d in inherited)
    for rule in result.transaction_fee_rules:
        if rule.id == "paypal_checkout":
            assert rule.fixed_fee_schedule == "paypal_checkout"
            assert rule.international_surcharge_schedule == "paypal_checkout"
            assert result.fixed_fee_schedules["paypal_checkout"].origin == "inherited"
            assert result.international_surcharge_schedules["paypal_checkout"].origin == "inherited"
        if rule.id == "other_commercial":
            assert rule.fixed_fee_schedule == "other_commercial"
            assert rule.international_surcharge_schedule == "other_commercial"
            assert result.fixed_fee_schedules["other_commercial"].origin == "inherited"
            assert result.international_surcharge_schedules["other_commercial"].origin == "inherited"


def test_nested_reference_schedule_validated() -> None:
    """Resolved references must not carry dangling schedule references."""
    commercial = _table(
        "Standardgebühr beim Empfang von Inlandstransaktionen",
        [
            [
                "Zahlungen mit Kredit- und Debitkarten mit erweiterten Funktionen",
                "Es gelten die Gebühren für Online-Kartenzahlungen",
            ],
            ["PayPal Checkout", "2.99% + 0.39 EUR"],
        ],
    )
    online_card = _table(
        "Empfang von Inlandstransaktionen über die PayPal-Dienste für Online-Zahlungen",
        [
            ["Zahlungen mit Kredit- und Debitkarten mit erweiterten Funktionen", "2.99% + 0.39 EUR"],
        ],
    )
    result = classify_tables([commercial, online_card])
    rule = next(r for r in result.transaction_fee_rules if r.id == "advanced_card_payments")
    assert rule.rate_reference is not None
    assert rule.rate_reference.resolved_rate is not None
    # The referenced online-card rule uses a fixed-fee schedule; the commercial
    # advanced-card row should not retain a nested schedule that is not present.
    if rule.rate_reference.resolved_rate.fixed_fee_schedule:
        assert rule.rate_reference.resolved_rate.fixed_fee_schedule in result.fixed_fee_schedules


def test_nacionales_keyword_does_not_match_internacionales() -> None:
    # "internacionales" (international) contains the substring "nacionales" but
    # must not be classified as a domestic commercial rate table.
    from paypal_fee_crawler.classify import _classify_table_category

    table = _table(
        "Recepción de transacciones internacionales",
        [],
    )
    assert _classify_table_category(table) is None


def test_comision_porcentual_adicional_classified_as_surcharge() -> None:
    from paypal_fee_crawler.classify import _classify_table_category

    table = _table(
        "Comisión porcentual adicional por transacciones comerciales internacionales",
        [],
    )
    assert _classify_table_category(table) == "international_surcharge_table"


def _micropayment_result(currency: str, fixed_amount: str) -> DerivedFeeResult:
    rate_table = _table(
        "Micropayments",
        [
            ["Domestic micropayments", f"5% + {fixed_amount} {currency}"],
            ["International micropayments", f"6% + {fixed_amount} {currency}"],
        ],
    )
    fixed_fee = _table(
        "Micropayments fixed fee",
        [
            [currency, f"{fixed_amount} {currency}"],
        ],
    )
    return classify_tables([rate_table, fixed_fee])


def test_micropayments_domestic_and_international_use_direct_base_schedule() -> None:
    """Variant-specific micropayment rules must use the direct micropayments fixed-fee schedule."""
    result = _micropayment_result("GBP", "0.05")
    domestic = next(r for r in result.transaction_fee_rules if r.id == "micropayments" and r.variant_id == "domestic")
    international = next(
        r for r in result.transaction_fee_rules if r.id == "micropayments" and r.variant_id == "international"
    )
    assert domestic.percentage == "5"
    assert international.percentage == "6"
    assert domestic.fixed_fee_schedule == "micropayments"
    assert international.fixed_fee_schedule == "micropayments"
    assert result.fixed_fee_schedules["micropayments"].entries["GBP"] == "0.05"
    assert result.fixed_fee_schedules["micropayments"].origin == "direct"
    assert "micropayments_domestic" not in result.fixed_fee_schedules
    assert "micropayments_international" not in result.fixed_fee_schedules


def test_micropayments_fall_back_to_product_base_before_commercial() -> None:
    """A direct micropayments fixed-fee schedule must win over commercial inheritance."""
    rate_table = _table(
        "Micropayments",
        [
            ["Domestic micropayments", "5% + 0.05 GBP"],
        ],
    )
    micropayments_fee = _table(
        "Micropayments fixed fee",
        [
            ["GBP", "0.05 GBP"],
        ],
    )
    commercial_fee = _table(
        "Commercial fixed fee",
        [
            ["GBP", "0.30 GBP"],
        ],
    )
    result = classify_tables([rate_table, micropayments_fee, commercial_fee])
    rule = next(r for r in result.transaction_fee_rules if r.id == "micropayments" and r.variant_id == "domestic")
    assert rule.fixed_fee_schedule == "micropayments"
    assert result.fixed_fee_schedules["micropayments"].entries["GBP"] == "0.05"


def test_micropayments_inr_fixed_fee() -> None:
    """IN international micropayments use the direct INR micropayments schedule."""
    rate_table = _table(
        "Micropayments",
        [
            ["International micropayments", "6% + 0.25 INR"],
        ],
    )
    fixed_fee = _table(
        "Micropayments fixed fee",
        [
            ["INR", "0.25 INR"],
        ],
    )
    result = classify_tables([rate_table, fixed_fee])
    rule = next(r for r in result.transaction_fee_rules if r.id == "micropayments" and r.variant_id == "international")
    assert rule.percentage == "6"
    assert rule.fixed_fee_schedule == "micropayments"
    assert result.fixed_fee_schedules["micropayments"].entries["INR"] == "0.25"
