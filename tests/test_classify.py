"""Tests for the rule-based transaction fee classifier."""

from __future__ import annotations

from pathlib import Path

from paypal_fee_crawler.classify import (
    classify_tables,
)
from paypal_fee_crawler.cms_context import extract_cms_context
from paypal_fee_crawler.components import ComponentsExtractor
from paypal_fee_crawler.models import Cell, Row, Table, TableHeader
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
    assert schedule.model_extra.get("EUR") == "0.39"
    assert schedule.model_extra.get("USD") == "0.49"


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


def test_classify_real_germany_fixture() -> None:
    text = Path("tests/fixtures/paypal-de-real.html").read_text(encoding="utf-8")
    cms = extract_cms_context(text)
    sections, tables, warnings = ComponentsExtractor().extract(cms)
    result = classify_tables(tables)
    ids = {r.id for r in result.transaction_fee_rules}
    assert "paypal_checkout" in ids
    assert "goods_and_services" in ids
    assert "advanced_card_payments" in ids
    assert result.fixed_fee_schedules
    assert result.international_surcharge_schedules
    assert result.status in {"complete", "partial"}
