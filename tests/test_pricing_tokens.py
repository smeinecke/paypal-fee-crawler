"""Tests for pricing-token normalization."""

from __future__ import annotations

from paypal_fee_crawler.pricing_tokens import normalize_pricing_token, render_rich_text_node, tokenize_text


def test_percentage_token() -> None:
    token = normalize_pricing_token("2,99%")
    assert token.kind == "percentage"
    assert token.value == "2.99"
    assert token.raw == "2,99%"


def test_money_token() -> None:
    token = normalize_pricing_token("0.39 EUR")
    assert token.kind == "money"
    assert token.amount == "0.39"
    assert token.currency == "EUR"


def test_adjustment_token() -> None:
    token = normalize_pricing_token("+1,29%")
    assert token.kind == "percentage"
    assert token.value == "1.29"
    assert token.operator == "add"


def test_unclassified_text() -> None:
    token = normalize_pricing_token("no minimum fee")
    assert token.kind == "text"
    assert token.raw == "no minimum fee"


def test_non_breaking_space() -> None:
    token = normalize_pricing_token("0.39\u00a0EUR")
    assert token.kind == "money"
    assert token.currency == "EUR"


def test_invalid_currency_rejected() -> None:
    token = normalize_pricing_token("0.39 XYZ")
    assert token.kind == "text"


def test_token_preserves_raw() -> None:
    raw = "2,99% + fixed fee"
    token = normalize_pricing_token(raw)
    assert token.raw == raw


def test_tokenize_text_extracts_multiple() -> None:
    tokens = tokenize_text("2.99% + 0.39 EUR")
    kinds = [t.kind for t in tokens]
    assert "percentage" in kinds
    assert "money" in kinds


def test_tokenize_text_ignores_bare_numbers() -> None:
    tokens = tokenize_text("within 5 days")
    assert all(t.kind == "text" for t in tokens)


def test_render_rich_text_document() -> None:
    cell = render_rich_text_node(
        {
            "nodeType": "document",
            "content": [{"nodeType": "paragraph", "content": [{"nodeType": "text", "value": "2,99%", "marks": []}]}],
        }
    )
    assert cell.text == "2,99%"
    assert any(t.kind == "percentage" for t in cell.tokens)


def test_render_rich_text_list() -> None:
    cell = render_rich_text_node(
        {
            "nodeType": "list",
            "content": [{"nodeType": "list-item", "content": [{"nodeType": "text", "value": "0.39 EUR", "marks": []}]}],
        }
    )
    assert any(t.kind == "money" for t in cell.tokens)


def test_render_rich_text_embedded_entry() -> None:
    cell = render_rich_text_node(
        {
            "nodeType": "EmbeddedEntryBlock",
            "data": {
                "target": {
                    "sys": {"id": "tok-1", "contentType": {"sys": {"id": "cvPricingToken"}}},
                    "fields": {"feeDataKey": "2.99%", "value": "2.99%", "internalName": "pct"},
                }
            },
        }
    )
    assert cell.text == "2.99%"
    assert cell.tokens[0].kind == "percentage"


def test_render_rich_text_embedded_entry_inline() -> None:
    cell = render_rich_text_node(
        {
            "nodeType": "embedded-entry-inline",
            "data": {
                "target": {
                    "sys": {"id": "tok-2", "contentType": {"sys": {"id": "cvPricingToken"}}},
                    "fields": {"feeDataKey": "0.39 EUR", "value": "0.39 EUR", "internalName": "fixed"},
                }
            },
        }
    )
    assert cell.text == "0.39 EUR"
    assert cell.tokens[0].kind == "money"
    assert cell.tokens[0].amount == "0.39"
    assert cell.tokens[0].currency == "EUR"
