"""Tests for rich-text rendering."""

from __future__ import annotations

from paypal_fee_crawler.pricing_tokens import render_rich_text_node


def test_render_plain_text() -> None:
    cell = render_rich_text_node(
        {
            "nodeType": "document",
            "content": [
                {"nodeType": "paragraph", "content": [{"nodeType": "text", "value": "2,99% + fixed fee", "marks": []}]}
            ],
        }
    )
    assert cell.text == "2,99% + fixed fee"
    assert len(cell.tokens) == 1
    assert cell.tokens[0].kind == "percentage"


def test_render_hyperlink() -> None:
    cell = render_rich_text_node(
        {
            "nodeType": "document",
            "content": [
                {
                    "nodeType": "paragraph",
                    "content": [
                        {"nodeType": "text", "value": "See ", "marks": []},
                        {
                            "nodeType": "hyperlink",
                            "data": {"uri": "#fixed-fee"},
                            "content": [{"nodeType": "text", "value": "fixed fee", "marks": []}],
                        },
                    ],
                }
            ],
        }
    )
    assert cell.text == "See fixed fee"
    assert len(cell.links) == 1
    assert cell.links[0].uri == "#fixed-fee"


def test_render_non_breaking_space() -> None:
    cell = render_rich_text_node(
        {
            "nodeType": "document",
            "content": [
                {"nodeType": "paragraph", "content": [{"nodeType": "text", "value": "0.39\u00a0EUR", "marks": []}]}
            ],
        }
    )
    assert cell.text == "0.39 EUR"


def test_render_empty_node() -> None:
    cell = render_rich_text_node({"nodeType": "document", "content": []})
    assert cell.text == ""
