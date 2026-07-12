"""Tests for deterministic structural profiles and table contexts."""

from __future__ import annotations

from paypal_fee_crawler.models import Row, Table
from paypal_fee_crawler.pricing_tokens import render_rich_text_node
from paypal_fee_crawler.profiles import TableContext, build_table_profile


def _table(caption: str, rows: list[list[str]]) -> Table:
    return Table(
        caption=caption,
        section_path=[caption],
        component_id="c-1",
        document_id="FEETB001",
        source_order=1,
        parent_path=["section"],
        rows=[Row(cells=[render_rich_text_node(cell) for cell in row]) for row in rows],
    )


def test_build_table_profile_counts() -> None:
    table = _table("Commercial transaction fees", [["Commercial transactions", "2.99% + 0.39 EUR"]])
    profile = build_table_profile(table)
    assert profile.row_count == 1
    assert profile.column_count == 2
    assert profile.has_percentage
    assert profile.has_money
    assert profile.mixed_percentage_money_rows == {0}
    assert profile.percentage_columns == {1}
    assert profile.money_columns == {1}


def test_build_table_profile_currencies() -> None:
    table = _table("Fixed fee by received currency", [["EUR", "0.39 EUR"], ["USD", "0.49 USD"]])
    profile = build_table_profile(table)
    assert profile.currencies == {"EUR", "USD"}
    assert profile.has_multiple_currencies
    assert profile.money_columns == {1}
    assert profile.percentage_columns == set()
    assert profile.additive_percentage_count == 0


def test_row_profile_token_pattern() -> None:
    table = _table("Fixed fee by received currency", [["EUR", "0.39 EUR"], ["USD", "0.49 USD"]])
    profile = build_table_profile(table)
    assert len(profile.rows) == 2
    assert profile.rows[0].token_kind_pattern == ("text", "money")
    assert profile.rows[0].money_count == 1
    assert profile.rows[0].currencies == {"EUR"}


def test_column_profile_counts() -> None:
    table = _table("Commercial transaction fees", [["Commercial transactions", "2.99% + 0.39 EUR"]])
    profile = build_table_profile(table)
    assert len(profile.columns) == 2
    col0 = profile.columns[0]
    col1 = profile.columns[1]
    assert col0.percentage_row_count == 0
    assert col0.money_row_count == 0
    assert col0.text_row_count == 1
    assert col1.percentage_row_count == 1
    assert col1.money_row_count == 1
    assert col1.text_row_count == 0


def test_table_profile_from_context() -> None:
    table = _table("Commercial transaction fees", [["Commercial transactions", "2.99% + 0.39 EUR"]])
    context = TableContext(
        component_id="c-2",
        caption="override",
        section_path=["override"],
        parent_path=["root"],
        source_order=2,
        reference_id="ref-2",
    )
    profile = build_table_profile(table, contexts=(context,))
    assert profile.contexts == (context,)
    assert profile.contexts[0].component_id == "c-2"


def test_table_profile_default_context() -> None:
    table = _table("Commercial transaction fees", [["Commercial transactions", "2.99% + 0.39 EUR"]])
    profile = build_table_profile(table)
    assert len(profile.contexts) == 1
    assert profile.contexts[0].component_id == "c-1"
    assert profile.contexts[0].reference_id is None


def test_probable_header_and_note_detection() -> None:
    table = _table(
        "Multi row table",
        [
            ["Header", "Value"],
            ["EUR", "0.39 EUR"],
            ["* footnote text"],
        ],
    )
    profile = build_table_profile(table)
    assert profile.rows[0].is_probable_header
    assert not profile.rows[1].is_probable_header
    assert profile.rows[2].is_probable_note
    assert not profile.rows[0].is_probable_note


def test_additive_percentage_count() -> None:
    table = _table("International surcharge", [["EEA", "0%"], ["GB", "+1.29%"], ["Other", "+1.99%"]])
    profile = build_table_profile(table)
    assert profile.additive_percentage_count == 2
    assert profile.has_additive_percentages
    assert profile.rows[1].additive_percentage_count == 1


def test_empty_table_profile() -> None:
    table = _table("Empty", [])
    profile = build_table_profile(table)
    assert profile.row_count == 0
    assert profile.column_count == 0
    assert not profile.has_percentage
    assert not profile.has_money
    assert profile.rows == ()
    assert profile.columns == ()
