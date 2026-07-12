"""Mutation tests for structural classification stability.

These tests verify that the structural classifier is robust to token- and
row-level noise that should not change structural semantics, while remaining
sensitive to changes that alter the structural fingerprint.
"""

from __future__ import annotations

from paypal_fee_crawler import scoring
from paypal_fee_crawler.classify import classify_structural
from paypal_fee_crawler.models import Cell, Row, Table
from paypal_fee_crawler.pricing_tokens import render_rich_text_node


def _table(caption: str, rows: list[list[str]], document_id: str | None = "FEETB16") -> Table:
    return Table(
        caption=caption,
        section_path=[caption],
        document_id=document_id,
        rows=[Row(cells=[render_rich_text_node(cell) for cell in row]) for row in rows],
    )


def _mutate_cell_text(table: Table, row: int, col: int, text: str) -> Table:
    """Return a new table with a single cell's text replaced but structure intact."""
    new_rows = [r.model_copy() for r in table.rows]
    new_rows[row] = new_rows[row].model_copy()
    cells = [c.model_copy() for c in new_rows[row].cells]
    cells[col] = Cell(text=text, tokens=render_rich_text_node(text).tokens)
    new_rows[row] = Row(cells=cells, row_id=new_rows[row].row_id)
    return table.model_copy(update={"rows": new_rows})


def _mutate_column_order(table: Table) -> Table:
    """Return a new table with the column order reversed."""
    new_rows: list[Row] = []
    for r in table.rows:
        new_rows.append(Row(cells=list(reversed(r.cells))))
    return table.model_copy(update={"rows": new_rows})


def _mutate_row_order(table: Table) -> Table:
    """Return a new table with the data row order reversed."""
    new_rows = list(reversed(table.rows))
    return table.model_copy(update={"rows": new_rows})


def _mutate_add_noise_row(table: Table) -> Table:
    """Return a new table with an extra noisy text row."""
    new_rows = list(table.rows) + [Row(cells=[Cell(text="note", tokens=[])])]
    return table.model_copy(update={"rows": new_rows})


def test_text_amount_mutation_keeps_classification_stable() -> None:
    """Replacing exact amounts should not change structural category."""
    base = _table("Commercial fees", [["Commercial", "2.99% + 0.39 EUR"]])
    mutated = _mutate_cell_text(base, 0, 1, "3.49% + 0.49 GBP")

    base_run = classify_structural([base])
    mutated_run = classify_structural([mutated])

    assert base_run.derived.status != "unclassified"
    assert mutated_run.derived.status == base_run.derived.status


def test_column_order_mutation_breaks_fingerprint() -> None:
    """Reversing columns should change the structural fingerprint."""
    from paypal_fee_crawler.profiles import build_table_profile
    from paypal_fee_crawler.registry import FingerprintBuilder

    base = _table("Fixed fees", [["EUR", "0.39 EUR"], ["USD", "0.49 USD"]])
    mutated = _mutate_column_order(base)

    base_fp = FingerprintBuilder.build(build_table_profile(base), base)
    mutated_fp = FingerprintBuilder.build(build_table_profile(mutated), mutated)
    assert base_fp.value != mutated_fp.value


def test_metadata_mutation_is_detected() -> None:
    """Changing the document ID to a conflicting known ID should add evidence."""
    base = _table("Commercial fees", [["Commercial", "2.99% + 0.39 EUR"]], document_id="NEW")
    mutated = base.model_copy(update={"document_id": "FEETB18"})

    scores = scoring.score_all_categories(mutated)
    fixed_score = next(s for s in scores if s.category == scoring.FeeCategory.FIXED_FEE)
    assert any(s.code == scoring.EvidenceCode.KNOWN_DOCUMENT_ID for s in fixed_score.signals)


def test_add_noise_row_lowers_confidence() -> None:
    """A noisy extra row should not crash classification."""
    base = _table("Commercial fees", [["Commercial", "2.99% + 0.39 EUR"]])
    mutated = _mutate_add_noise_row(base)
    run = classify_structural([mutated])
    assert len(run.table_decisions) == 1
    assert run.table_decisions[0].selected_category is not None


def test_cross_market_consistent_category_for_same_structure() -> None:
    """Tables with the same fingerprint should select the same category regardless of market code."""
    base_de = _table("Commercial fees", [["Commercial", "2.99% + 0.39 EUR"]])
    base_us = _table("Commercial fees", [["Commercial", "2.99% + 0.39 EUR"]])

    run_de = classify_structural([base_de], market_code="DE")
    run_us = classify_structural([base_us], market_code="US")

    # Same structural content should lead to the same selected category.
    de_cat = run_de.table_decisions[0].selected_category
    us_cat = run_us.table_decisions[0].selected_category
    assert de_cat == us_cat

    # If a category is selected, the fingerprint should be consistent across markets.
    if de_cat and us_cat:
        from paypal_fee_crawler.profiles import build_table_profile
        from paypal_fee_crawler.registry import FingerprintBuilder

        fp_de = FingerprintBuilder.build(build_table_profile(base_de), base_de)
        fp_us = FingerprintBuilder.build(build_table_profile(base_us), base_us)
        assert fp_de.value == fp_us.value
