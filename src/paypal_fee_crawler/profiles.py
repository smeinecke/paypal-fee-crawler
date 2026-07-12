"""Deterministic structural profiles and table context objects.

The classes in this module are intentionally side-effect-free and are used by
both the scoring engine and the component extractor to agree on a single,
stable representation of a table's shape.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from .models import Cell, FeeToken, Row, Table


class TableContext(BaseModel):
    """Reference context preserved for a physical or referenced table occurrence.

    ``TableContext`` is captured at the moment a table is created so that
    downstream classifiers and the fingerprint registry can match referenced
    tables and their parents without re-parsing the page tree.
    """

    model_config = ConfigDict(frozen=True)

    component_id: str | None = None
    caption: str | None = None
    section_path: list[str] = []
    parent_path: list[str] = []
    source_order: int = 0
    reference_id: str | None = None


@dataclass(frozen=True)
class RowProfile:
    """Structural profile for a single table row."""

    row_index: int
    cell_count: int
    percentage_count: int
    money_count: int
    currencies: frozenset[str]
    additive_percentage_count: int
    fee_data_keys: frozenset[str]
    internal_names: frozenset[str]
    content_types: frozenset[str]
    token_kind_pattern: tuple[str, ...]
    is_probable_header: bool
    is_probable_note: bool


@dataclass(frozen=True)
class ColumnProfile:
    """Structural profile for a single table column."""

    column_index: int
    percentage_row_count: int
    money_row_count: int
    text_row_count: int
    currencies: frozenset[str]
    token_kind_pattern: tuple[str, ...]


@dataclass(frozen=True)
class TableProfile:
    """Structural profile for a normalized table.

    All fields are computed deterministically from the ``Table`` rows,
    headers, and metadata.  No field is populated from external state or
    mutable global caches.
    """

    row_count: int
    column_count: int

    rows: tuple[RowProfile, ...]
    columns: tuple[ColumnProfile, ...]

    percentage_rows: frozenset[int]
    money_rows: frozenset[int]
    mixed_percentage_money_rows: frozenset[int]

    percentage_columns: frozenset[int]
    money_columns: frozenset[int]

    currencies: frozenset[str]
    additive_percentage_count: int

    fee_data_keys: frozenset[str]
    internal_names: frozenset[str]
    content_types: frozenset[str]

    document_id: str | None = None
    source_table_ids: tuple[str, ...] = ()
    contexts: tuple[TableContext, ...] = ()

    @property
    def has_percentage(self) -> bool:
        return bool(self.percentage_rows)

    @property
    def has_money(self) -> bool:
        return bool(self.money_rows)

    @property
    def has_multiple_currencies(self) -> bool:
        return len(self.currencies) > 1

    @property
    def has_additive_percentages(self) -> bool:
        return self.additive_percentage_count > 0


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------


def _cell_token_kinds(cell: Cell) -> tuple[str, ...]:
    return tuple(token.kind for token in cell.tokens)


def _cell_kind_string(cell: Cell) -> str:
    if cell.tokens:
        return " ".join(token.kind for token in cell.tokens)
    if cell.text.strip():
        return "text"
    return "empty"


def _cell_has_kind(cell: Cell, kind: str) -> bool:
    return any(token.kind == kind for token in cell.tokens)


def _percentage_count(row: Row) -> int:
    return sum(1 for cell in row.cells for token in cell.tokens if token.kind == "percentage")


def _money_count(row: Row) -> int:
    return sum(1 for cell in row.cells for token in cell.tokens if token.kind == "money")


def _has_percentage(row: Row) -> bool:
    return _percentage_count(row) > 0


def _has_money(row: Row) -> bool:
    return _money_count(row) > 0


def _collect_tokens(table: Table) -> list[FeeToken]:
    tokens: list[FeeToken] = []
    for header in table.headers:
        tokens.extend(header.tokens)
    for row in table.rows:
        for cell in row.cells:
            tokens.extend(cell.tokens)
    return tokens


def _additive_percentage_count(row: Row) -> int:
    """Count percentage tokens that are explicitly additive or appear in pairs."""
    pct_tokens = [
        token for cell in row.cells for token in cell.tokens if token.kind == "percentage"
    ]
    if len(pct_tokens) > 1:
        return len(pct_tokens)
    if len(pct_tokens) == 1 and pct_tokens[0].operator == "add":
        return 1
    return 0


def _is_probable_header(row_index: int, row: Row) -> bool:
    """A header row is typically the first row with no value-bearing tokens."""
    if row_index != 0:
        return False
    if row.cells and all(_cell_has_kind(cell, "text") for cell in row.cells):
        return True
    return not any(_cell_has_kind(cell, "percentage") or _cell_has_kind(cell, "money") for cell in row.cells)


def _is_probable_note(row_index: int, row: Row, row_count: int) -> bool:
    """A note/footer row has no value-bearing tokens and appears after data rows."""
    if row_index == 0:
        return False
    has_values = any(
        _cell_has_kind(cell, "percentage") or _cell_has_kind(cell, "money")
        for cell in row.cells
    )
    if has_values:
        return False
    return row_index >= row_count - 1 or bool(
        row.cells and all(len(cell.text.strip()) < 80 for cell in row.cells)
    )


def _token_metadata_sets(tokens: list[FeeToken]) -> tuple[set[str], set[str], set[str]]:
    keys: set[str] = set()
    names: set[str] = set()
    content_types: set[str] = set()
    for token in tokens:
        if token.fee_data_key:
            keys.add(token.fee_data_key.lower())
        if token.internal_name:
            names.add(token.internal_name.lower())
        if token.content_type:
            content_types.add(token.content_type.lower())
    return keys, names, content_types


def build_table_profile(table: Table, context: TableContext | None = None) -> TableProfile:
    """Return a deterministic structural profile for *table*."""
    rows = table.rows
    num_rows = len(rows)
    num_cols = max((len(row.cells) for row in rows), default=0) or table.column_count or 0

    row_profiles: list[RowProfile] = []
    percentage_rows: set[int] = set()
    money_rows: set[int] = set()
    mixed_rows: set[int] = set()
    currencies: set[str] = set()
    additive_count = 0

    for idx, row in enumerate(rows):
        pct_count = _percentage_count(row)
        mon_count = _money_count(row)
        row_currencies = {
            token.currency
            for cell in row.cells
            for token in cell.tokens
            if token.kind == "money" and token.currency
        }
        currencies.update(row_currencies)
        if pct_count:
            percentage_rows.add(idx)
        if mon_count:
            money_rows.add(idx)
        if pct_count and mon_count:
            mixed_rows.add(idx)

        row_tokens = [token for cell in row.cells for token in cell.tokens]
        keys, names, cts = _token_metadata_sets(row_tokens)
        add_pct = _additive_percentage_count(row)
        additive_count += add_pct

        row_profiles.append(
            RowProfile(
                row_index=idx,
                cell_count=len(row.cells),
                percentage_count=pct_count,
                money_count=mon_count,
                currencies=frozenset(row_currencies),
                additive_percentage_count=add_pct,
                fee_data_keys=frozenset(keys),
                internal_names=frozenset(names),
                content_types=frozenset(cts),
                token_kind_pattern=tuple(_cell_kind_string(cell) for cell in row.cells),
                is_probable_header=_is_probable_header(idx, row),
                is_probable_note=_is_probable_note(idx, row, num_rows),
            )
        )

    all_tokens = _collect_tokens(table)
    table_keys, table_names, table_cts = _token_metadata_sets(all_tokens)

    percentage_columns: set[int] = set()
    money_columns: set[int] = set()
    column_profiles: list[ColumnProfile] = []

    for col in range(num_cols):
        pct_count = 0
        mon_count = 0
        text_count = 0
        col_currencies: set[str] = set()
        col_pattern: list[str] = []
        for row in rows:
            if col < len(row.cells):
                cell = row.cells[col]
                cell_pcts = sum(1 for token in cell.tokens if token.kind == "percentage")
                cell_mons = sum(1 for token in cell.tokens if token.kind == "money")
                cell_texts = int(cell.text.strip() and not cell_pcts and not cell_mons)
                if cell_pcts:
                    pct_count += 1
                if cell_mons:
                    mon_count += 1
                if cell_texts and not cell_pcts and not cell_mons:
                    text_count += 1
                col_currencies.update(
                    token.currency
                    for token in cell.tokens
                    if token.kind == "money" and token.currency
                )
                col_pattern.append(_cell_kind_string(cell))
            else:
                col_pattern.append("missing")
        if pct_count > 0 and pct_count / max(num_rows, 1) > 0.5:
            percentage_columns.add(col)
        if mon_count > 0 and mon_count / max(num_rows, 1) > 0.5:
            money_columns.add(col)
        column_profiles.append(
            ColumnProfile(
                column_index=col,
                percentage_row_count=pct_count,
                money_row_count=mon_count,
                text_row_count=text_count,
                currencies=frozenset(col_currencies),
                token_kind_pattern=tuple(col_pattern),
            )
        )

    contexts: tuple[TableContext, ...] = (context,) if context is not None else ()
    if not contexts:
        contexts = (
            TableContext(
                component_id=table.component_id,
                caption=table.caption,
                section_path=list(table.section_path or []),
                parent_path=list(table.parent_path or []),
                source_order=table.source_order,
                reference_id=table.reference_id,
            ),
        )

    return TableProfile(
        row_count=num_rows,
        column_count=num_cols,
        rows=tuple(row_profiles),
        columns=tuple(column_profiles),
        percentage_rows=frozenset(percentage_rows),
        money_rows=frozenset(money_rows),
        mixed_percentage_money_rows=frozenset(mixed_rows),
        percentage_columns=frozenset(percentage_columns),
        money_columns=frozenset(money_columns),
        currencies=frozenset(currencies),
        additive_percentage_count=additive_count,
        fee_data_keys=frozenset(table_keys),
        internal_names=frozenset(table_names),
        content_types=frozenset(table_cts),
        document_id=table.document_id,
        source_table_ids=tuple(table.source_table_ids or []),
        contexts=contexts,
    )
