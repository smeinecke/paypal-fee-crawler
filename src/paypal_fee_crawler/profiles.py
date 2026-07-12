"""Deterministic structural profiles and table context objects.

The classes in this module are intentionally side-effect-free and are used by
both the scoring engine and the component extractor to agree on a single,
stable representation of a table's shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict


class TableProfile(TypedDict):
    """Structural profile for a normalized table.

    All keys are computed deterministically from the ``Table`` rows, headers,
    and metadata.  No key should be populated from external state or mutable
    global caches.
    """

    row_count: int
    column_count: int
    row_percentages: frozenset[int]
    row_moneys: frozenset[int]
    mixed_rows: frozenset[int]
    percentage_columns: frozenset[int]
    money_columns: frozenset[int]
    currencies: frozenset[str]
    multiple_currencies: bool
    additive_percentages: bool
    metadata_keys: frozenset[str]
    internal_names: frozenset[str]
    content_types: frozenset[str]
    has_percentage: bool
    has_money: bool


@dataclass(frozen=True)
class TableContext:
    """Reference context preserved while traversing CMS components.

    ``TableContext`` is captured at the moment a table is created so that
    downstream classifiers can match referenced tables and their parents without
    re-parsing the page tree.
    """

    document_id: str | None = None
    component_id: str | None = None
    parent_path: tuple[str, ...] = ()
    section_path: tuple[str, ...] = ()
    source_table_ids: tuple[str, ...] = ()
    reference_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a shallow dict suitable for storage on ``Table.context``."""
        return {
            "document_id": self.document_id,
            "component_id": self.component_id,
            "parent_path": list(self.parent_path),
            "section_path": list(self.section_path),
            "source_table_ids": list(self.source_table_ids),
            "reference_id": self.reference_id,
        }

    @classmethod
    def from_table(cls, table: Any) -> TableContext:
        """Build a context from an existing table."""
        return cls(
            document_id=table.document_id,
            component_id=table.component_id,
            parent_path=tuple(table.parent_path or []),
            section_path=tuple(table.section_path or []),
            source_table_ids=tuple(table.source_table_ids or []),
            reference_id=table.reference_id,
        )
