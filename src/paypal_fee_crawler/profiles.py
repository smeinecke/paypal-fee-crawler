"""Minimal structural profile helpers retained for component extraction."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from .models import Table


class TableContext(BaseModel):
    """Reference context preserved for a physical or referenced table occurrence."""

    model_config = ConfigDict(frozen=True)

    component_id: str | None = None
    caption: str | None = None
    section_path: tuple[str, ...] = ()
    parent_path: tuple[str, ...] = ()
    source_order: int = 0
    reference_id: str | None = None

    @classmethod
    def from_table(cls, table: Table) -> TableContext:
        """Build a context from an existing table."""
        return cls(
            component_id=table.component_id,
            caption=table.caption,
            section_path=tuple(table.section_path or []),
            parent_path=tuple(table.parent_path or []),
            source_order=table.source_order,
            reference_id=table.reference_id,
        )


class NormalizedTableRecord(BaseModel):
    """A table together with the contexts where it appears."""

    model_config = ConfigDict(frozen=True)

    table: Table
    contexts: tuple[TableContext, ...]
