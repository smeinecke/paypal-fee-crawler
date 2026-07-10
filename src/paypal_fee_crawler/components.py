"""Recursive traversal and normalization of PayPal CMS components."""

from __future__ import annotations

import logging
from typing import Any

from .models import Cell, ParserWarning, Row, Section, Table, TableHeader
from .pricing_tokens import render_rich_text_node

logger = logging.getLogger(__name__)


class ComponentsExtractor:
    """Extract normalized sections and tables from a PayPal CMS context."""

    def __init__(self) -> None:
        self.tables: list[Table] = []
        self.sections: list[Section] = []
        self.warnings: list[ParserWarning] = []
        self._table_by_id: dict[str, Table] = {}
        self._path: list[str] = []

    def extract(self, cms: dict[str, Any]) -> tuple[list[Section], list[Table], list[ParserWarning]]:
        """Return (sections, tables, warnings)."""
        self.tables = []
        self.sections = []
        self.warnings = []
        self._table_by_id = {}
        self._path = []

        page_ref = cms.get("pageReference") or {}
        page_model = page_ref.get("pageModel") or {}
        middle = page_model.get("middle") or page_model.get("content") or []
        if not isinstance(middle, list):
            middle = [middle]

        for component in middle:
            self._traverse(component)

        # Resolve references after all tables are indexed.
        self._resolve_references()
        return self.sections, self.tables, self.warnings

    def _traverse(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        component_type = data.get("componentType") or data.get("type")
        component_id = data.get("componentId") or data.get("id")
        document_id = data.get("documentId") or data.get("feeTableDocumentId")

        # Mark component_id and document_id as used for diagnostics.
        _ = component_id, document_id

        # Capture section headings to build the section path.
        heading = self._extract_heading(data)
        if heading:
            self._path.append(heading)

        if component_type in {"FeeTableSection", "FeeTable", "FeeTableReference", "FeeTableRow"}:
            self._handle_table_component(data, component_type)
        elif component_type in {
            "TextSectionType",
            "TextGroup",
            "TextHeaderInner",
            "FeatureNavigationSection",
            "PopoverModal",
            "SubNav",
            "Button",
        }:
            self._handle_generic_section(data, component_type)

        # Recurse into nested content. Preserve path for direct children only.
        children = self._get_children(data)
        for child in children:
            self._traverse(child)

        if heading:
            self._path.pop()

    def _get_children(self, data: dict[str, Any]) -> list[Any]:
        """Return child components that should be traversed recursively."""
        children: list[Any] = []
        content = data.get("content") or data.get("fields") or {}
        if isinstance(content, dict):
            for value in content.values():
                if isinstance(value, list):
                    children.extend(value)
                elif isinstance(value, dict):
                    children.append(value)
        # Also check common nested containers.
        for key in ("middle", "top", "bottom", "components", "items", "children", "rows", "columns"):
            value = data.get(key)
            if isinstance(value, list):
                children.extend(value)
            elif isinstance(value, dict):
                children.append(value)
        return children

    def _extract_heading(self, data: dict[str, Any]) -> str | None:
        """Extract a human-readable heading from a section component."""
        for key in ("heading", "title", "displayName", "name", "caption"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        content = data.get("content") or data.get("fields") or {}
        if isinstance(content, dict):
            for key in ("heading", "title", "displayName", "name", "caption"):
                value = content.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, dict):
                    rendered = render_rich_text_node(value)
                    if rendered.text.strip():
                        return rendered.text.strip()
        return None

    def _handle_generic_section(self, data: dict[str, Any], component_type: str | Any) -> None:
        heading = self._extract_heading(data)
        body = ""
        content = data.get("content") or data.get("fields") or {}
        if isinstance(content, dict):
            for key in ("body", "text", "description"):
                node = content.get(key)
                if isinstance(node, dict):
                    rendered = render_rich_text_node(node)
                    if rendered.text.strip():
                        body = rendered.text.strip()
                        break
                elif isinstance(node, str) and node.strip():
                    body = node.strip()
                    break
        self.sections.append(
            Section(
                component_id=data.get("componentId") or data.get("id"),
                component_type=str(component_type),
                heading=heading,
                body=body or None,
                section_path=list(self._path),
            )
        )

    def _handle_table_component(self, data: dict[str, Any], component_type: str | Any) -> None:
        document_id = data.get("documentId") or data.get("feeTableDocumentId")
        component_id = data.get("componentId") or data.get("id")
        caption = self._extract_table_caption(data)

        if component_type == "FeeTableReference" and document_id:
            # Defer resolution; store a placeholder that references the target.
            self.tables.append(
                Table(
                    component_type="FeeTableReference",
                    document_id=str(document_id),
                    component_id=component_id,
                    caption=caption,
                    section_path=list(self._path),
                    source_table_ids=[str(document_id)],
                )
            )
            return

        if component_type == "FeeTable":
            table = self._build_table(data, document_id, component_id, caption)
            if table is not None:
                # Merge duplicate document IDs safely instead of overwriting.
                if table.document_id and table.document_id in self._table_by_id:
                    existing = self._table_by_id[table.document_id]
                    merged = self._merge_tables(existing, table)
                    self._table_by_id[table.document_id] = merged
                    # Replace in list.
                    self.tables = [merged if t.document_id == table.document_id else t for t in self.tables]
                    return
                if table.document_id:
                    self._table_by_id[table.document_id] = table
                self.tables.append(table)
            return

        # FeeTableSection and FeeTableRow are handled via recursion; rows are built
        # when their parent FeeTable is encountered.

    def _extract_table_caption(self, data: dict[str, Any]) -> str | None:
        content = data.get("content") or data.get("fields") or {}
        if not isinstance(content, dict):
            return None
        for key in ("caption", "title", "heading", "displayName"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                rendered = render_rich_text_node(value)
                if rendered.text.strip():
                    return rendered.text.strip()
        return None

    def _build_table(
        self,
        data: dict[str, Any],
        document_id: str | Any,
        component_id: str | Any,
        caption: str | None,
    ) -> Table | None:
        content = data.get("content") or data.get("fields") or {}
        if not isinstance(content, dict):
            content = {}
        headers: list[TableHeader] = []
        rows: list[Row] = []
        column_count: int | None = None

        columns = content.get("columns") or content.get("header") or []
        if isinstance(columns, list):
            headers = [self._render_header_cell(col) for col in columns]
            column_count = len(columns)

        raw_rows = content.get("rows") or content.get("data") or []
        if isinstance(raw_rows, list):
            for row_data in raw_rows:
                if isinstance(row_data, dict):
                    row = self._build_row(row_data)
                    if row.cells:
                        rows.append(row)

        # If no explicit rows, look for nested FeeTableRow children.
        if not rows:
            children = self._get_children(data)
            for child in children:
                if isinstance(child, dict) and child.get("componentType") == "FeeTableRow":
                    row = self._build_row(child)
                    if row.cells:
                        rows.append(row)

        # Determine column count from data if not already known.
        if column_count is None and rows:
            column_count = max(len(row.cells) for row in rows) if rows else None

        return Table(
            component_type="FeeTable",
            document_id=str(document_id) if document_id else None,
            component_id=component_id,
            caption=caption,
            section_path=list(self._path),
            column_count=column_count,
            headers=headers,
            rows=rows,
        )

    def _render_header_cell(self, data: Any) -> TableHeader:
        if isinstance(data, dict):
            rendered = render_rich_text_node(data)
            return TableHeader(
                text=rendered.text,
                tokens=rendered.tokens,
                links=rendered.links,
            )
        return TableHeader(text=str(data) if data is not None else "")

    def _build_row(self, data: dict[str, Any]) -> Row:
        cells: list[Cell] = []
        raw_cells = data.get("cells") or data.get("content") or data.get("fields") or []
        if isinstance(raw_cells, dict):
            raw_cells = list(raw_cells.values())
        if not isinstance(raw_cells, list):
            raw_cells = []
        for cell_data in raw_cells:
            if cell_data is None:
                cells.append(Cell(text=""))
                continue
            if isinstance(cell_data, dict):
                rendered = render_rich_text_node(cell_data)
                cells.append(rendered)
            else:
                cells.append(Cell(text=str(cell_data)))
        return Row(cells=cells)

    def _merge_tables(self, first: Table, second: Table) -> Table:
        """Merge two tables with the same document ID/caption, preserving order."""
        all_rows = list(first.rows) + list(second.rows)
        all_headers = list(first.headers) or list(second.headers)
        column_count = first.column_count or second.column_count
        if all_rows and column_count is None:
            column_count = max(len(row.cells) for row in all_rows)
        return Table(
            component_type=first.component_type or second.component_type,
            document_id=first.document_id,
            component_id=first.component_id or second.component_id,
            caption=first.caption or second.caption,
            section_path=first.section_path or second.section_path,
            column_count=column_count,
            headers=all_headers,
            rows=all_rows,
            source_table_ids=list(first.source_table_ids) + list(second.source_table_ids),
        )

    def _resolve_references(self) -> None:
        """Resolve FeeTableReference placeholders to their target tables."""
        reference_tables: list[Table] = []
        content_tables: list[Table] = []
        for table in self.tables:
            if table.component_type == "FeeTableReference" or (
                table.source_table_ids and not table.rows and not table.headers
            ):
                reference_tables.append(table)
            else:
                content_tables.append(table)

        for ref in reference_tables:
            target_id = ref.source_table_ids[0]
            target = self._table_by_id.get(target_id)
            if target is None:
                self.warnings.append(
                    ParserWarning(
                        code="unresolved_table_reference",
                        message=f"FeeTableReference {target_id} could not be resolved",
                        context={"document_id": target_id, "component_id": ref.component_id},
                    )
                )
                content_tables.append(ref)
                continue
            # Merge the reference's section path/context into the existing target.
            merged = self._merge_tables(
                target,
                Table(
                    component_type="FeeTableReference",
                    document_id=target.document_id,
                    component_id=ref.component_id,
                    caption=ref.caption,
                    section_path=ref.section_path,
                    column_count=target.column_count,
                    headers=target.headers,
                    rows=target.rows,
                    source_table_ids=[target_id],
                ),
            )
            # Replace the target in the content list and the index.
            self._table_by_id[target_id] = merged
            content_tables = [
                merged if t.document_id == target_id and t.component_type != "FeeTableReference" else t
                for t in content_tables
            ]

        self.tables = content_tables

    def has_any_table(self) -> bool:
        return bool(self.tables)

    def has_any_row(self) -> bool:
        return any(table.rows for table in self.tables)
