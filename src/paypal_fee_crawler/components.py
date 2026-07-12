"""Recursive traversal and normalization of PayPal CMS components."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from typing import Any

from .models import Cell, ParserWarning, Row, Section, Table, TableHeader
from .pricing_tokens import render_rich_text_node
from .profiles import NormalizedTableRecord, TableContext

logger = logging.getLogger(__name__)

# Component types that carry page structure or table data.
_TABLE_COMPONENT_TYPES = {"FeeTable", "FeeTableReference", "FeeTableRow", "FeeTableSection", "FeeTableSplitRow"}
_SECTION_COMPONENT_TYPES = {
    "TextSectionType",
    "TextGroup",
    "TextHeaderInner",
    "FeatureNavigationSection",
    "PopoverModal",
    "PopoverModalItem",
    "SubNav",
    "Button",
    "Content",
}

# Container keys that commonly hold nested component collections.
_CHILD_CONTAINER_KEYS = {
    "middle",
    "top",
    "bottom",
    "content",
    "fields",
    "components",
    "items",
    "children",
    "rows",
    "columns",
    "tables",
    "collection",
    "componentReference",
    "textGroup",
    "paragraph",
    "subheading",
    "ctaCollection",
    "pageReference",
    "pageModel",
    "pageContext",
    "environment",
}

_NUMBERED_COLUMN_RE = re.compile(r"^column(?P<index>\d+)(?P<kind>Header|Text)$")


def iter_components(value: object, component_type: str | None = None) -> Iterator[dict[str, object]]:
    """Recursively yield every component-like dict in *value*.

    If *component_type* is given, only yield dicts whose ``componentType`` or
    ``type`` matches. The iterator safely walks dicts and lists and avoids
    recursing into strings or cycling on malformed objects.
    """
    seen: set[int] = set()

    def _walk(obj: object) -> Iterator[dict[str, object]]:
        obj_id = id(obj)
        if obj_id in seen:
            return
        if isinstance(obj, dict):
            seen.add(obj_id)
            ct = obj.get("componentType") or obj.get("type")
            if component_type is None or ct == component_type:
                yield obj
            for value in obj.values():
                yield from _walk(value)
        elif isinstance(obj, list):
            seen.add(obj_id)
            for item in obj:
                yield from _walk(item)

    yield from _walk(value)


class ComponentsExtractor:
    """Extract normalized sections and tables from a PayPal CMS context."""

    def __init__(self) -> None:
        self.tables: list[Table] = []
        self.table_records: list[NormalizedTableRecord] = []
        self.sections: list[Section] = []
        self.warnings: list[ParserWarning] = []
        self._table_by_id: dict[str, Table] = {}
        self._table_records_by_id: dict[str, NormalizedTableRecord] = {}
        self._path: list[str] = []
        self._section_path: list[str] = []
        self._source_order: int = 0
        self._seen_component_ids: set[str] = set()

    def extract(self, cms: dict[str, Any]) -> tuple[list[Section], list[Table], list[ParserWarning]]:
        """Return (sections, tables, warnings)."""
        self.tables = []
        self.table_records = []
        self.sections = []
        self.warnings = []
        self._table_by_id = {}
        self._table_records_by_id = {}
        self._path = []
        self._section_path = []
        self._source_order = 0
        self._seen_component_ids = set()

        self._traverse(cms)
        self._resolve_references()
        self._assign_table_ids()
        self.tables = [record.table for record in self.table_records]
        return self.sections, self.tables, self.warnings

    def _traverse(self, data: Any) -> None:
        if not isinstance(data, dict):
            return
        component_id = data.get("componentId") or data.get("id")
        if isinstance(component_id, str) and component_id:
            if component_id in self._seen_component_ids:
                return
            self._seen_component_ids.add(component_id)
        component_type = data.get("componentType") or data.get("type")

        heading = self._extract_heading(data)
        is_section_like = component_type in _SECTION_COMPONENT_TYPES or component_type in _TABLE_COMPONENT_TYPES
        if heading and is_section_like:
            self._section_path.append(heading)

        if component_type in _TABLE_COMPONENT_TYPES:
            self._handle_table_component(data, component_type)
        elif component_type in _SECTION_COMPONENT_TYPES:
            self._handle_generic_section(data, component_type)

        children = self._get_children(data)
        for child in children:
            self._traverse(child)

        if heading and is_section_like:
            self._section_path.pop()

    def _get_children(self, data: dict[str, Any]) -> list[Any]:
        """Return child components that should be traversed recursively."""
        children: list[Any] = []
        for key in _CHILD_CONTAINER_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                children.extend(value)
            elif isinstance(value, dict):
                children.append(value)
        # Some components wrap others under arbitrary keys; add dict/list values
        # that look like component collections, but skip plain rich-text nodes.
        for key, value in data.items():
            if key in _CHILD_CONTAINER_KEYS or key in ("componentType", "type", "id", "componentId", "documentId"):
                continue
            if isinstance(value, list):
                children.extend(value)
            elif isinstance(value, dict) and (value.get("componentType") or value.get("type")):
                children.append(value)
        return children

    def _extract_heading(self, data: dict[str, Any]) -> str | None:
        """Extract a human-readable heading from a component."""
        for key in ("heading", "title", "displayName", "name", "subheading", "caption"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                rendered = render_rich_text_node(value)
                if rendered.text.strip():
                    return rendered.text.strip()
        content = data.get("content") or data.get("fields") or {}
        if isinstance(content, dict):
            for key in ("heading", "title", "displayName", "name", "subheading", "caption"):
                value = content.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, dict):
                    rendered = render_rich_text_node(value)
                    if rendered.text.strip():
                        return rendered.text.strip()
            # Rich-text headings may be nested under textGroup.
            tg = content.get("textGroup") or content.get("paragraph")
            if isinstance(tg, dict):
                rendered = render_rich_text_node(tg)
                if rendered.text.strip():
                    return rendered.text.strip()
        return None

    def _handle_generic_section(self, data: dict[str, Any], component_type: str | Any) -> None:
        component_id = data.get("componentId") or data.get("id")
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
            # Fallback: render the whole content dict if it looks like rich text.
            if not body and (content.get("nodeType") or content.get("type")):
                rendered = render_rich_text_node(content)
                body = rendered.text.strip()
        self.sections.append(
            Section(
                component_id=component_id,
                component_type=str(component_type),
                heading=heading,
                body=body or None,
                section_path=list(self._section_path),
            )
        )

    def _handle_table_component(self, data: dict[str, Any], component_type: str | Any) -> None:
        document_id = data.get("documentId") or data.get("feeTableDocumentId")
        component_id = data.get("componentId") or data.get("id")
        caption = self._extract_table_caption(data)

        if component_type == "FeeTableReference" and document_id:
            self._source_order += 1
            table = Table(
                component_type="FeeTableReference",
                document_id=str(document_id),
                component_id=component_id,
                caption=caption,
                section_path=list(self._section_path),
                parent_path=list(self._path),
                source_order=self._source_order,
                source_table_ids=[str(document_id)],
                reference_id=str(document_id),
            )
            context = TableContext.from_table(table)
            record = NormalizedTableRecord(table=table, contexts=(context,))
            self.tables.append(table)
            self.table_records.append(record)
            return

        if component_type == "FeeTable":
            self._source_order += 1
            table = self._build_table(data, document_id, component_id, caption)
            if table is not None:
                context = TableContext.from_table(table)
                record = NormalizedTableRecord(table=table, contexts=(context,))
                # Distinct components (split tables) keep their own document IDs
                # and are not merged by caption alone. Duplicate document IDs are
                # merged so a referenced table and its source stay consistent.
                if table.document_id and table.document_id in self._table_records_by_id:
                    existing = self._table_records_by_id[table.document_id]
                    merged_table = self._merge_tables(existing.table, table)
                    merged_record = self._merge_table_records(existing, record, merged_table)
                    self._table_by_id[table.document_id] = merged_table
                    self._table_records_by_id[table.document_id] = merged_record
                    self.tables = [
                        merged_table if t.document_id == table.document_id else t
                        for t in self.tables
                    ]
                    self.table_records = [
                        merged_record if r.table.document_id == table.document_id else r
                        for r in self.table_records
                    ]
                    return
                if table.document_id:
                    self._table_by_id[table.document_id] = table
                    self._table_records_by_id[table.document_id] = record
                self.tables.append(table)
                self.table_records.append(record)
            return

        # FeeTableSection, FeeTableRow, and FeeTableSplitRow are handled through
        # recursion; rows are materialized when their parent FeeTable is built.

    def _extract_table_caption(self, data: dict[str, Any]) -> str | None:
        for key in ("caption", "richTextCaption", "title", "heading", "displayName", "subheading"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                rendered = render_rich_text_node(value)
                if rendered.text.strip():
                    return rendered.text.strip()
        content = data.get("content") or data.get("fields") or {}
        if not isinstance(content, dict):
            return None
        for key in ("caption", "richTextCaption", "title", "heading", "displayName", "subheading"):
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

        headers, header_count = self._extract_numbered_headers(data)
        rows, row_count = self._extract_numbered_rows(data)
        declared_count = content.get("columns")
        if isinstance(declared_count, int):
            declared_column_count = declared_count
        elif isinstance(declared_count, list):
            declared_column_count = len(declared_count)
        else:
            declared_column_count = None

        if header_count is not None and row_count is not None and row_count > header_count:
            self.warnings.append(
                ParserWarning(
                    code="row_wider_than_header",
                    message=f"Table {document_id} has rows with {row_count} cells but only {header_count} headers",
                    context={"document_id": str(document_id), "component_id": component_id},
                )
            )

        column_count = header_count or row_count or declared_column_count

        return Table(
            component_type="FeeTable",
            document_id=str(document_id) if document_id else None,
            component_id=component_id,
            caption=caption,
            section_path=list(self._section_path),
            parent_path=list(self._path),
            source_order=self._source_order,
            column_count=column_count,
            declared_column_count=declared_column_count,
            headers=headers,
            rows=rows,
        )

    def _extract_numbered_headers(self, data: dict[str, Any]) -> tuple[list[TableHeader], int | None]:
        """Extract headers keyed as column<N>Header, ordered numerically."""
        headers: list[tuple[int, TableHeader]] = []
        for key, value in data.items():
            match = _NUMBERED_COLUMN_RE.match(key)
            if not match or match.group("kind") != "Header":
                continue
            idx = int(match.group("index")) - 1
            rendered = render_rich_text_node(value)
            headers.append((idx, TableHeader(text=rendered.text, tokens=rendered.tokens, links=rendered.links)))
        # Also check content/fields wrappers.
        content = data.get("content") or data.get("fields") or {}
        if isinstance(content, dict):
            for key, value in content.items():
                match = _NUMBERED_COLUMN_RE.match(key)
                if not match or match.group("kind") != "Header":
                    continue
                idx = int(match.group("index")) - 1
                rendered = render_rich_text_node(value)
                headers.append((idx, TableHeader(text=rendered.text, tokens=rendered.tokens, links=rendered.links)))
        if not headers:
            return [], None
        max_idx = max(idx for idx, _ in headers)
        ordered = [TableHeader(text="")] * (max_idx + 1)
        for idx, header in headers:
            ordered[idx] = header
        return ordered, len(ordered)

    def _extract_numbered_rows(self, data: dict[str, Any]) -> tuple[list[Row], int | None]:
        """Extract rows from content.rows or nested FeeTableRow/FeeTableSplitRow components."""
        rows: list[Row] = []
        max_cells = 0

        content = data.get("content") or data.get("fields") or {}
        if isinstance(content, dict):
            raw_rows = content.get("rows") or []
            if isinstance(raw_rows, list):
                for row_data in raw_rows:
                    if isinstance(row_data, dict):
                        row = self._build_row(row_data)
                        rows.append(row)
                        max_cells = max(max_cells, len(row.cells))

        # If no explicit rows, look for nested FeeTableRow children.
        if not rows:
            for child in self._get_children(data):
                if isinstance(child, dict) and child.get("componentType") in {
                    "FeeTableRow",
                    "FeeTableSplitRow",
                }:
                    row = self._build_row(child)
                    rows.append(row)
                    max_cells = max(max_cells, len(row.cells))

        return rows, max_cells if max_cells > 0 else None

    def _build_row(self, data: dict[str, Any]) -> Row:
        row_id = data.get("id") or data.get("componentId")
        cells, cell_count = self._extract_numbered_cells(data)
        return Row(row_id=str(row_id) if row_id else None, cells=cells)

    def _extract_numbered_cells(self, data: dict[str, Any]) -> tuple[list[Cell], int]:
        """Extract cells keyed as column<N>Text, ordered numerically, padding empties."""
        cells: list[tuple[int, Cell]] = []
        for key, value in data.items():
            match = _NUMBERED_COLUMN_RE.match(key)
            if not match or match.group("kind") != "Text":
                continue
            idx = int(match.group("index")) - 1
            rendered = render_rich_text_node(value)
            cells.append((idx, rendered))
        # Also check content/fields wrappers.
        content = data.get("content") or data.get("fields") or {}
        if isinstance(content, dict):
            for key, value in content.items():
                match = _NUMBERED_COLUMN_RE.match(key)
                if not match or match.group("kind") != "Text":
                    continue
                idx = int(match.group("index")) - 1
                rendered = render_rich_text_node(value)
                cells.append((idx, rendered))
        # Legacy cells array.
        raw_cells = data.get("cells") or []
        if isinstance(raw_cells, dict):
            raw_cells = list(raw_cells.values())
        if isinstance(raw_cells, list):
            for idx, cell_data in enumerate(raw_cells):
                if cell_data is None:
                    cells.append((idx, Cell(text="")))
                elif isinstance(cell_data, dict):
                    cells.append((idx, render_rich_text_node(cell_data)))
                else:
                    cells.append((idx, Cell(text=str(cell_data))))
        if not cells:
            return [], 0
        max_idx = max(idx for idx, _ in cells)
        ordered = [Cell(text="")] * (max_idx + 1)
        for idx, cell in cells:
            ordered[idx] = cell
        return ordered, len(ordered)

    def _assign_table_ids(self) -> None:
        """Assign a stable internal identity to every physical table."""
        updated: list[NormalizedTableRecord] = []
        for record in self.table_records:
            table = record.table
            parts = [
                table.component_type or "table",
                table.document_id or table.component_id or "",
                str(table.source_order),
            ]
            table_id = "::".join(p for p in parts if p)
            # Reassign via model_copy because the model is frozen.
            updated_table = table.model_copy(update={"table_id": table_id})
            updated.append(NormalizedTableRecord(table=updated_table, contexts=record.contexts))
        self.table_records = updated
        self.tables = [record.table for record in self.table_records]

    def _merge_tables(self, first: Table, second: Table) -> Table:
        """Merge two tables with the same document ID, preserving order and IDs."""
        all_rows = list(first.rows) + list(second.rows)
        all_headers = list(first.headers) or list(second.headers)
        column_count = first.column_count or second.column_count
        if all_rows and column_count is None:
            column_count = max(len(row.cells) for row in all_rows)
        return first.model_copy(
            update={
                "column_count": column_count,
                "headers": all_headers,
                "rows": all_rows,
                "source_table_ids": list(first.source_table_ids) + list(second.source_table_ids),
            }
        )

    def _merge_table_records(
        self,
        first: NormalizedTableRecord,
        second: NormalizedTableRecord,
        merged_table: Table,
    ) -> NormalizedTableRecord:
        """Merge two records for the same document ID, preserving contexts."""
        combined = list(first.contexts)
        for context in second.contexts:
            if context not in combined:
                combined.append(context)
        return NormalizedTableRecord(table=merged_table, contexts=tuple(combined))

    def _resolve_references(self) -> None:
        """Resolve FeeTableReference placeholders to their target tables."""
        # Index target records by document ID.
        target_records: dict[str, NormalizedTableRecord] = {}
        for record in self.table_records:
            if record.table.component_type == "FeeTable" and record.table.document_id:
                if record.table.document_id in target_records:
                    existing = target_records[record.table.document_id]
                    merged_table = self._merge_tables(existing.table, record.table)
                    target_records[record.table.document_id] = self._merge_table_records(
                        existing, record, merged_table
                    )
                else:
                    target_records[record.table.document_id] = record

        reference_records: list[NormalizedTableRecord] = []
        content_records: list[NormalizedTableRecord] = []
        for record in self.table_records:
            table = record.table
            if table.component_type == "FeeTableReference" or (
                table.source_table_ids and not table.rows and not table.headers
            ):
                reference_records.append(record)
            else:
                content_records.append(record)

        for ref_record in reference_records:
            ref = ref_record.table
            target_id = ref.source_table_ids[0]
            target_record = target_records.get(target_id)
            if target_record is None:
                self.warnings.append(
                    ParserWarning(
                        code="unresolved_table_reference",
                        message=f"FeeTableReference {target_id} could not be resolved",
                        context={"document_id": target_id, "component_id": ref.component_id},
                    )
                )
                content_records.append(ref_record)
                continue
            ref_table = Table(
                component_type="FeeTableReference",
                document_id=target_record.table.document_id,
                component_id=ref.component_id,
                caption=ref.caption,
                section_path=ref.section_path,
                parent_path=ref.parent_path,
                source_order=ref.source_order,
                column_count=target_record.table.column_count,
                declared_column_count=target_record.table.declared_column_count,
                headers=[],
                rows=[],
                source_table_ids=[target_id],
                reference_id=ref.reference_id,
            )
            merged_table = self._merge_tables(target_record.table, ref_table)
            merged_record = self._merge_table_records(
                target_record,
                NormalizedTableRecord(table=ref_table, contexts=ref_record.contexts),
                merged_table,
            )
            target_records[target_id] = merged_record
            content_records = [
                merged_record
                if r.table.document_id == target_record.table.document_id
                and r.table.component_id == target_record.table.component_id
                and r.table.component_type != "FeeTableReference"
                else r
                for r in content_records
            ]

        self.table_records = content_records
        self.tables = [record.table for record in content_records]
        self._table_by_id = {t.document_id: t for t in self.tables if t.document_id}
        self._table_records_by_id = {
            record.table.document_id: record
            for record in content_records
            if record.table.document_id
        }

    def has_any_table(self) -> bool:
        return bool(self.tables)

    def has_any_row(self) -> bool:
        return any(table.rows for table in self.tables)
