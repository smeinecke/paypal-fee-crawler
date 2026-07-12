"""Tests for CMS component extraction and table context preservation."""

from __future__ import annotations

from paypal_fee_crawler.components import ComponentsExtractor


def _simple_table(component_id: str, document_id: str, value: str) -> dict:
    return {
        "componentType": "FeeTable",
        "componentId": component_id,
        "documentId": document_id,
        "caption": "Commercial fees",
        "content": {
            "rows": [
                {"column1Text": "Commercial transactions", "column2Text": value},
            ]
        },
    }


def _reference(component_id: str, document_id: str) -> dict:
    return {
        "componentType": "FeeTableReference",
        "componentId": component_id,
        "documentId": document_id,
        "caption": "Also applies",
    }


def test_components_table_records_preserves_reference_context() -> None:
    cms = {
        "components": [
            _simple_table("c-1", "FEETB001", "2.99% + 0.39 EUR"),
            _reference("c-2", "FEETB001"),
        ]
    }
    extractor = ComponentsExtractor()
    sections, tables, warnings = extractor.extract(cms)

    assert len(tables) == 1
    assert len(extractor.table_records) == 1
    record = extractor.table_records[0]
    assert record.table.document_id == "FEETB001"
    assert len(record.contexts) == 2
    context_ids = {c.component_id for c in record.contexts}
    assert context_ids == {"c-1", "c-2"}


def test_components_table_records_duplicate_id_merges_contexts() -> None:
    cms = {
        "components": [
            _simple_table("c-1", "FEETB001", "2.99% + 0.39 EUR"),
            _simple_table("c-3", "FEETB001", "2.99% + 0.49 EUR"),
        ]
    }
    extractor = ComponentsExtractor()
    sections, tables, warnings = extractor.extract(cms)

    assert len(tables) == 1
    assert len(extractor.table_records) == 1
    record = extractor.table_records[0]
    assert len(record.contexts) == 2
    assert {c.component_id for c in record.contexts} == {"c-1", "c-3"}
    assert len(record.table.rows) == 2


def test_components_table_records_dedupes_identical_contexts() -> None:
    cms = {
        "components": [
            _simple_table("c-1", "FEETB001", "2.99% + 0.39 EUR"),
            _simple_table("c-1", "FEETB001", "2.99% + 0.39 EUR"),
        ]
    }
    extractor = ComponentsExtractor()
    extractor.extract(cms)

    record = extractor.table_records[0]
    assert len(record.contexts) == 1
    assert record.contexts[0].component_id == "c-1"


def test_components_table_records_section_and_parent_path() -> None:
    cms = {
        "components": [
            {
                "componentType": "TextSectionType",
                "componentId": "s-1",
                "heading": "Fees",
            },
            _simple_table("c-1", "FEETB001", "2.99%"),
        ]
    }
    extractor = ComponentsExtractor()
    extractor.extract(cms)

    record = extractor.table_records[0]
    assert record.contexts[0].section_path == ("Commercial fees",)
    assert record.contexts[0].parent_path == ()
