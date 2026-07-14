"""Tests for component traversal and fee table extraction."""

from __future__ import annotations

from paypal_fee_crawler.cms_context import extract_cms_context
from paypal_fee_crawler.components import ComponentsExtractor


def test_extract_tables_from_de_fixture(de_real_html: str) -> None:
    cms = extract_cms_context(de_real_html)
    extractor = ComponentsExtractor()
    sections, tables, warnings = extractor.extract(cms)
    assert tables
    assert any(t.component_type == "FeeTable" for t in tables)
    assert sum(len(t.rows) for t in tables) > 0


def test_fee_table_reference_resolved(de_real_html: str) -> None:
    cms = extract_cms_context(de_real_html)
    extractor = ComponentsExtractor()
    _, tables, _ = extractor.extract(cms)
    ids = [t.document_id for t in tables if t.document_id]
    # Real DE fixture references this fixed-fee table in multiple places.
    assert any(id and id.upper() == "FEETB18" for id in ids)
    referenced_tables = [t for t in tables if t.document_id and t.document_id.upper() == "FEETB18"]
    assert len(referenced_tables) == 1
    assert len(referenced_tables[0].rows) >= 4


def test_split_tables_preserved(de_real_html: str) -> None:
    cms = extract_cms_context(de_real_html)
    extractor = ComponentsExtractor()
    _, tables, _ = extractor.extract(cms)
    captions = [t.caption for t in tables]
    # Real DE fixture splits the commercial fixed-fee table across two components
    # with the same caption. They are now merged into a single logical table.
    assert (
        captions.count(
            "Gebührentabelle: Festgebühr bei geschäftlichen Transaktionen (auf Basis der empfangenen Währung)"
        )
        == 1
    )
    merged = [
        t
        for t in tables
        if t.caption
        == "Gebührentabelle: Festgebühr bei geschäftlichen Transaktionen (auf Basis der empfangenen Währung)"
    ]
    assert merged
    # The merged table should contain all rows from both fragments.
    assert len(merged[0].rows) >= 24


def test_missing_fee_table_raises() -> None:
    html = '<script>window.__CMS_ENGINE_RENDER_CONTEXT__ = {"pageReference": {"pageModel": {"middle": []}}};</script>'
    cms = extract_cms_context(html)
    extractor = ComponentsExtractor()
    _, tables, _ = extractor.extract(cms)
    assert not tables
    assert not extractor.has_any_table()


def test_duplicate_document_id_warning() -> None:
    cms = {
        "pageReference": {
            "pageModel": {
                "middle": [
                    {
                        "componentType": "FeeTable",
                        "documentId": "DUP",
                        "content": {"rows": [{"cells": ["a"]}]},
                    },
                    {
                        "componentType": "FeeTable",
                        "documentId": "DUP",
                        "content": {"rows": [{"cells": ["b"]}]},
                    },
                ]
            }
        }
    }
    extractor = ComponentsExtractor()
    _, tables, warnings = extractor.extract(cms)
    assert len(tables) == 1
    assert len(tables[0].rows) == 2
