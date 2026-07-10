"""Tests for component traversal and fee table extraction."""

from __future__ import annotations

from paypal_fee_crawler.cms_context import extract_cms_context
from paypal_fee_crawler.components import ComponentsExtractor


def test_extract_tables_from_de_fixture(de_html: str) -> None:
    cms = extract_cms_context(de_html)
    extractor = ComponentsExtractor()
    sections, tables, warnings = extractor.extract(cms)
    assert tables
    assert any(t.component_type == "FeeTable" for t in tables)
    assert sum(len(t.rows) for t in tables) > 0


def test_fee_table_reference_resolved(de_html: str) -> None:
    cms = extract_cms_context(de_html)
    extractor = ComponentsExtractor()
    _, tables, _ = extractor.extract(cms)
    ids = [t.document_id for t in tables if t.document_id]
    assert any(id and id.upper() == "FEETB-DE-05" for id in ids)
    split_tables = [t for t in tables if t.document_id and t.document_id.upper() == "FEETB-DE-05"]
    assert len(split_tables) == 1
    assert len(split_tables[0].rows) >= 4


def test_split_tables_preserved(de_html: str) -> None:
    cms = extract_cms_context(de_html)
    extractor = ComponentsExtractor()
    _, tables, _ = extractor.extract(cms)
    captions = [t.caption for t in tables]
    assert captions.count("Currency fixed fees") == 1


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
