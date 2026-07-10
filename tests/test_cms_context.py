"""Tests for CMS context extraction."""

from __future__ import annotations

import json

import pytest

from paypal_fee_crawler.cms_context import extract_cms_context, find_global_json_assignments
from paypal_fee_crawler.exceptions import ParserError


def test_extract_cms_context_from_de_fixture(de_html: str) -> None:
    ctx = extract_cms_context(de_html)
    assert ctx["pageId"] == "business/paypal-business-fees"
    assert "countrySelector" in ctx


def test_extract_cms_context_invalid_json() -> None:
    html = "<script>window.__CMS_ENGINE_RENDER_CONTEXT__ = {not valid json};</script>"
    with pytest.raises(ParserError):
        extract_cms_context(html)


def test_extract_cms_context_missing() -> None:
    html = "<html><body><script>var x = 1;</script></body></html>"
    with pytest.raises(ParserError, match="No CMS render context"):
        extract_cms_context(html)


def test_extract_cms_context_duplicate() -> None:
    ctx = json.dumps({"pageId": "test"})
    html = f"<script>window.__CMS_ENGINE_RENDER_CONTEXT__ = {ctx};</script><script>window.__CMS_ENGINE_RENDER_CONTEXT__ = {ctx};</script>"
    with pytest.raises(ParserError, match="Multiple"):
        extract_cms_context(html)


def test_find_global_json_assignments() -> None:
    data = json.dumps({"key": "value"})
    html = f"<script>window.__CONFIG__ = {data};</script>"
    result = find_global_json_assignments(html, ["window.__CONFIG__"])
    assert result["window.__CONFIG__"]["key"] == "value"
