"""Extract the strict JSON CMS render context from PayPal HTML pages."""

from __future__ import annotations

import json
import logging
from typing import Any

from lxml import html

from .exceptions import ParserError

logger = logging.getLogger(__name__)

TARGET_ASSIGNMENT = "window.__CMS_ENGINE_RENDER_CONTEXT__"


def extract_cms_context(html_text: str) -> dict[str, Any]:
    """Parse the HTML and return exactly one CMS render context object.

    Raises:
        ParserError: if zero or multiple contexts are found or if the JSON is invalid.
    """
    try:
        tree = html.fromstring(html_text)
    except Exception as exc:
        raise ParserError(f"Failed to parse HTML: {exc}") from exc

    scripts = tree.xpath("//script")
    contexts: list[dict[str, Any]] = []
    for script in scripts:
        text = script.text or ""
        if not text or TARGET_ASSIGNMENT not in text:
            continue
        # Find the assignment and strip the wrapper, keeping only the JSON object.
        idx = text.index(TARGET_ASSIGNMENT)
        after = text[idx + len(TARGET_ASSIGNMENT) :]
        # Remove optional whitespace and the '=' sign.
        after = after.lstrip()
        if after.startswith("="):
            after = after[1:].lstrip()
        # Remove trailing semicolon if present.
        after = after.rstrip()
        if after.endswith(";"):
            after = after[:-1].rstrip()
        try:
            data = json.loads(after)
        except json.JSONDecodeError as exc:
            raise ParserError(f"Invalid JSON in CMS render context: {exc}") from exc
        if not isinstance(data, dict):
            raise ParserError("CMS render context is not a JSON object")
        contexts.append(data)

    if len(contexts) == 0:
        raise ParserError("No CMS render context found in page")
    if len(contexts) > 1:
        raise ParserError("Multiple CMS render contexts found in page")
    return contexts[0]


def find_global_json_assignments(html_text: str, variable_names: list[str]) -> dict[str, Any]:
    """Find strict JSON assignments for a list of global variable names.

    This is used as a secondary navigation/country-discovery helper. It is not a
    general JavaScript interpreter and only parses strict JSON payloads.
    """
    results: dict[str, Any] = {}
    try:
        tree = html.fromstring(html_text)
    except Exception:
        return results

    scripts = tree.xpath("//script")
    for script in scripts:
        text = script.text or ""
        for name in variable_names:
            if name not in text:
                continue
            try:
                idx = text.index(name)
            except ValueError:
                continue
            after = text[idx + len(name) :].lstrip()
            if after.startswith("="):
                after = after[1:].lstrip()
            after = after.rstrip()
            if after.endswith(";"):
                after = after[:-1].rstrip()
            try:
                data = json.loads(after)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                results[name] = data
    return results
