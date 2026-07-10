"""Extract the strict JSON CMS render context from PayPal HTML pages."""

from __future__ import annotations

import json
import logging
from typing import Any

from lxml import html

from .exceptions import ParserError

logger = logging.getLogger(__name__)

TARGET_ASSIGNMENT = "window.__CMS_ENGINE_RENDER_CONTEXT__"

# Global contexts that are allowlisted for structured discovery. Do not add
# arbitrary JavaScript variables here; only parse strict JSON from explicitly
# named PayPal server-rendered objects.
ALLOWLISTED_GLOBAL_CONTEXTS = {
    "window.__CMS_ENGINE_RENDER_CONTEXT__",
    "window.__GLOBAL_NAV_CONTEXT_FOOTER__",
    "window.__GLOBAL_NAV_CONTEXT__",
    "window.__GLOBAL_NAV_CONTEXT_HEADER__",
}


def _strip_assignment(text: str, name: str) -> str:
    """Return the strict JSON payload after a global assignment, if present."""
    idx = text.index(name)
    after = text[idx + len(name) :].lstrip()
    if after.startswith("="):
        after = after[1:].lstrip()
    after = after.rstrip()
    if after.endswith(";"):
        after = after[:-1].rstrip()
    return after


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
        try:
            after = _strip_assignment(text, TARGET_ASSIGNMENT)
        except ValueError:
            continue
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
                after = _strip_assignment(text, name)
            except ValueError:
                continue
            try:
                data = json.loads(after)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                results[name] = data
    return results


def extract_all_contexts(html_text: str) -> dict[str, Any]:
    """Return all successfully parsed, allowlisted global contexts.

    The CMS render context is required; other contexts are optional. Malformed
    optional contexts are reported as warnings, not fatal errors.
    """
    results: dict[str, Any] = {}
    warnings: list[str] = []
    try:
        tree = html.fromstring(html_text)
    except Exception as exc:
        raise ParserError(f"Failed to parse HTML: {exc}") from exc

    scripts = tree.xpath("//script")
    for script in scripts:
        text = script.text or ""
        for name in ALLOWLISTED_GLOBAL_CONTEXTS:
            if name not in text or name in results:
                continue
            try:
                after = _strip_assignment(text, name)
            except ValueError:
                warnings.append(f"Could not strip assignment wrapper for {name}")
                continue
            try:
                data = json.loads(after)
            except json.JSONDecodeError as exc:
                if name == TARGET_ASSIGNMENT:
                    raise ParserError(f"Invalid JSON in CMS render context: {exc}") from exc
                warnings.append(f"Malformed JSON in {name}: {exc}")
                continue
            if not isinstance(data, dict):
                if name == TARGET_ASSIGNMENT:
                    raise ParserError("CMS render context is not a JSON object")
                warnings.append(f"{name} is not a JSON object")
                continue
            results[name] = data

    if TARGET_ASSIGNMENT not in results:
        raise ParserError("No CMS render context found in page")

    return {"contexts": results, "warnings": warnings}
