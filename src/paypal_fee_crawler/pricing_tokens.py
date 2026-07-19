"""Rich-text rendering and pricing-token normalization."""

from __future__ import annotations

import re
from decimal import InvalidOperation
from typing import Any

from .models import Cell, FeeToken, Link
from .normalize import CURRENCY_CODES, _normalize_decimal, _to_canonical_string

# Regex for numeric tokens with optional sign and decimal comma/point.
_NUMBER_RE = re.compile(r"^(?P<operator>[+\-])?(?P<value>[0-9]+(?:[.,][0-9]+)?)(?P<suffix>%|\s*%)?$")
_MONEY_RE = re.compile(r"^(?P<operator>[+\-])?(?P<amount>[0-9]+(?:[.,][0-9]+)?)\s*(?P<currency>[A-Z]{3})$")
_RANGE_RE = re.compile(r"^(?P<from>[0-9]+(?:[.,][0-9]+)?)\s*[-\u2013\u2014]\s*(?P<to>[0-9]+(?:[.,][0-9]+)?)$")


def _operator_name(operator: str | None) -> str | None:
    """Map an operator character to its canonical name."""
    if operator == "+":
        return "add"
    if operator == "-":
        return "subtract"
    return None


def _looks_like_percentage(text: str) -> bool:
    return "%" in text


def _looks_like_money(text: str) -> bool:
    parts = text.split()
    return any(part.upper() in CURRENCY_CODES for part in parts[-3:])


def _parse_number_token(text: str) -> FeeToken | None:
    text = text.strip()
    if not text:
        return None

    # Range token.
    range_match = _RANGE_RE.match(text)
    if range_match:
        try:
            from_val = _normalize_decimal(range_match.group("from"))
            to_val = _normalize_decimal(range_match.group("to"))
            return FeeToken(
                raw=text,
                kind="range",
                value=f"{_to_canonical_string(from_val)}-{_to_canonical_string(to_val)}",
            )
        except (InvalidOperation, ValueError):
            pass

    # Percentage/number token.
    number_match = _NUMBER_RE.match(text)
    if number_match:
        value_str = number_match.group("value")
        operator = number_match.group("operator")
        suffix = number_match.group("suffix")
        try:
            value = _normalize_decimal(value_str)
        except (InvalidOperation, ValueError):
            return None
        kind = "percentage" if suffix or _looks_like_percentage(text) else "number"
        return FeeToken(
            raw=text,
            kind=kind,
            value=_to_canonical_string(value),
            operator=_operator_name(operator),
        )

    # Money token.
    money_match = _MONEY_RE.match(text)
    if money_match:
        currency = money_match.group("currency").upper()
        if currency in CURRENCY_CODES:
            try:
                amount = _normalize_decimal(money_match.group("amount"))
            except (InvalidOperation, ValueError):
                return None
            return FeeToken(
                raw=text,
                kind="money",
                amount=_to_canonical_string(amount),
                currency=currency,
                operator=_operator_name(money_match.group("operator")),
            )

    return None


def normalize_pricing_token(
    raw_text: str,
    token_id: str | None = None,
    internal_name: str | None = None,
    fee_data_key: str | None = None,
    content_type: str | None = None,
) -> FeeToken:
    """Normalize a single pricing token value."""
    text = raw_text.strip()
    if not text:
        return FeeToken(raw=raw_text, kind="text")

    token = _parse_number_token(text)
    if token is None:
        # If the raw text contains a currency code at the end or start, try again with a stricter split.
        parts = text.split()
        if len(parts) >= 2:
            if parts[-1].upper() in CURRENCY_CODES:
                token = _parse_number_token(f"{parts[0]} {parts[-1].upper()}")
            elif parts[0].upper() in CURRENCY_CODES:
                token = _parse_number_token(f"{parts[-1]} {parts[0].upper()}")
    if token is None:
        token = FeeToken(raw=raw_text, kind="text")

    return FeeToken(
        raw=token.raw,
        kind=token.kind,
        value=token.value,
        amount=token.amount,
        currency=token.currency,
        operator=token.operator,
        token_id=token_id,
        internal_name=internal_name,
        fee_data_key=fee_data_key,
        content_type=content_type,
    )


_CONTAINER_TYPES = {
    "document",
    "Document",
    "rich-text",
    "paragraph",
    "Paragraph",
    "text",
    "Text",
    "hyperlink",
    "link",
    "list",
    "List",
    "list-item",
    "ListItem",
}

_EMBEDDED_TYPES = {
    "EmbeddedEntryBlock",
    "embedded-entry",
    "embedded-entry-block",
    "embedded-entry-inline",
}

_TOKENIZE_RE = re.compile(
    r"(?:(?P<currency_pre>[A-Z]{3})\s+)?(?P<operator>[+\-])?\s*(?P<value>[0-9]+(?:[.,][0-9]+)?)\s*(?P<currency_post>[A-Z]{3})?\s*(?P<suffix>%)?"
)


def _is_percentage_points_context(window: str) -> bool:
    """Return True when the surrounding text spells out percentage points."""
    lowered = window.lower()
    return bool(re.search(r"prozentpunkt(?:en?)?|percentage\s+points?", lowered))


def tokenize_text(text: str) -> list[FeeToken]:
    """Extract pricing tokens from a longer text string."""
    tokens: list[FeeToken] = []
    seen: set[str] = set()
    for match in _TOKENIZE_RE.finditer(text):
        raw = match.group(0).strip()
        if not raw or raw in seen:
            continue
        seen.add(raw)
        # Avoid matching isolated numbers without context.
        if (
            not match.group("suffix")
            and not match.group("currency_pre")
            and not match.group("currency_post")
            and not match.group("operator")
        ):
            # Still accept if the surrounding text contains a % or currency nearby.
            start, end = match.start(), match.end()
            window = text[max(0, start - 5) : min(len(text), end + 30)]
            if (
                "%" not in window
                and not any(part.upper() in CURRENCY_CODES for part in window.split())
                and not _is_percentage_points_context(window)
            ):
                continue
        # Build a clean token string so operators sit next to values for normalize_pricing_token.
        operator = match.group("operator")
        value = match.group("value")
        currency_pre = match.group("currency_pre")
        currency_post = match.group("currency_post")
        suffix = match.group("suffix")
        parts: list[str | None] = [operator, value] if operator else [value]
        if currency_pre:
            parts.insert(0, currency_pre)
        if currency_post:
            parts.append(currency_post)
        if suffix:
            parts.append(suffix)
        clean_raw = "".join(str(p) for p in parts if p is not None)

        # If the text spells out "percentage points", treat it as a percentage token.
        if not suffix:
            start, end = match.start(), match.end()
            window = text[max(0, start - 5) : min(len(text), end + 30)]
            if _is_percentage_points_context(window):
                clean_raw = clean_raw + "%"

        token = normalize_pricing_token(clean_raw)
        if token.kind != "text":
            tokens.append(token)
    return tokens


def _render_text_node(node: dict[str, Any]) -> tuple[str, list[FeeToken], list[Link]]:
    """Render a single text node and extract inline tokens/links."""
    text = ""
    tokens: list[FeeToken] = []
    links: list[Link] = []
    if not isinstance(node, dict):
        return text, tokens, links

    raw = node.get("value", "")
    if not isinstance(raw, str):
        raw = str(raw) if raw is not None else ""
    text = raw

    # Extract pricing tokens from the text itself.
    tokens.extend(tokenize_text(text))

    # Marks (bold, etc.) are not represented in text; they affect only styling.
    # Hyperlinks are captured.
    data = node.get("data") or {}
    uri = data.get("uri") or data.get("href")
    if uri and isinstance(uri, str):
        links.append(Link(text=text, uri=uri))

    # Embedded pricing token reference (block or inline).
    node_type = node.get("nodeType") or node.get("type") or ""
    if node_type in {"EmbeddedEntryBlock", "embedded-entry", "embedded-entry-block", "embedded-entry-inline"}:
        target = node.get("data", {}).get("target", {})
        fields = target.get("fields", {})
        token_value = fields.get("feeDataKey") or fields.get("value") or fields.get("displayValue")
        if token_value:
            token = normalize_pricing_token(
                str(token_value),
                token_id=target.get("sys", {}).get("id"),
                internal_name=fields.get("internalName"),
                fee_data_key=fields.get("feeDataKey"),
                content_type=target.get("sys", {}).get("contentType", {}).get("sys", {}).get("id"),
            )
            tokens.append(token)
            text = token.raw
        elif node_type in {"embedded-entry-inline", "EmbeddedEntryInline"}:
            # Preserve the node type so callers know an inline token was present.
            text = ""

    return text, tokens, links


_CHILD_CATEGORY = {
    "text": "text",
    "Text": "text",
    "hyperlink": "link",
    "link": "link",
    "Hyperlink": "link",
    "paragraph": "paragraph",
    "Paragraph": "paragraph",
    "list-item": "paragraph",
    "ListItem": "paragraph",
    "line-break": "linebreak",
    "linebreak": "linebreak",
    "LineBreak": "linebreak",
    "embedded-entry-block": "embedded",
    "embedded-entry": "embedded",
    "embedded-entry-inline": "embedded",
}


def _render_single_value_node(node: dict[str, Any]) -> Cell:
    value = node.get("value") or node.get("text") or node.get("displayValue") or ""
    if isinstance(value, dict):
        return render_rich_text_node(value)
    return Cell(text=str(value).strip())


def _render_child(child: Any) -> tuple[str, list[FeeToken], list[Link]]:
    """Render a single child node and return its text, tokens and links."""
    if not isinstance(child, dict):
        return str(child), [], []

    child_type = child.get("nodeType") or child.get("type") or ""
    category = _CHILD_CATEGORY.get(child_type, "default")

    if category == "linebreak":
        return "\n", [], []

    if category == "text":
        return _render_text_node(child)

    rendered = render_rich_text_node(child)
    text = rendered.text
    if not text.strip():
        return "", [], []

    if category == "link":
        links = rendered.links or []
        if not links:
            uri = child.get("data", {}).get("uri") or child.get("data", {}).get("href")
            if uri:
                links = [Link(text=text, uri=uri)]
        return text, rendered.tokens, links

    return rendered.text, rendered.tokens, rendered.links


def _render_children(content: list[Any]) -> tuple[list[str], list[FeeToken], list[Link]]:
    text_parts: list[str] = []
    tokens: list[FeeToken] = []
    links: list[Link] = []
    for child in content:
        child_text, child_tokens, child_links = _render_child(child)
        text_parts.append(child_text)
        tokens.extend(child_tokens)
        links.extend(child_links)
    return text_parts, tokens, links


def render_rich_text_node(node: Any) -> Cell:
    """Render a Contentful-like rich-text node into a lossless Cell."""
    if node is None:
        return Cell(text="")
    if isinstance(node, str):
        return Cell(text=node, tokens=tokenize_text(node))
    if not isinstance(node, dict):
        return Cell(text=str(node))

    node_type = node.get("nodeType") or node.get("type") or ""
    if node_type in _EMBEDDED_TYPES:
        text, tokens, links = _render_text_node(node)
        return Cell(text=text, tokens=tokens, links=links)

    if node_type in _CONTAINER_TYPES or "content" in node:
        content = node.get("content") or []
    else:
        return _render_single_value_node(node)

    text_parts, tokens, links = _render_children(content)
    full_text = "".join(text_parts)
    full_text = full_text.replace("\u00a0", " ").replace("\u202f", " ")
    return Cell(text=full_text.strip(), tokens=tokens, links=links)
