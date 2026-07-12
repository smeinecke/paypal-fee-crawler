"""Rich-text rendering and pricing-token normalization."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any

from .models import Cell, FeeToken, Link

# ISO 4217 currency codes (selected common set; not exhaustive).
CURRENCY_CODES = {
    "AED",
    "AFN",
    "ALL",
    "AMD",
    "ANG",
    "AOA",
    "ARS",
    "AUD",
    "AWG",
    "AZN",
    "BAM",
    "BBD",
    "BDT",
    "BGN",
    "BHD",
    "BIF",
    "BMD",
    "BND",
    "BOB",
    "BRL",
    "BSD",
    "BTN",
    "BWP",
    "BYN",
    "BZD",
    "CAD",
    "CDF",
    "CHF",
    "CLP",
    "CNY",
    "COP",
    "CRC",
    "CUP",
    "CVE",
    "CZK",
    "DJF",
    "DKK",
    "DOP",
    "DZD",
    "EGP",
    "ERN",
    "ETB",
    "EUR",
    "FJD",
    "FKP",
    "FOK",
    "GBP",
    "GEL",
    "GGP",
    "GHS",
    "GIP",
    "GMD",
    "GNF",
    "GTQ",
    "GYD",
    "HKD",
    "HNL",
    "HRK",
    "HTG",
    "HUF",
    "IDR",
    "ILS",
    "IMP",
    "INR",
    "IQD",
    "IRR",
    "ISK",
    "JEP",
    "JMD",
    "JOD",
    "JPY",
    "KES",
    "KGS",
    "KHR",
    "KID",
    "KMF",
    "KRW",
    "KWD",
    "KYD",
    "KZT",
    "LAK",
    "LBP",
    "LKR",
    "LRD",
    "LSL",
    "LYD",
    "MAD",
    "MDL",
    "MGA",
    "MKD",
    "MMK",
    "MNT",
    "MOP",
    "MRU",
    "MUR",
    "MVR",
    "MWK",
    "MXN",
    "MYR",
    "MZN",
    "NAD",
    "NGN",
    "NIO",
    "NOK",
    "NPR",
    "NZD",
    "OMR",
    "PAB",
    "PEN",
    "PGK",
    "PHP",
    "PKR",
    "PLN",
    "PYG",
    "QAR",
    "RON",
    "RSD",
    "RUB",
    "RWF",
    "SAR",
    "SBD",
    "SCR",
    "SDG",
    "SEK",
    "SGD",
    "SHP",
    "SLE",
    "SLL",
    "SOS",
    "SRD",
    "SSP",
    "STN",
    "SYP",
    "SZL",
    "THB",
    "TJS",
    "TMT",
    "TND",
    "TOP",
    "TRY",
    "TTD",
    "TVD",
    "TWD",
    "TZS",
    "UAH",
    "UGX",
    "USD",
    "UYU",
    "UZS",
    "VED",
    "VES",
    "VND",
    "VUV",
    "WST",
    "XAF",
    "XCD",
    "XOF",
    "XPF",
    "YER",
    "ZAR",
    "ZMW",
    "ZWL",
}

# Regex for numeric tokens with optional sign and decimal comma/point.
_NUMBER_RE = re.compile(r"^(?P<operator>[+\-])?(?P<value>[0-9]+(?:[.,][0-9]+)?)(?P<suffix>%|\s*%)?$")
_MONEY_RE = re.compile(r"^(?P<operator>[+\-])?(?P<amount>[0-9]+(?:[.,][0-9]+)?)\s*(?P<currency>[A-Z]{3})$")
_RANGE_RE = re.compile(r"^(?P<from>[0-9]+(?:[.,][0-9]+)?)\s*[-\u2013\u2014]\s*(?P<to>[0-9]+(?:[.,][0-9]+)?)$")


def _normalize_decimal(value: str) -> Decimal:
    """Parse a decimal string that may use comma or point as decimal separator."""
    cleaned = value.replace("\u00a0", "").replace(" ", "").replace("\u202f", "")
    has_dot = "." in cleaned
    has_comma = "," in cleaned
    if not has_dot and not has_comma:
        return Decimal(cleaned)

    if has_dot and not has_comma:
        # "1234.56" or "1.234.56" (unusual). Use dot as decimal; remove thousands dots.
        if cleaned.count(".") > 1:
            cleaned = cleaned.replace(".", "")
        return Decimal(cleaned)

    if has_comma and not has_dot:
        if cleaned.count(",") == 1:
            cleaned = cleaned.replace(",", ".")
        else:
            # Keep the last comma as decimal separator, remove the rest.
            parts = cleaned.split(",")
            cleaned = "".join(parts[:-1]) + "." + parts[-1]
        return Decimal(cleaned)

    # Both comma and dot present. Use the last separator as the decimal marker.
    last_dot = cleaned.rfind(".")
    last_comma = cleaned.rfind(",")
    cleaned = cleaned.replace(",", "") if last_dot > last_comma else cleaned.replace(".", "").replace(",", ".")
    return Decimal(cleaned)


def _to_canonical_string(value: Decimal) -> str:
    """Return a canonical decimal string without exponent notation."""
    normalized = value.normalize()
    return f"{normalized:f}"


_PLAIN_NUMBER_RE = re.compile(r"^[+\-]?[0-9]+(?:[.,][0-9]+)?$")


def is_numeric_amount(text: str) -> bool:
    """Return True if *text* is a plain numeric amount (no currency or %)."""
    return bool(_PLAIN_NUMBER_RE.match(text.strip()))


def parse_amount(text: str) -> str | None:
    """Normalize a plain numeric amount to a canonical decimal string."""
    match = _PLAIN_NUMBER_RE.match(text.strip())
    if not match:
        return None
    try:
        return _to_canonical_string(_normalize_decimal(match.group(0)))
    except (InvalidOperation, ValueError):
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

    # Percentage token.
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
        operator_name = None
        if operator == "+":
            operator_name = "add"
        elif operator == "-":
            operator_name = "subtract"
        token = FeeToken(
            raw=text,
            kind=kind,
            value=_to_canonical_string(value),
        )
        if operator_name:
            token = FeeToken(
                raw=token.raw,
                kind=token.kind,
                value=token.value,
                operator=operator_name,
            )
        return token

    # Money token.
    money_match = _MONEY_RE.match(text)
    if money_match:
        currency = money_match.group("currency").upper()
        if currency in CURRENCY_CODES:
            try:
                amount = _normalize_decimal(money_match.group("amount"))
            except (InvalidOperation, ValueError):
                return None
            operator = money_match.group("operator")
            operator_name = None
            if operator == "+":
                operator_name = "add"
            elif operator == "-":
                operator_name = "subtract"
            return FeeToken(
                raw=text,
                kind="money",
                amount=_to_canonical_string(amount),
                currency=currency,
                operator=operator_name,
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


_TOKENIZE_RE = re.compile(
    r"(?:(?P<currency_pre>[A-Z]{3})\s+)?(?P<operator>[+\-])?\s*(?P<value>[0-9]+(?:[.,][0-9]+)?)\s*(?P<currency_post>[A-Z]{3})?\s*(?P<suffix>%)?"
)


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
        if not match.group("suffix") and not match.group("currency_pre") and not match.group("currency_post") and not match.group("operator"):
            # Still accept if the surrounding text contains a % or currency nearby.
            start, end = match.start(), match.end()
            window = text[max(0, start - 10) : min(len(text), end + 10)]
            if "%" not in window and not any(part.upper() in CURRENCY_CODES for part in window.split()):
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


def render_rich_text_node(node: Any) -> Cell:
    """Render a Contentful-like rich-text node into a lossless Cell."""
    if node is None:
        return Cell(text="")
    if isinstance(node, str):
        return Cell(text=node, tokens=tokenize_text(node))
    if not isinstance(node, dict):
        return Cell(text=str(node))

    node_type = node.get("nodeType") or node.get("type") or ""
    text_parts: list[str] = []
    tokens: list[FeeToken] = []
    links: list[Link] = []

    if (
        node_type in {"document", "Document", "rich-text"}
        or node_type in {"paragraph", "Paragraph", "text", "Text"}
        or node_type in {"hyperlink", "link"}
        or node_type in {"list", "List"}
        or node_type in {"list-item", "ListItem"}
    ):
        content = node.get("content") or []
    elif node_type in {"EmbeddedEntryBlock", "embedded-entry", "embedded-entry-block", "embedded-entry-inline"}:
        text, tokens, links = _render_text_node(node)
        return Cell(text=text, tokens=tokens, links=links)
    elif "content" in node:
        content = node.get("content") or []
    else:
        # Single value node.
        value = node.get("value") or node.get("text") or node.get("displayValue") or ""
        if isinstance(value, dict):
            return render_rich_text_node(value)
        text = str(value).strip()
        return Cell(text=text)

    for child in content:
        if not isinstance(child, dict):
            text_parts.append(str(child))
            continue
        child_type = child.get("nodeType") or child.get("type") or ""
        if child_type in {"text", "Text"}:
            child_text, child_tokens, child_links = _render_text_node(child)
            text_parts.append(child_text)
            tokens.extend(child_tokens)
            links.extend(child_links)
        elif child_type in {"hyperlink", "link", "Hyperlink"}:
            # Render the link text and attach the URI.
            rendered = render_rich_text_node(child)
            if rendered.text.strip():
                for link in rendered.links:
                    links.append(Link(text=rendered.text, uri=link.uri))
                if not rendered.links:
                    data = child.get("data", {})
                    uri = data.get("uri") or data.get("href")
                    if uri:
                        links.append(Link(text=rendered.text, uri=uri))
                text_parts.append(rendered.text)
                tokens.extend(rendered.tokens)
        elif child_type in {"paragraph", "Paragraph", "list-item", "ListItem"}:
            rendered = render_rich_text_node(child)
            if rendered.text.strip():
                text_parts.append(rendered.text)
            tokens.extend(rendered.tokens)
            links.extend(rendered.links)
        elif child_type in {"line-break", "linebreak", "LineBreak"}:
            text_parts.append("\n")
        elif child_type in {"embedded-entry-block", "embedded-entry", "embedded-entry-inline"}:
            rendered = render_rich_text_node(child)
            if rendered.text.strip():
                text_parts.append(rendered.text)
            tokens.extend(rendered.tokens)
            links.extend(rendered.links)
        else:
            # Generic recursion.
            rendered = render_rich_text_node(child)
            if rendered.text.strip():
                text_parts.append(rendered.text)
            tokens.extend(rendered.tokens)
            links.extend(rendered.links)

    full_text = "".join(text_parts)
    # Normalize non-breaking spaces to regular spaces for readability but preserve in raw tokens.
    full_text = full_text.replace("\u00a0", " ").replace("\u202f", " ")
    return Cell(text=full_text.strip(), tokens=tokens, links=links)
