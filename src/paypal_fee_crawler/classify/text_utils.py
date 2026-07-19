from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from functools import lru_cache
from typing import Any

from ..models import (
    Row,
    Source,
    Table,
)
from ..normalize import clean_text, normalize_decimal_string
from ..pricing_tokens import CURRENCY_CODES
from .patterns import (
    _CANONICAL_AMOUNT_RE,
    _DEFAULT_CURRENCY_BY_MARKET,
    _FEE_HEADER_KEYWORDS,
    _NON_FEE_HEADER_KEYWORDS,
    _PERCENTAGE_RE,
    _PLUS_FIXED_RE,
    _PRODUCT_ALIASES,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=100_000)
def _norm(text: str | None) -> str:
    return clean_text(text or "").lower()


_NORMALIZED_PRODUCT_ALIASES: dict[str, tuple[str, ...]] = {
    product: tuple(_norm(alias) for alias in aliases) for product, aliases in _PRODUCT_ALIASES.items()
}


@lru_cache(maxsize=1024)
def _compile_keyword_pattern(keywords: tuple[str, ...], word_boundary: bool) -> re.Pattern[str]:
    """Return a compiled alternation regex for a static keyword group."""
    escaped = [re.escape(kw) for kw in keywords]
    pattern = r"(?<!\w)(?:" + "|".join(escaped) + r")(?!\w)" if word_boundary else "|".join(escaped)
    return re.compile(pattern)


def _keyword_match(text: str, keywords: Iterable[str], word_boundary: bool = True) -> bool:
    """Return True when any keyword in ``keywords`` is found in ``text``.

    The matcher is unified across word-boundary and substring lookups and
    pre-compiles static keyword groups so ``re.escape`` is not called per row.
    """
    if not text:
        return False
    keywords_tuple = tuple(keywords)
    if not keywords_tuple:
        return False
    pattern = _compile_keyword_pattern(keywords_tuple, word_boundary)
    return bool(pattern.search(text))


def _keyword_in_text(text: str, keyword: str) -> bool:
    """Return True when ``keyword`` appears as a whole word/phrase in ``text``."""
    # Use word boundaries to avoid matching the keyword as a substring inside a
    # larger word (e.g. Portuguese "até" inside Czech "přijaté").  This keeps
    # punctuation-delimited tokens such as "<" and ">" working as well.
    return _keyword_match(text, (keyword,), word_boundary=True)


def _table_text(table: Table) -> str:
    parts = list(table.section_path or []) + [table.caption or ""]
    for header in table.headers:
        parts.append(header.text)
    return _norm(" ".join(parts))


def _table_context_original(table: Table) -> str:
    """Return original-case table heading context for applicability parsing."""
    parts = list(table.section_path or []) + [table.caption or ""]
    return " ".join(p for p in parts if p)


def _row_cells_text(row: Row) -> list[str]:
    return [c.text for c in row.cells]


def _text_indicates_percentage(text: str | None) -> bool:
    """Return True if text contains a percentage marker or spelling."""
    if not text:
        return False
    lowered = text.lower()
    return "%" in lowered or "prozentpunkt" in lowered or "percentage point" in lowered


def _token_text_indicates_percentage(token) -> bool:
    """Return True if token metadata describes a percentage value.

    PayPal embeds some percentage-point surcharges as raw numbers whose
    internal name or fee-data key contains the word "Prozentpunkte" or
    "percentage points".
    """
    for candidate in (token.raw, token.internal_name, token.fee_data_key):
        if _text_indicates_percentage(candidate):
            return True
    return False


def _first_percentage(row: Row) -> str | None:
    for cell in row.cells:
        cell_indicates_pct = _text_indicates_percentage(cell.text)
        for token in cell.tokens:
            if token.kind == "percentage" and token.value:
                return token.value
            if (
                token.kind == "number"
                and token.value
                and (cell_indicates_pct or _token_text_indicates_percentage(token))
            ):
                return token.value
    return None


def _first_money(row: Row) -> tuple[str, str] | None:
    for cell in row.cells:
        for token in cell.tokens:
            if token.kind == "money" and token.amount and token.currency:
                return token.currency, token.amount
    return None


def _cell_money(cell: Any) -> tuple[str, str] | None:
    for token in cell.tokens:
        if token.kind == "money" and token.amount and token.currency:
            return token.currency, token.amount
    return None


def _row_has_percentage(row: Row) -> bool:
    return _first_percentage(row) is not None


def _row_label(row: Row) -> str:
    """Return the label cell of a row (typically the first non-empty cell)."""
    for cell in row.cells:
        if cell.text.strip():
            return cell.text.strip()
    return ""


def _row_fee_cell(row: Row) -> str:
    """Return the fee/rate cell of a row (typically the last non-empty cell)."""
    for cell in reversed(row.cells):
        if cell.text.strip():
            return cell.text.strip()
    return ""


def _infer_currency_for_row(row: Row, table: Table, source: Source | None) -> str | None:
    """Return an explicit or inferred ISO 4217 currency code for a row."""
    # Money tokens already carry a currency.
    for cell in row.cells:
        for token in cell.tokens:
            if token.kind == "money" and token.currency:
                return token.currency
    # Look for an explicit three-letter currency code anywhere in the row or headers.
    sources = [cell.text for cell in row.cells]
    sources += [h.text for h in table.headers]
    sources += [table.caption or ""] + list(table.section_path)
    for text in sources:
        for part in re.findall(r"(?<!\w)[A-Z]{3}(?!\w)", text):
            if part in CURRENCY_CODES:
                return part
    # Fall back to the market default from the source URL.
    if source and source.requested_url:
        market = _market_code_from_url(source.requested_url)
        if market:
            return _DEFAULT_CURRENCY_BY_MARKET.get(market)
    return None


def _parse_canonical_amount(amount_str: str) -> str | None:
    """Return a canonical decimal string for an amount that may use thousands or decimal separators."""
    amount_str = amount_str.replace("\u00a0", "").replace("\u202f", "").replace(" ", "")
    if not amount_str:
        return None
    # Reject non-numeric strings before trying to interpret separators.
    if not _CANONICAL_AMOUNT_RE.fullmatch(amount_str):
        return None
    has_dot = "." in amount_str
    has_comma = "," in amount_str
    if not has_dot and not has_comma:
        return normalize_decimal_string(amount_str)
    if has_dot and has_comma:
        last_dot = amount_str.rfind(".")
        last_comma = amount_str.rfind(",")
        if last_dot > last_comma:
            decimal_str = amount_str.replace(",", "")
        else:
            decimal_str = amount_str.replace(".", "").replace(",", ".")
        return normalize_decimal_string(decimal_str)
    sep = "." if has_dot else ","
    parts = amount_str.split(sep)
    if len(parts) == 2:
        int_part, frac_part = parts
        if len(frac_part) in (1, 2):
            decimal = True
        elif len(frac_part) == 3:
            decimal = not (int_part.isdigit() and len(int_part) <= 3)
        else:
            decimal = True
        if decimal:
            if sep == ",":
                return normalize_decimal_string(amount_str.replace(",", "."))
            return normalize_decimal_string(amount_str)
        return int_part + frac_part
    if len(parts[-1]) == 3 and all(len(p) <= 3 for p in parts[:-1]):
        return "".join(parts)
    decimal_str = "".join(parts[:-1]) + "." + parts[-1]
    return normalize_decimal_string(decimal_str)


def _cell_looks_like_fee_cell(header_text: str, table: Table) -> bool:
    """Return True when a numeric cell is plausibly a fee value, not a code/prefix."""
    header_lower = _norm(header_text)
    if _keyword_match(header_lower, _NON_FEE_HEADER_KEYWORDS, word_boundary=False):
        return False
    if _keyword_match(header_lower, _FEE_HEADER_KEYWORDS, word_boundary=False):
        return True
    table_lower = _norm((table.caption or "") + " ".join(table.section_path))
    return _keyword_match(table_lower, _FEE_HEADER_KEYWORDS, word_boundary=False)


def _last_non_empty_cell_text(row: Row) -> str:
    for cell in reversed(row.cells):
        text = cell.text.strip()
        if text:
            return text
    return ""


def _has_likely_numeric_fee_candidate(row: Row, table: Table) -> bool:
    """Return True when a row contains a probable money or number fee value."""
    last_cell_text = _last_non_empty_cell_text(row)
    for i, cell in enumerate(row.cells):
        for token in cell.tokens:
            if token.kind == "money":
                return True
            if token.kind == "number" and token.value:
                header = table.headers[i].text if i < len(table.headers) else ""
                if _cell_looks_like_fee_cell(header, table):
                    return True
                # A bare number in the last (fee) cell of a row is a fee
                # candidate unless the header explicitly marks it as a code or prefix.
                if cell.text.strip() == last_cell_text:
                    header_lower = _norm(header)
                    if not _keyword_match(header_lower, _NON_FEE_HEADER_KEYWORDS, word_boundary=False):
                        return True
    return False


def _first_variant_match(text: str, rules: Iterable[tuple[Iterable[str], str]]) -> str | None:
    for keywords, variant_id in rules:
        if _keyword_match(text, keywords, word_boundary=True):
            return variant_id
    return None


def _all_variant_matches(text: str, rules: Iterable[tuple[Iterable[str], str]]) -> list[str]:
    """Return all variant ids whose keywords appear in the normalized text."""
    seen: set[str] = set()
    result: list[str] = []
    for keywords, variant_id in rules:
        if _keyword_match(text, keywords, word_boundary=True) and variant_id not in seen:
            seen.add(variant_id)
            result.append(variant_id)
    return result


def _market_code_from_url(url: str | None) -> str | None:
    """Return the 2-letter market code from a PayPal URL path, if present."""
    if not url:
        return None
    match = re.search(r"paypal\.com/([a-zA-Z0-9]+)/", url)
    if match:
        return match.group(1).upper()
    return None


def _parse_rate_expression(fee_text: str) -> tuple[str | None, str | None]:
    """Parse a German/English percentage + fixed-fee expression.

    Returns (percentage, fixed_fee_currency_amount_text).
    """
    pct: str | None = None
    # Find a percentage token anywhere in the text.
    for match in _PERCENTAGE_RE.finditer(fee_text):
        pct = normalize_decimal_string(match.group(1))
        break
    # Money amount is everything after the plus/extra token, if present.
    fixed: str | None = None
    plus_match = _PLUS_FIXED_RE.search(fee_text)
    if plus_match:
        fixed = plus_match.group(1).strip()
    return pct, fixed
