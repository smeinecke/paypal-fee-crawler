from __future__ import annotations

import logging

from ..models import (
    AmbiguousFeeRow,
    Row,
    Source,
    Table,
    UnclassifiedFeeRow,
)
from .apm import _is_apm_special_label
from .patterns import (
    _DIRECT_FIXED_FEE_KEYWORDS,
    _FIXED_FEE_KEYWORDS,
    _INTERNATIONAL_SURCHARGE_KEYWORDS,
    _LIMIT_OR_CAP_KEYWORDS,
    _MIN_MAX_FEE_KEYWORDS,
    _PRODUCT_CATEGORY_MAP,
    _TABLE_CATEGORY_KEYWORDS,
    _TABLE_NEGATIVE_SIGNALS,
)
from .references import _detect_reference, _provenance, _reference_product_id
from .text_utils import (
    _NORMALIZED_PRODUCT_ALIASES,
    _has_likely_numeric_fee_candidate,
    _keyword_match,
    _norm,
    _parse_rate_expression,
    _row_cells_text,
    _row_fee_cell,
    _row_has_percentage,
    _row_label,
    _table_text,
    _text_indicates_percentage,
)

logger = logging.getLogger(__name__)


def _score_label_against_product(label: str, normalized_aliases: tuple[str, ...]) -> int:
    normalized = _norm(label)
    best = 0
    for alias in normalized_aliases:
        if alias == normalized:
            return max(best, len(alias) * 10)
        if alias in normalized:
            best = max(best, len(alias))
    return best


def _classify_product(label: str) -> tuple[str | None, list[str]]:
    """Return the best matching product ID and any ambiguous alternatives."""
    scores: dict[str, int] = {}
    for product_id, aliases in _NORMALIZED_PRODUCT_ALIASES.items():
        score = _score_label_against_product(label, aliases)
        if score:
            scores[product_id] = score
    if not scores:
        return None, []
    max_score = max(scores.values())
    candidates = sorted([pid for pid, sc in scores.items() if sc == max_score])
    if len(candidates) > 1:
        return None, candidates
    return candidates[0], []


def _is_currency_conversion_text(text: str) -> bool:
    """Return True if the table text describes a currency conversion table."""
    return (
        "währungsumrechnung" in text
        or "umrechnung des guthabens" in text
        or "currency conversion" in text
        or "converting payments" in text
        or "conversions in" in text
        or ("converting" in text and "currency" in text)
        or ("conversion" in text and "currency" in text)
    )


def _is_maximum_fee_table(text: str) -> bool:
    """Return True if the table is a payout/withdrawal maximum fee cap table.

    ``text`` is expected to already be normalized.
    """
    return ("payout" in text or "withdrawal" in text or "withdraw" in text or "payouts" in text) and (
        "maximum fee cap" in text
        or "max fee cap" in text
        or "maximum payout fee" in text
        or "max payout fee" in text
        or ("fee" in text and ("max cap" in text or "maximum cap" in text))
    )


def _is_withdrawals_rate_table(table: Table, text: str) -> bool:
    """Return True if the table is a withdrawals/payouts rate table.

    ``text`` is expected to already be normalized.
    """
    if not (
        "payout" in text
        or "withdrawal" in text
        or "withdraw" in text
        or "payouts" in text
        or "wypłaty" in text
        or "wypłata" in text
        or "výběry" in text
        or "výběr" in text
        or "výbery" in text
    ):
        return False
    # Look for a Rate/% column. Tables that merely list limits or currencies are
    # not rate tables.
    header_text = " ".join(h.text for h in table.headers)
    if "rate" in _norm(header_text) or "%" in header_text:
        return True
    # Some rate tables put the rate in the second column without a header.
    for row in table.rows:
        cells = [c.text for c in row.cells if c.text.strip()]
        if any("%" in c or "rate" in _norm(c) for c in cells):
            return True
    return False


def _table_has_fixed_fee_rate(table: Table) -> bool:
    """Return True when any data row of ``table`` contains a percentage plus a fixed fee."""
    for row in table.rows:
        fee_text = _row_fee_cell(row)
        _, has_fixed = _parse_rate_expression(fee_text)
        if has_fixed:
            return True
    return False


def _classify_table_category(table: Table) -> str | None:
    text = _table_text(table)
    # Explicit schedule-type captions are authoritative and win over product
    # rate-table keywords such as "commercial transactions" or "donations".
    if _keyword_match(text, _FIXED_FEE_KEYWORDS, word_boundary=False):
        return "fixed_fee_table"

    # Maximum fee cap tables (e.g. "Maximum fee cap for PayPal Payouts") are
    # fee schedules, not generic limits.
    if _is_maximum_fee_table(text):
        return "maximum_fee_table"

    # Limits, caps, min/max and ceiling/floor tables are not transaction fees
    # and must be detected before direct fixed or rate-table keywords.
    if _keyword_match(text, _MIN_MAX_FEE_KEYWORDS, word_boundary=False):
        return "min_max_fee_table"

    # Some tables are captioned with international-surcharge language but
    # actually list full transaction rates (percentage + fixed fee) by buyer
    # country. Classify these as commercial rate tables so the rows become
    # product rules with market applicability instead of surcharge schedules.
    if (
        "receiving international transactions" in text or "sending international transactions" in text
    ) and _table_has_fixed_fee_rate(table):
        return "commercial_rate_table"

    # Withdrawals/payouts with a Rate column are rate tables (e.g. "Sending
    # PayPal Payouts"). This must come before direct_fixed because those
    # keywords also match "payout" / "withdrawal".
    if _is_withdrawals_rate_table(table, text):
        return "withdrawals_rate_table"

    # Direct monetary fee tables (chargebacks, disputes, withdrawals, refunds,
    # card verification, authorisation) are not generic fixed-fee schedules and
    # must be identified separately.
    if _keyword_match(text, _DIRECT_FIXED_FEE_KEYWORDS, word_boundary=False):
        return "fixed_fee_table"

    if _keyword_match(text, _INTERNATIONAL_SURCHARGE_KEYWORDS, word_boundary=False):
        return "international_surcharge_table"
    if _is_currency_conversion_text(text):
        return "currency_conversion_table"

    category = _select_category_from_scores(table, text)
    # Tables that score as international surcharge schedules but actually
    # contain full percentage + fixed-fee rates are commercial rate tables
    # (e.g. "Receiving international transactions").
    if category == "international_surcharge_table" and _table_has_fixed_fee_rate(table):
        return "commercial_rate_table"
    return category


def _is_limit_or_cap_row(label: str, fee_text: str = "") -> bool:
    """Return True if a row describes a limit, cap, min/max or ceiling/floor."""
    if _text_indicates_percentage(fee_text):
        return False
    combined = _norm(label + " " + fee_text)
    return _keyword_match(combined, _LIMIT_OR_CAP_KEYWORDS, word_boundary=False)


def _select_category_from_scores(table: Table, text: str) -> str | None:
    """Score table text against category keywords and resolve the best category."""
    scores: dict[str, int] = {}
    for category, keywords in _TABLE_CATEGORY_KEYWORDS.items():
        score = 0
        for kw in keywords:
            kw_norm = _norm(kw)
            if kw_norm in text:
                score = max(score, len(kw_norm))
        if score:
            scores[category] = score
    if not scores:
        return _classify_table_by_row_labels(table)
    candidates = _top_category_candidates(scores)
    candidates = _filter_category_negative_signals(candidates, text)
    if not candidates:
        return _fallback_category_candidate(scores, text, table)
    return candidates[0]


def _top_category_candidates(scores: dict[str, int]) -> list[str]:
    max_score = max(scores.values())
    return [cat for cat, sc in scores.items() if sc == max_score]


def _filter_category_negative_signals(candidates: list[str], text: str) -> list[str]:
    kept: list[str] = []
    for category in candidates:
        negatives = _TABLE_NEGATIVE_SIGNALS.get(category, ())
        if _keyword_match(text, tuple(map(_norm, negatives)), word_boundary=False):
            continue
        kept.append(category)
    return kept


def _fallback_category_candidate(scores: dict[str, int], text: str, table: Table) -> str | None:
    # If the top candidates were removed, fall back to the next-highest-scoring
    # category or to row-label inference.
    removed = set(_TABLE_NEGATIVE_SIGNALS.keys())
    remaining = {cat: sc for cat, sc in scores.items() if cat not in removed}
    if not remaining:
        remaining = scores
    next_score = max(remaining.values())
    candidates = [cat for cat, sc in remaining.items() if sc == next_score]
    candidates = _filter_category_negative_signals(candidates, text)
    if len(candidates) == 1:
        return candidates[0]
    return _classify_table_by_row_labels(table)


def _classify_table_by_row_labels(table: Table) -> str | None:
    """Infer a table category from the product ids of its rows."""
    category_counts: dict[str, int] = {}
    for row in table.rows:
        label = _row_label(row)
        if not label:
            continue
        product_id, _ = _classify_product(label)
        if product_id:
            category = _PRODUCT_CATEGORY_MAP.get(product_id)
            if category:
                category_counts[category] = category_counts.get(category, 0) + 1
    if not category_counts:
        return None
    best_category, _ = max(category_counts.items(), key=lambda kv: kv[1])
    # Negative signals can override a row-label inference (e.g. "Other fees"
    # tables that happen to contain a credit-card row should be ignored).
    negatives = _TABLE_NEGATIVE_SIGNALS.get(best_category, ())
    text = _table_text(table)
    if _keyword_match(text, tuple(map(_norm, negatives)), word_boundary=False):
        return None
    return best_category


def _classify_product_or_apm(label: str) -> tuple[str | None, list[str]]:
    """Classify a row label, treating APM special labels as unambiguous APM."""
    if _is_apm_special_label(label):
        return "alternative_payment_methods", []
    return _classify_product(label)


def _resolve_ambiguous_product(
    label: str,
    row: Row,
    idx: int,
    table: Table,
    source: Source | None,
    ambiguous_candidates: list[str],
    default_product: str | None,
    force_default_product: bool,
    ambiguous: list[AmbiguousFeeRow],
    ignored: list[UnclassifiedFeeRow],
) -> str | None:
    """Decide how to handle a row with ambiguous product candidates.

    Returns the resolved product id, or None if the row is queued for
    ``ambiguous`` / ``ignored``.
    """
    if force_default_product and default_product:
        return default_product
    if _row_has_percentage(row) or _has_likely_numeric_fee_candidate(row, table):
        ambiguous.append(
            AmbiguousFeeRow(
                normalized_cells=_row_cells_text(row),
                original_label=label,
                source=_provenance(table, row, idx, source, original_label=label),
                candidates=ambiguous_candidates,
            )
        )
        return None
    # A row with no determinable rate is informational, not a genuine ambiguity.
    ignored.append(
        UnclassifiedFeeRow(
            normalized_cells=_row_cells_text(row),
            original_label=label,
            source=_provenance(table, row, idx, source, original_label=label),
            reason="ambiguous product without fee",
        )
    )
    return None


def _resolve_missing_product(
    label: str,
    row: Row,
    idx: int,
    table: Table,
    source: Source | None,
    reference: str | None,
    default_product: str | None,
    force_default_product: bool,
    unclassified: list[UnclassifiedFeeRow],
    ignored: list[UnclassifiedFeeRow],
) -> str | None:
    """Resolve a product id for rows that did not match any product alias."""
    # Category-specific tables always fall back to their default product when
    # the label is not a product name.
    if force_default_product and default_product and (_row_has_percentage(row) or reference):
        return default_product
    # For mixed-product rate tables (e.g. commercial), use the default product
    # only when the row has its own rate and does not explicitly reference a
    # different product family.
    if default_product and not reference and _row_has_percentage(row):
        return default_product
    if reference:
        ref_product = _reference_product_id(reference)
        if ref_product:
            # A reference row that carries no product alias should be tagged
            # with the product it points to rather than the table's default.
            return ref_product
    if len(label) > 3 and _row_has_percentage(row):
        unclassified.append(
            UnclassifiedFeeRow(
                normalized_cells=_row_cells_text(row),
                original_label=label,
                source=_provenance(table, row, idx, source, original_label=label),
                reason="no product alias matched",
            )
        )
        return None
    if len(label) > 3 and _has_likely_numeric_fee_candidate(row, table):
        unclassified.append(
            UnclassifiedFeeRow(
                normalized_cells=_row_cells_text(row),
                original_label=label,
                source=_provenance(table, row, idx, source, original_label=label),
                reason="unclassified_fee_candidate",
            )
        )
        return None
    ignored.append(
        UnclassifiedFeeRow(
            normalized_cells=_row_cells_text(row),
            original_label=label,
            source=_provenance(table, row, idx, source, original_label=label),
            reason="no product alias and no rate",
        )
    )
    return None


def _resolve_product_id(
    label: str,
    row: Row,
    idx: int,
    table: Table,
    source: Source | None,
    default_product: str | None,
    force_default_product: bool,
    unclassified: list[UnclassifiedFeeRow],
    ambiguous: list[AmbiguousFeeRow],
    ignored: list[UnclassifiedFeeRow],
) -> tuple[str | None, str | None]:
    """Determine the product id and textual reference for a single table row."""
    product_id, ambiguous_candidates = _classify_product_or_apm(label)
    if ambiguous_candidates:
        product_id = _resolve_ambiguous_product(
            label,
            row,
            idx,
            table,
            source,
            ambiguous_candidates,
            default_product,
            force_default_product,
            ambiguous,
            ignored,
        )
        if product_id is None:
            return None, None
    if force_default_product and default_product:
        product_id = default_product

    reference = _detect_reference(row, product_id)
    if product_id is None:
        product_id = _resolve_missing_product(
            label,
            row,
            idx,
            table,
            source,
            reference,
            default_product,
            force_default_product,
            unclassified,
            ignored,
        )
        if product_id is None:
            return None, None
    return product_id, reference
