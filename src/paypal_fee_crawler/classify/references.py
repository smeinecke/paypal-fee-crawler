from __future__ import annotations

import logging
from typing import Any

from ..models import (
    Provenance,
    ResolvedRate,
    Row,
    Source,
    Table,
    TransactionFeeRule,
)
from .patterns import (
    _CLASSIFIER_VERSION,
    _REFERENCE_PRODUCT_SUFFIX,
    _REFERENCE_SCHEDULE_KEYWORDS,
    _REFERENCE_SUFFIX_TO_PRODUCT,
)
from .text_utils import (
    _NORMALIZED_PRODUCT_ALIASES,
    _first_money,
    _keyword_match,
    _market_code_from_url,
    _norm,
    _row_fee_cell,
    _row_has_percentage,
    _row_label,
)

logger = logging.getLogger(__name__)


def _detect_reference(row: Row, product_id: str | None) -> str | None:
    """Detect when a row does not contain a numeric rate but refers to another schedule."""
    if _row_has_percentage(row):
        return None
    label_text = _norm(_row_label(row))
    fee_text = _norm(_row_fee_cell(row))
    if not fee_text or "{{" in fee_text:
        return None
    # A single-cell row with a reference-looking label is usually a section
    # header, not a textual schedule reference.
    non_empty_cells = [c for c in row.cells if c.text.strip()]
    if len(non_empty_cells) == 1 and fee_text == label_text:
        return None
    # A reference is a textual pointer; if it already contains money, it is
    # likely a flat-fee rule, not a reference.
    if _first_money(row):
        return None
    for schedule_name, keywords in _REFERENCE_SCHEDULE_KEYWORDS.items():
        if _keyword_match(fee_text, keywords, word_boundary=False):
            suffix = _REFERENCE_PRODUCT_SUFFIX.get(product_id or "", "")
            if suffix:
                return f"{schedule_name}.{suffix}"
            return schedule_name
    return None


def _reference_product_id(reference: str) -> str | None:
    """Return the product id a textual reference points to."""
    if "." in reference:
        base, suffix = reference.split(".", 1)
        return _REFERENCE_SUFFIX_TO_PRODUCT.get(suffix, base)
    return reference


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _condition_score(rule: TransactionFeeRule, source_conditions: dict[str, Any]) -> int:
    """Score how specific a rule's conditions are relative to the source context.

    Higher scores mean the rule is a better match. Keys that are present in the
    source and match the rule are rewarded; extra, unrequested constraints in
    the rule are penalised. List-valued keys are also penalised by length (shorter
    lists are more specific), and the generic ``all_other_markets`` fallback is
    heavily penalised.
    """
    score = 0
    for key, rule_value in (rule.conditions or {}).items():
        if key in source_conditions:
            # A matching key is strong evidence that this rule is the right one.
            if key in ("applies_to_markets", "payment_methods"):
                if key == "applies_to_markets" and isinstance(rule_value, list) and "all_other_markets" in rule_value:
                    score -= 1000
                elif isinstance(rule_value, list):
                    score += 100 - len(rule_value)
                else:
                    score += 100
            else:
                score += 100
        else:
            # An extra, unrequested condition makes the rule too specific for a
            # generic source row.
            if key in ("applies_to_markets", "payment_methods"):
                penalty = 50
                if isinstance(rule_value, list):
                    penalty += len(rule_value)
                score -= penalty
            else:
                score -= 10
    return score


def _conditions_match_for_reference(
    rule_conditions: dict[str, Any],
    source_conditions: dict[str, Any],
) -> bool:
    """Return True when rule conditions are compatible with the source context.

    For ``applies_to_markets`` the source values must be included in the rule's
    list (the rule applies to the source market). For ``payment_methods`` the
    rule's methods must be a subset of the source's methods.  Scalar keys must
    match exactly.  When the source does not constrain a list key, the rule must
    not either (or use the generic ``all_other_markets`` fallback) so that a
    generic APM row does not resolve to a method-specific rule.
    """
    for key, rule_value in rule_conditions.items():
        source_value = source_conditions.get(key)
        if key == "applies_to_markets":
            if source_value is None or source_value == []:
                if _as_list(rule_value) == ["all_other_markets"]:
                    continue
                return False
            rule_markets = _as_list(rule_value)
            if not all(m in rule_markets for m in _as_list(source_value)):
                return False
        elif key == "payment_methods":
            if source_value is None or source_value == []:
                return False
            source_methods = _as_list(source_value)
            if not all(m in source_methods for m in _as_list(rule_value)):
                return False
        elif source_value is None:
            # The source has no constraint for this scalar key; only accept the
            # rule if its scalar value is the generic/default value, i.e. the
            # key is not actually narrowing the rule.
            continue
        elif rule_value != source_value:
            return False
    return True


def _reference_target_id(reference: str) -> str:
    """Resolve a textual reference to the target rule id it points to."""
    if "." not in reference:
        return reference
    base, suffix = reference.split(".", 1)
    suffix_product = _REFERENCE_SUFFIX_TO_PRODUCT.get(suffix)
    return suffix_product or _REFERENCE_SUFFIX_TO_PRODUCT.get(base, base)


def _reference_candidates(rules: list[TransactionFeeRule | None], target_id: str) -> list[TransactionFeeRule]:
    """Find candidate rules matching a reference target id or label aliases."""
    candidates = [r for r in rules if r is not None and r.id == target_id and r.percentage is not None]
    if not candidates:
        aliases = _NORMALIZED_PRODUCT_ALIASES.get(target_id, ())
        candidates = [
            r
            for r in rules
            if r is not None and r.label and _keyword_match(_norm(r.label), aliases, word_boundary=False)
        ]
    return candidates


def _resolved_source_conditions(
    candidates: list[TransactionFeeRule],
    source_conditions: dict[str, Any] | None,
    source: Provenance | None,
) -> dict[str, Any]:
    """Build source conditions, injecting the page market only when useful."""
    resolved = dict(source_conditions or {})
    if "applies_to_markets" not in resolved and source:
        market = _market_code_from_url(source.requested_url)
        if market and any(market in _as_list((r.conditions or {}).get("applies_to_markets")) for r in candidates):
            resolved["applies_to_markets"] = [market]
    return resolved


def _condition_matched_candidates(
    candidates: list[TransactionFeeRule],
    conditions: dict[str, Any],
) -> list[TransactionFeeRule]:
    """Filter candidates by condition compatibility and tie-break by specificity."""
    matched = [r for r in candidates if _conditions_match_for_reference(r.conditions or {}, conditions)]
    if matched and len(matched) > 1:
        max_score = max(_condition_score(r, conditions) for r in matched)
        matched = [r for r in matched if _condition_score(r, conditions) == max_score]
    return matched


def _build_resolved_rate(rule: TransactionFeeRule) -> ResolvedRate:
    return ResolvedRate(
        percentage=rule.percentage,
        fixed_fee_schedule=rule.fixed_fee_schedule,
        international_surcharge_schedule=rule.international_surcharge_schedule,
        maximum_fee_schedule=rule.maximum_fee_schedule,
        source=rule.source,
        rule_id=rule.id,
    )


def _resolve_reference(
    reference: str,
    rules: list[TransactionFeeRule | None],
    source_variant_id: str | None = None,
    source_conditions: dict[str, Any] | None = None,
    source: Provenance | None = None,
) -> tuple[ResolvedRate | None, bool]:
    """Resolve a textual reference to a concrete percentage and schedule names.

    A reference resolves only unambiguously. If more than one target rule
    matches, the reference is reported as ambiguous and ``(None, True)`` is
    returned. The source variant, source conditions and source provenance are
    used to disambiguate when the reference is tied to a specific variant or
    context.
    """
    target_id = _reference_target_id(reference)
    candidates = _reference_candidates(rules, target_id)
    if not candidates:
        return None, False
    if len(candidates) == 1:
        return _build_resolved_rate(candidates[0]), False

    resolved_source_conditions = _resolved_source_conditions(candidates, source_conditions, source)

    matched = _condition_matched_candidates(candidates, resolved_source_conditions)
    if matched:
        if len(matched) == 1:
            return _build_resolved_rate(matched[0]), False
        candidates = matched

    if resolved_source_conditions and "transaction_region" in resolved_source_conditions:
        relaxed = _condition_matched_candidates(
            candidates,
            {k: v for k, v in resolved_source_conditions.items() if k != "transaction_region"},
        )
        if relaxed:
            if len(relaxed) == 1:
                return _build_resolved_rate(relaxed[0]), False
            candidates = relaxed

    if source_variant_id is not None:
        matched = [r for r in candidates if r.variant_id == source_variant_id]
        if len(matched) == 1:
            return _build_resolved_rate(matched[0]), False
        if not matched:
            default_candidates = [r for r in candidates if r.variant_id in (None, "default", "standard")]
            if len(default_candidates) == 1:
                return _build_resolved_rate(default_candidates[0]), False

    return None, True


def _provenance(
    table: Table,
    row: Row,
    row_index: int,
    source: Source | None,
    section_heading: str | None = None,
    original_label: str | None = None,
) -> Provenance:
    return Provenance(
        requested_url=source.requested_url if source else None,
        canonical_url=source.canonical_url if source else None,
        page_id=source.page_id if source else None,
        page_title=source.page_title if source else None,
        document_id=row.source_document_id or table.document_id,
        component_id=row.source_component_id or table.component_id,
        table_id=table.table_id,
        row_id=row.row_id,
        row_index=row_index,
        section_heading=section_heading or (table.section_path[-1] if table.section_path else table.caption),
        original_label=original_label,
        classifier_version=_CLASSIFIER_VERSION,
    )
