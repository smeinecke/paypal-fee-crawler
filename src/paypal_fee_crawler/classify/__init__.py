from __future__ import annotations

import json
import logging
from typing import Any

from ..models import (
    AmbiguousFeeRow,
    CoverageSummary,
    CurrencyConversion,
    DerivedFeeResult,
    Diagnostic,
    FeeComponent,
    FixedFeeSchedule,
    InternationalSurchargeSchedule,
    RateReference,
    Source,
    Table,
    TransactionFeeRule,
    UnclassifiedFeeRow,
)
from .apm import (
    _extract_apm_methods,
    _is_apm_special_label,
    _is_domestic_label,
    _is_international_label,
    _tokenize_apm_label,
)
from .conditions import (
    _card_payment_methods_from_label,
    _conditions_for_advanced_card,
    _conditions_for_apm,
    _conditions_for_donations,
    _conditions_for_other_commercial,
    _conditions_for_paypal_checkout,
    _conditions_for_pos,
    _conditions_for_row,
    _extract_amount_condition,
    _extract_country_group_condition,
    _is_charity_label,
    _is_generic_apm_label,
    _is_generic_other_commercial_label,
    _maximum_fee_schedule_for_conditions,
    _pricing_plan_for_label,
    _service_for_advanced_card,
    _service_for_donation_label,
    _service_for_other_commercial_ach,
    _transaction_region_for_variant,
)

# Re-export all public and private helpers
from .patterns import (
    _ADVANCED_CARD_SCHEDULE_KEYWORDS,
    _ADVANCED_CARD_VARIANTS,
    _APM_EXAMPLE_PHRASE_RE,
    _APM_HEADER_PHRASES,
    _APM_HEADER_TOKENS,
    _APM_METHOD_ALIASES,
    _APM_METHOD_MATCHERS,
    _APM_PUNCTUATION_RE,
    _APM_SEPARATOR_RE,
    _APM_SORTED_ALIASES,
    _APM_SPECIAL_METHOD_IDS,
    _APM_VARIANT_ONLY_VARIANTS,
    _APM_VARIANTS,
    _BANK_TOKENS,
    _BASE_VARIANTS_BY_PRODUCT,
    _CANONICAL_AMOUNT_RE,
    _CARD_VERIFICATION_VARIANTS,
    _CATEGORY_SPECIFIC_TABLES,
    _CLASSIFIER_VERSION,
    _DEFAULT_CURRENCY_BY_MARKET,
    _DIRECT_FIXED_FEE_KEYWORDS,
    _DIRECT_FIXED_FEE_PRODUCTS,
    _DIRECT_FIXED_FEE_SCHEDULE_PRODUCTS,
    _DISPUTE_VARIANTS,
    _DONATIONS_VARIANTS,
    _ESTONIAN_TOKENS,
    _FEE_HEADER_KEYWORDS,
    _FIXED_FEE_INHERITANCE,
    _FIXED_FEE_KEYWORDS,
    _FIXED_FEE_SCHEDULE_FALLBACK,
    _FIXED_FEE_SCHEDULE_FOR,
    _FRAUD_PROTECTION_VARIANTS,
    _INTERNATIONAL_SURCHARGE_INHERITANCE,
    _INTERNATIONAL_SURCHARGE_KEYWORDS,
    _INTERNATIONAL_SURCHARGE_SCHEDULE_FALLBACK,
    _INTERNATIONAL_SURCHARGE_SCHEDULE_FOR,
    _INVOICE_VARIANTS,
    _LATVIAN_TOKENS,
    _LIMIT_OR_CAP_KEYWORDS,
    _LITHUANIAN_TOKENS,
    _MAXIMUM_FEE_SCHEDULE_FALLBACK,
    _MICROPAYMENT_VARIANTS,
    _MIN_MAX_FEE_KEYWORDS,
    _NON_FEE_HEADER_KEYWORDS,
    _NONPROFIT_VARIANTS,
    _ONLINE_TOKENS,
    _OTHER_COMMERCIAL_VARIANTS,
    _PAY_LATER_VARIANTS,
    _PAYPAL_CHECKOUT_VARIANTS,
    _PERCENTAGE_RE,
    _PLUS_FIXED_RE,
    _POS_VARIANTS,
    _PRODUCT_ALIASES,
    _PRODUCT_CATEGORY_MAP,
    _PRODUCT_ORDER,
    _QR_ABOVE_THRESHOLD,
    _QR_BELOW_THRESHOLD,
    _QR_VARIANTS,
    _RECORDS_REQUEST_VARIANTS,
    _REFERENCE_PRODUCT_SUFFIX,
    _REFERENCE_SCHEDULE_KEYWORDS,
    _REFERENCE_SUFFIX_TO_PRODUCT,
    _REGION_EXACT,
    _REGION_PATTERNS,
    _SCHEDULE_NAME_FROM_TABLE_MAPPING,
    _SEPA_DIRECT_DEBIT_VARIANTS,
    _STATUS_DEFECT_DIAGNOSTICS,
    _TABLE_CATEGORY_KEYWORDS,
    _TABLE_CATEGORY_PRODUCT,
    _TABLE_CATEGORY_SCHEDULE,
    _TABLE_NEGATIVE_SIGNALS,
    _THAI_TOKENS,
    _VARIANT_RULES_BY_PRODUCT,
    _WITHDRAWAL_VARIANTS,
    CLASSIFIER_VERSION,
    RegionPattern,
)
from .products import (
    _classify_product,
    _classify_product_or_apm,
    _classify_table_by_row_labels,
    _classify_table_category,
    _fallback_category_candidate,
    _filter_category_negative_signals,
    _is_currency_conversion_text,
    _is_limit_or_cap_row,
    _is_maximum_fee_table,
    _is_withdrawals_rate_table,
    _resolve_ambiguous_product,
    _resolve_missing_product,
    _resolve_product_id,
    _score_label_against_product,
    _select_category_from_scores,
    _table_has_fixed_fee_rate,
    _top_category_candidates,
)
from .references import (
    _as_list,
    _build_resolved_rate,
    _condition_matched_candidates,
    _condition_score,
    _conditions_match_for_reference,
    _detect_reference,
    _provenance,
    _reference_candidates,
    _reference_product_id,
    _reference_target_id,
    _resolve_reference,
    _resolved_source_conditions,
)
from .rules import (
    _build_direct_fixed_rules,
    _build_standard_rate_rule,
    _create_direct_fixed_fee_rules,
    _direct_fixed_product_variant,
    _extract_direct_fixed_amounts,
    _extract_rules_from_rate_table,
    _ExtractedRule,
    _handle_unusable_rate_row,
    _ignored_rate_row,
)
from .schedules import (
    _collect_fixed_fee_table,
    _collect_international_surcharge_table,
    _collect_maximum_fee_table,
    _collect_schedules,
    _conditions_from_schedule_id,
    _create_inherited_schedule,
    _extract_fixed_fee_schedule,
    _extract_international_surcharge_schedule,
    _extract_maximum_fee_schedule,
    _fixed_fee_schedule_for,
    _inheritance_evidence,
    _inheritance_map_for,
    _international_surcharge_schedule_for,
    _matches_region_pattern,
    _max_fee_base_id,
    _max_fee_money,
    _max_fee_schedule_sources,
    _merge_fixed_like_schedules,
    _merge_international_surcharge_schedules,
    _merge_max_fee_schedule,
    _merge_provenance_sources,
    _normalize_region,
    _product_family_for_schedule_id,
    _resolve_schedule,
    _resolve_schedule_inheritance,
    _schedule_heading_text,
    _schedule_id,
    _schedule_ids_for_table,
    _schedule_name_from_table,
    _schedule_signature_for_row,
    _schedule_suffix_from_signature,
    _select_schedule_id,
    _signature_from_conditions,
    _signature_key,
    _source_schedule_id,
    _source_text_evidence,
    _source_text_of,
    _table_context_evidence,
    _validate_inheritance_priorities,
)
from .text_utils import (
    _NORMALIZED_PRODUCT_ALIASES,
    _all_variant_matches,
    _cell_looks_like_fee_cell,
    _cell_money,
    _first_money,
    _first_percentage,
    _first_variant_match,
    _has_likely_numeric_fee_candidate,
    _infer_currency_for_row,
    _keyword_in_text,
    _last_non_empty_cell_text,
    _market_code_from_url,
    _norm,
    _parse_canonical_amount,
    _parse_rate_expression,
    _row_cells_text,
    _row_fee_cell,
    _row_has_percentage,
    _row_label,
    _table_context_original,
    _table_text,
    _text_indicates_percentage,
    _token_text_indicates_percentage,
)
from .variants import (
    _VARIANT_DISPATCH,
    _applicable_variants_for_table,
    _is_sending_donation_table,
    _variant_for_advanced_card,
    _variant_for_apm,
    _variant_for_card_verification,
    _variant_for_donations,
    _variant_for_fraud_protection,
    _variant_for_invoice_pay_later,
    _variant_for_micropayments,
    _variant_for_nonprofit,
    _variant_for_other_commercial,
    _variant_for_pay_later_consumer,
    _variant_for_paypal_checkout,
    _variant_for_pos_transactions,
    _variant_for_qr_code,
    _variant_for_records_request,
    _variant_for_sepa_direct_debit,
    _variant_for_withdrawals,
    _variant_id_for_row,
)

"""Derive product-specific transaction fee rules from normalized PayPal tables.

The classifier works at the row level: a single PayPal table may contain several
independent payment products, and each relevant fee row becomes a separate
``TransactionFeeRule``.  Fixed-fee and international-surcharge schedules are kept
separate per product or product family so that an HTTP fee calculator can select
the schedule that applies to a given rule.
"""

logger = logging.getLogger(__name__)


def _derive_status(
    rules: list[TransactionFeeRule],
    unclassified: list[UnclassifiedFeeRow],
    ambiguous: list[AmbiguousFeeRow],
    ignored: list[UnclassifiedFeeRow],
    diagnostics: list[Diagnostic],
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
) -> str:
    if not rules and not fixed_schedules and not international_schedules and not maximum_fee_schedules:
        return "unclassified"
    # Informational or explicitly ignored rows are not defects. Ambiguous and
    # unclassified rows are.
    if ambiguous or unclassified:
        return "partial"
    if any(d.type in _STATUS_DEFECT_DIAGNOSTICS for d in diagnostics):
        return "partial"
    # A complete result must expose at least one of the core PayPal payment
    # products (PayPal Checkout or goods and services).  The generic
    # ``other_commercial`` fallback alone does not imply full product coverage.
    required_core_ids = {"paypal_checkout", "goods_and_services"}
    generic_core_id = "other_commercial"
    core_ids = required_core_ids | {generic_core_id}
    core_rules = [r for r in rules if r.id in core_ids]
    found_required = any(r.id in required_core_ids for r in rules)
    if (
        found_required
        and core_rules
        and all(r.calculation_status == "calculable" for r in core_rules)
        and bool(fixed_schedules)
    ):
        return "complete"
    if any(r.id in required_core_ids for r in rules):
        return "partial"
    # Non-commercial markets may still be partial if any rule is incomplete.
    if any(r.calculation_status != "calculable" for r in rules if r.id not in core_ids):
        return "partial"
    return "partial"


def _rule_identity(rule: TransactionFeeRule) -> str:
    """Return a stable selector key for deduplicating equivalent rules.

    A rule selector is uniquely defined by product family, variant and
    applicable conditions. The localized ``label``, percentage and schedule
    references are intentionally excluded: the same selector with different
    fees is a conflict, not a different rule. Different rates must be
    expressed through different variants or conditions.
    """
    return json.dumps(
        {
            "id": rule.id,
            "variant_id": rule.variant_id,
            "conditions": {k: rule.conditions[k] for k in sorted(rule.conditions)} if rule.conditions else {},
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _rule_has_rate(rule: TransactionFeeRule) -> bool:
    """Return True if the rule carries a directly usable percentage."""
    return bool(
        rule.percentage is not None
        or (rule.rate_reference is not None and rule.rate_reference.resolved_rate is not None)
    )


def _is_reference_source(rule: TransactionFeeRule) -> bool:
    """Return True if the rule is a reference that resolves to another rule."""
    return bool(rule.rate_reference is not None and rule.rate_reference.resolved_rate is not None)


def _fee_components_for_rule(rule: TransactionFeeRule) -> list[FeeComponent]:
    """Build the explicit fee components for a transaction rule.

    The legacy ``percentage`` / ``fixed_fee_schedule`` / ``international_surcharge_schedule``
    fields are translated into a list of typed components so that a fee
    calculator can consume a single structure. Direct ``fixed_amount`` components
    already present on the rule are preserved.
    """
    components: list[FeeComponent] = []
    for component in rule.fee_components or []:
        if component.type == "fixed_amount":
            components.append(component)
    if rule.percentage is not None:
        components.append(FeeComponent(type="percentage", value=rule.percentage))
    if rule.fixed_fee_schedule is not None:
        components.append(FeeComponent(type="fixed_fee_schedule", schedule_id=rule.fixed_fee_schedule))
    if rule.international_surcharge_schedule is not None:
        components.append(
            FeeComponent(type="international_surcharge_schedule", schedule_id=rule.international_surcharge_schedule)
        )
    if rule.maximum_fee_schedule is not None:
        components.append(FeeComponent(type="maximum_fee_schedule", schedule_id=rule.maximum_fee_schedule))
    if (
        rule.rate_reference
        and rule.rate_reference.resolved_rate
        and rule.percentage is None
        and rule.rate_reference.resolved_rate.percentage
    ):
        components.append(FeeComponent(type="resolved_percentage", value=rule.rate_reference.resolved_rate.percentage))
    return components


def _derive_calculation_status(rule: TransactionFeeRule) -> str:
    """Return the calculability status for a rule.

    A rule is ``calculable`` when it has at least one usable fee component:
    a percentage, a resolved percentage, a fixed-fee schedule or an
    international-surcharge schedule.  Rules that are purely informational or
    have no usable component are marked ``incomplete``.
    """
    if _fee_components_for_rule(rule):
        return "calculable"
    if rule.rate_reference and rule.rate_reference.resolved_rate is None:
        return "reference_only"
    return "incomplete"


def _rule_fee_signature(rule: TransactionFeeRule) -> str:
    """Return a canonical signature of the complete fee definition carried by a rule.

    The signature includes percentage, fixed-fee schedule, international
    surcharge schedule, maximum-fee schedule, direct fixed-amount components and
    resolved reference percentage so that two rules with the same identity but
    different fee definitions are reported as conflicts.
    """
    return json.dumps(
        {
            "components": [
                _canonical_json(component.model_dump(mode="json")) for component in _fee_components_for_rule(rule)
            ],
            "maximum_fee_schedule": rule.maximum_fee_schedule,
            "calculation_status": _derive_calculation_status(rule),
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _deduplicate_rules(
    rules: list[TransactionFeeRule],
    diagnostics: list[Diagnostic] | None = None,
) -> list[TransactionFeeRule]:
    """Merge equivalent rules, preserving variants and preferring resolved references.

    Rules are equivalent when their product family, variant and conditions are
    identical. Within an equivalence group we prefer:
    1. a rule with a usable rate (or a resolved reference), and
    2. a rule that carries a reference (because it ties the source and target
       together), then
    3. the first rule in source order.

    If the same selector has different fee values, the group is a genuine
    conflict: a synthetic incomplete rule is emitted and a
    ``conflicting_rule_identity`` diagnostic is added.
    """
    groups: dict[str, list[tuple[int, TransactionFeeRule]]] = {}
    for idx, rule in enumerate(rules):
        groups.setdefault(_rule_identity(rule), []).append((idx, rule))

    selected: list[TransactionFeeRule] = []
    for group in groups.values():
        if len(group) > 1:
            signatures = {_rule_fee_signature(rule) for _, rule in group}
            if len(signatures) > 1:
                # Genuine conflict: the same selector resolves to different fees.
                if diagnostics is not None:
                    first = min(group, key=lambda item: item[0])[1]
                    diagnostics.append(
                        Diagnostic(
                            type="conflicting_rule_identity",
                            rule_id=first.id,
                            label=first.label,
                            values=[_rule_fee_signature(rule) for _, rule in group],
                            sources=[rule.source for _, rule in group if rule.source],
                        )
                    )
                # Emit an incomplete placeholder for the selector so the
                # conflict is visible but no authoritative value is exposed.
                representative = min(group, key=lambda item: item[0])[1]
                selected.append(
                    representative.model_copy(
                        update={
                            "percentage": None,
                            "fixed_fee_schedule": None,
                            "international_surcharge_schedule": None,
                            "maximum_fee_schedule": None,
                            "rate_reference": None,
                            "fee_components": [],
                            "calculation_status": "incomplete",
                        }
                    )
                )
                continue

        group.sort(
            key=lambda item: (
                0 if _rule_has_rate(item[1]) else 1,
                # Prefer a source row that carries a resolved reference so the
                # reference is preserved in the final output.
                0 if _is_reference_source(item[1]) else 1,
                item[0],
            )
        )
        selected.append(group[0][1])
    # Preserve original source order when possible.
    selected.sort(key=lambda r: next((i for i, rule in enumerate(rules) if rule is r), 0))
    return selected


def _rule_sort_key(rule: TransactionFeeRule) -> tuple[int, str | None, str, str | None]:
    order = {pid: idx for idx, pid in enumerate(_PRODUCT_ORDER)}
    return (
        order.get(rule.id, 999),
        rule.variant_id or "",
        _canonical_json(rule.conditions),
        rule.label or "",
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _resolve_rate_references(
    extracted_rules: list[_ExtractedRule],
    unresolved_rules: list[TransactionFeeRule | None],
    diagnostics: list[Diagnostic],
) -> None:
    """Resolve textual references against all collected rules."""
    for i, extracted in enumerate(extracted_rules):
        if not extracted.reference:
            continue
        rule = unresolved_rules[i]
        if rule is None:
            continue
        resolved, ambiguous = _resolve_reference(
            extracted.reference,
            unresolved_rules,
            source_variant_id=rule.variant_id,
            source_conditions=rule.conditions,
            source=rule.source,
        )
        if resolved:
            percentage = rule.percentage
            if resolved.percentage and percentage is None:
                percentage = resolved.percentage
            fixed_fee_schedule = rule.fixed_fee_schedule
            if resolved.fixed_fee_schedule is not None:
                fixed_fee_schedule = resolved.fixed_fee_schedule
            international_surcharge_schedule = rule.international_surcharge_schedule
            if resolved.international_surcharge_schedule is not None:
                international_surcharge_schedule = resolved.international_surcharge_schedule
            maximum_fee_schedule = rule.maximum_fee_schedule
            if resolved.maximum_fee_schedule is not None:
                maximum_fee_schedule = resolved.maximum_fee_schedule
            unresolved_rules[i] = rule.model_copy(
                update={
                    "rate_reference": RateReference(
                        reference=extracted.reference,
                        resolved_rate=resolved,
                        source=rule.source,
                    ),
                    "percentage": percentage,
                    "fixed_fee_schedule": fixed_fee_schedule,
                    "international_surcharge_schedule": international_surcharge_schedule,
                    "maximum_fee_schedule": maximum_fee_schedule,
                }
            )
        else:
            diagnostics.append(
                Diagnostic(
                    type="ambiguous_reference" if ambiguous else "unresolved_reference",
                    rule_id=rule.id,
                    label=extracted.label,
                    sources=[rule.source] if rule.source else [],
                )
            )
            if rule.percentage is None:
                unresolved_rules[i] = None


def _validate_top_level_schedule_references(
    unresolved_rules: list[TransactionFeeRule],
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
    diagnostics: list[Diagnostic],
) -> None:
    """Validate top-level schedule references and emit any remaining missing diagnostics."""
    for idx, rule in enumerate(unresolved_rules):
        updates: dict[str, Any] = {}
        if rule.fixed_fee_schedule and rule.fixed_fee_schedule not in fixed_schedules:
            diagnostics.append(
                Diagnostic(
                    type="missing_required_schedule",
                    rule_id=rule.id,
                    schedule_type="fixed_fee",
                    expected_schedule=rule.fixed_fee_schedule,
                    sources=[rule.source] if rule.source else [],
                )
            )
            updates["fixed_fee_schedule"] = None
        if (
            rule.international_surcharge_schedule
            and rule.international_surcharge_schedule not in international_schedules
        ):
            diagnostics.append(
                Diagnostic(
                    type="missing_required_schedule",
                    rule_id=rule.id,
                    schedule_type="international_surcharge",
                    expected_schedule=rule.international_surcharge_schedule,
                    sources=[rule.source] if rule.source else [],
                )
            )
            updates["international_surcharge_schedule"] = None
        if rule.maximum_fee_schedule and rule.maximum_fee_schedule not in maximum_fee_schedules:
            diagnostics.append(
                Diagnostic(
                    type="missing_required_schedule",
                    rule_id=rule.id,
                    schedule_type="maximum_fee",
                    expected_schedule=rule.maximum_fee_schedule,
                    sources=[rule.source] if rule.source else [],
                )
            )
            updates["maximum_fee_schedule"] = None
        if updates:
            unresolved_rules[idx] = rule.model_copy(update=updates)


def _validate_nested_schedule_references(
    unresolved_rules: list[TransactionFeeRule],
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
    diagnostics: list[Diagnostic],
) -> None:
    """Validate nested schedule references inside resolved rates."""
    for idx, rule in enumerate(unresolved_rules):
        if not (rule.rate_reference and rule.rate_reference.resolved_rate):
            continue
        resolved = rule.rate_reference.resolved_rate
        new_rate = resolved
        if resolved.fixed_fee_schedule and resolved.fixed_fee_schedule not in fixed_schedules:
            diagnostics.append(
                Diagnostic(
                    type="unresolved_nested_reference",
                    rule_id=rule.id,
                    schedule_type="fixed_fee",
                    expected_schedule=resolved.fixed_fee_schedule,
                    sources=[rule.source] if rule.source else [],
                )
            )
            new_rate = new_rate.model_copy(update={"fixed_fee_schedule": None})
        if (
            resolved.international_surcharge_schedule
            and resolved.international_surcharge_schedule not in international_schedules
        ):
            diagnostics.append(
                Diagnostic(
                    type="unresolved_nested_reference",
                    rule_id=rule.id,
                    schedule_type="international_surcharge",
                    expected_schedule=resolved.international_surcharge_schedule,
                    sources=[rule.source] if rule.source else [],
                )
            )
            new_rate = new_rate.model_copy(update={"international_surcharge_schedule": None})
        if resolved.maximum_fee_schedule and resolved.maximum_fee_schedule not in maximum_fee_schedules:
            diagnostics.append(
                Diagnostic(
                    type="unresolved_nested_reference",
                    rule_id=rule.id,
                    schedule_type="maximum_fee",
                    expected_schedule=resolved.maximum_fee_schedule,
                    sources=[rule.source] if rule.source else [],
                )
            )
            new_rate = new_rate.model_copy(update={"maximum_fee_schedule": None})
        if new_rate is not resolved:
            unresolved_rules[idx] = rule.model_copy(
                update={"rate_reference": rule.rate_reference.model_copy(update={"resolved_rate": new_rate})}
            )


def _materialize_fee_components(
    rules: list[TransactionFeeRule],
) -> list[TransactionFeeRule]:
    """Materialize fee components and calculability status for each rule."""
    return [
        rule.model_copy(
            update={
                "calculation_status": _derive_calculation_status(rule),
                "fee_components": _fee_components_for_rule(rule),
            }
        )
        for rule in rules
    ]


def _count_inherited_schedule_references(
    rules: list[TransactionFeeRule],
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
) -> int:
    """Return the number of rule schedule references that point to inherited schedules."""
    inherited = 0
    for rule in rules:
        for attr, schedules in (
            ("fixed_fee_schedule", fixed_schedules),
            ("international_surcharge_schedule", international_schedules),
            ("maximum_fee_schedule", maximum_fee_schedules),
        ):
            schedule_id = getattr(rule, attr)
            if schedule_id:
                schedule = schedules.get(schedule_id)
                if schedule and schedule.origin == "inherited":
                    inherited += 1
    return inherited


def _count_inherited_schedule_objects(
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
) -> int:
    """Return the number of schedule objects that are inherited."""
    return sum(
        1
        for schedules in (fixed_schedules, international_schedules, maximum_fee_schedules)
        for schedule in schedules.values()
        if schedule.origin == "inherited"
    )


def _count_direct_fixed_fees(rules: list[TransactionFeeRule]) -> int:
    """Return the number of direct fixed monetary fee rules."""
    fixed_schedule_count = sum(
        1
        for r in rules
        if r.percentage is None and r.fixed_fee_schedule is not None and r.international_surcharge_schedule is None
    )
    fee_component_count = sum(
        1 for r in rules for c in r.fee_components if c.type == "fixed_amount" and c.amount is not None
    )
    return fixed_schedule_count + fee_component_count


def _diagnostic_counts(diagnostics: list[Diagnostic]) -> dict[str, int]:
    """Return counts for diagnostic types used by coverage summary."""
    types = [d.type for d in diagnostics]
    return {
        "conflicts": types.count("conflicting_schedule_entry"),
        "missing_schedules": types.count("missing_required_schedule"),
        "unresolved_references": types.count("unresolved_reference"),
        "unresolved_nested_references": types.count("unresolved_nested_reference"),
        "unknown_apm": types.count("unknown_apm_method"),
    }


def _build_coverage_summary(
    rules: list[TransactionFeeRule],
    unresolved_rules: list[TransactionFeeRule],
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
    ignored_rows: list[UnclassifiedFeeRow],
    unclassified_rows: list[UnclassifiedFeeRow],
    ambiguous_rows: list[AmbiguousFeeRow],
    diagnostics: list[Diagnostic],
    extracted_rules: list[_ExtractedRule],
    numeric_fee_candidates: int = 0,
    unclassified_fee_candidates: int = 0,
) -> CoverageSummary:
    """Compute the classification coverage summary."""
    counts = _diagnostic_counts(diagnostics)
    reference_target_ids = {
        r.rate_reference.resolved_rate.rule_id
        for r in unresolved_rules
        if r.rate_reference and r.rate_reference.resolved_rate and r.rate_reference.resolved_rate.rule_id
    }
    extracted_apm = sum(
        len(e.conditions.get("payment_methods", []))
        for e in extracted_rules
        if e.product_id == "alternative_payment_methods"
    )

    inherited_references = _count_inherited_schedule_references(
        rules, fixed_schedules, international_schedules, maximum_fee_schedules
    )
    return CoverageSummary(
        transaction_rules=len(rules),
        calculable_rules=sum(1 for r in rules if r.calculation_status == "calculable"),
        non_calculable_rules=sum(1 for r in rules if r.calculation_status != "calculable"),
        direct_fixed_fees=_count_direct_fixed_fees(rules),
        fixed_fee_entries=sum(len(s.entries) for s in fixed_schedules.values()),
        international_surcharge_entries=sum(len(s.entries) for s in international_schedules.values()),
        maximum_fee_entries=sum(len(s.entries) for s in maximum_fee_schedules.values()),
        reference_sources=sum(1 for e in extracted_rules if e.reference),
        reference_targets=len(reference_target_ids),
        ignored=len(ignored_rows),
        unclassified=len(unclassified_rows),
        ambiguous=len(ambiguous_rows),
        conflicts=counts["conflicts"],
        missing_required_schedules=counts["missing_schedules"],
        inherited_schedules=inherited_references,
        inherited_schedule_objects=_count_inherited_schedule_objects(
            fixed_schedules, international_schedules, maximum_fee_schedules
        ),
        inherited_schedule_references=inherited_references,
        unresolved_references=counts["unresolved_references"],
        unresolved_nested_references=counts["unresolved_nested_references"],
        extracted_apm_methods=extracted_apm,
        unknown_apm_methods=counts["unknown_apm"],
        numeric_fee_candidates=numeric_fee_candidates,
        unclassified_fee_candidates=unclassified_fee_candidates,
    )


def classify_tables(tables: list[Table], source: Source | None = None) -> DerivedFeeResult:
    """Derive product-specific transaction fee rules from normalized tables."""
    table_categories: dict[int, str | None] = {id(table): _classify_table_category(table) for table in tables}
    fixed_schedules, international_schedules, maximum_fee_schedules, schedule_diagnostics = _collect_schedules(
        tables, source=source, table_categories=table_categories
    )
    diagnostics: list[Diagnostic] = list(schedule_diagnostics)

    extracted_rules: list[_ExtractedRule] = []
    unclassified_rows: list[UnclassifiedFeeRow] = []
    ambiguous_rows: list[AmbiguousFeeRow] = []
    ignored_rows: list[UnclassifiedFeeRow] = []
    total_numeric_fee_candidates = 0
    total_unclassified_fee_candidates = 0

    for table in tables:
        category = table_categories[id(table)]
        if category in _TABLE_CATEGORY_SCHEDULE or category in {
            "commercial_rate_table",
            "online_card_rate_table",
            "goods_and_services_rate_table",
            "donation_rate_table",
            "nonprofit_rate_table",
            "apm_rate_table",
            "pos_rate_table",
            "micropayment_rate_table",
            "withdrawals_rate_table",
            "other_fees_table",
        }:
            rules, uncls, ambig, ignored, numeric_candidates, unclassified_candidates = _extract_rules_from_rate_table(
                table,
                category,
                source,
                fixed_schedules,
                international_schedules,
                maximum_fee_schedules,
            )
            extracted_rules.extend(rules)
            unclassified_rows.extend(uncls)
            ambiguous_rows.extend(ambig)
            ignored_rows.extend(ignored)
            total_numeric_fee_candidates += numeric_candidates
            total_unclassified_fee_candidates += unclassified_candidates

    # First pass: build TransactionFeeRule objects without resolving references so
    # that all candidate target rules exist for the second pass.
    unresolved_rules: list[TransactionFeeRule | None] = []
    for extracted in extracted_rules:
        prov = _provenance(
            extracted.table,
            extracted.row,
            extracted.row_index,
            source,
            original_label=extracted.label,
        )
        unresolved_rules.append(
            TransactionFeeRule(
                id=extracted.product_id,
                variant_id=extracted.variant_id,
                label=extracted.label,
                percentage=extracted.percentage,
                fixed_fee_schedule=extracted.fixed_fee_schedule,
                international_surcharge_schedule=extracted.international_surcharge_schedule,
                maximum_fee_schedule=extracted.maximum_fee_schedule,
                conditions=extracted.conditions,
                rate_reference=None,
                source=prov,
                calculation_status="calculable",
                fee_components=list(extracted.fee_components),
            )
        )
        for unknown in extracted.unknown_apm_methods or []:
            # Only emit APM diagnostics for rows that are actually classified as
            # alternative payment methods; other product rows may split into
            # unrecognised tokens but are not APM source rows.
            if extracted.product_id != "alternative_payment_methods":
                continue
            diagnostics.append(
                Diagnostic(
                    type="unknown_apm_method",
                    rule_id=extracted.product_id,
                    payment_method=unknown,
                    label=extracted.label,
                    sources=[prov],
                )
            )

    # Direct fixed-fee tables (chargebacks, refunds, withdrawals, etc.) are not
    # tied to a rate table. Create a rule per schedule for any schedule that is
    # not already referenced by a product's rate-table rule.
    referenced_schedules = {r.fixed_fee_schedule for r in unresolved_rules if r is not None and r.fixed_fee_schedule}
    direct_fixed_rules = _create_direct_fixed_fee_rules(fixed_schedules, referenced_schedules)
    unresolved_rules.extend(direct_fixed_rules)

    # Recipient service fees (e.g. UK recipient surcharge) are independent
    # surcharge schedules and not part of the commercial international surcharge.
    # Expose them as a separate rule so the fee is selectable.
    referenced_intl_schedules = {
        r.international_surcharge_schedule
        for r in unresolved_rules
        if r is not None and r.international_surcharge_schedule
    }
    for schedule_id, schedule in international_schedules.items():
        if schedule_id in referenced_intl_schedules or not schedule.entries:
            continue
        if schedule_id == "recipient_service":
            provenance = schedule.sources[0] if schedule.sources else None
            unresolved_rules.append(
                TransactionFeeRule(
                    id="recipient_service",
                    variant_id="standard",
                    label=provenance.section_heading if provenance else None,
                    percentage=None,
                    fixed_fee_schedule=None,
                    international_surcharge_schedule=schedule_id,
                    conditions={"recipient_location": "GB"},
                    rate_reference=None,
                    source=provenance,
                    calculation_status="calculable",
                    fee_components=[FeeComponent(type="international_surcharge_schedule", schedule_id=schedule_id)],
                )
            )

    # Resolve references, create explicitly inherited schedules, and validate
    # schedule references for rules that are not attached to a rate table.
    _resolve_rate_references(extracted_rules, unresolved_rules, diagnostics)
    resolved_rules: list[TransactionFeeRule] = [r for r in unresolved_rules if r is not None]
    _resolve_schedule_inheritance(
        extracted_rules,
        resolved_rules,
        fixed_schedules,
        international_schedules,
        maximum_fee_schedules,
        diagnostics,
    )
    _validate_inheritance_priorities(
        fixed_schedules,
        international_schedules,
        maximum_fee_schedules,
        diagnostics,
    )
    _validate_top_level_schedule_references(
        resolved_rules, fixed_schedules, international_schedules, maximum_fee_schedules, diagnostics
    )
    _validate_nested_schedule_references(
        resolved_rules, fixed_schedules, international_schedules, maximum_fee_schedules, diagnostics
    )

    # Merge equivalent rules and preserve legitimate variants.
    transaction_rules = _deduplicate_rules(resolved_rules, diagnostics)
    transaction_rules.sort(key=_rule_sort_key)

    # Materialize fee components and calculability status for each rule.
    transaction_rules = _materialize_fee_components(transaction_rules)

    # Currency conversion.
    currency_conversion = None
    for table in tables:
        if table_categories[id(table)] == "currency_conversion_table":
            for row in table.rows:
                pct = _first_percentage(row)
                if pct:
                    currency_conversion = CurrencyConversion(spread_percentage=pct)
                    break
            if currency_conversion:
                break

    coverage = _build_coverage_summary(
        transaction_rules,
        resolved_rules,
        fixed_schedules,
        international_schedules,
        maximum_fee_schedules,
        ignored_rows,
        unclassified_rows,
        ambiguous_rows,
        diagnostics,
        extracted_rules,
        numeric_fee_candidates=total_numeric_fee_candidates,
        unclassified_fee_candidates=total_unclassified_fee_candidates,
    )

    status = _derive_status(
        transaction_rules,
        unclassified_rows,
        ambiguous_rows,
        ignored_rows,
        diagnostics,
        fixed_schedules,
        international_schedules,
        maximum_fee_schedules,
    )

    return DerivedFeeResult(
        status=status,
        transaction_fee_rules=transaction_rules,
        fixed_fee_schedules=fixed_schedules,
        international_surcharge_schedules=international_schedules,
        maximum_fee_schedules=maximum_fee_schedules,
        currency_conversion=currency_conversion,
        unclassified_fee_rows=unclassified_rows,
        ambiguous_rows=ambiguous_rows,
        ignored_rows=ignored_rows,
        diagnostics=diagnostics,
        coverage_summary=coverage,
    )


__all__ = [
    "CLASSIFIER_VERSION",
    "_CLASSIFIER_VERSION",
    "_PRODUCT_ALIASES",
    "_PRODUCT_ORDER",
    "_TABLE_CATEGORY_KEYWORDS",
    "_TABLE_NEGATIVE_SIGNALS",
    "_TABLE_CATEGORY_SCHEDULE",
    "_CANONICAL_AMOUNT_RE",
    "_PERCENTAGE_RE",
    "_PLUS_FIXED_RE",
    "_DIRECT_FIXED_FEE_PRODUCTS",
    "_CATEGORY_SPECIFIC_TABLES",
    "_DEFAULT_CURRENCY_BY_MARKET",
    "_FEE_HEADER_KEYWORDS",
    "_NON_FEE_HEADER_KEYWORDS",
    "_PRODUCT_CATEGORY_MAP",
    "_TABLE_CATEGORY_PRODUCT",
    "_FIXED_FEE_KEYWORDS",
    "_MIN_MAX_FEE_KEYWORDS",
    "_DIRECT_FIXED_FEE_KEYWORDS",
    "_INTERNATIONAL_SURCHARGE_KEYWORDS",
    "_LIMIT_OR_CAP_KEYWORDS",
    "_APM_METHOD_ALIASES",
    "_APM_SPECIAL_METHOD_IDS",
    "_APM_SORTED_ALIASES",
    "_APM_SEPARATOR_RE",
    "_APM_PUNCTUATION_RE",
    "_APM_EXAMPLE_PHRASE_RE",
    "_APM_HEADER_PHRASES",
    "_APM_HEADER_TOKENS",
    "_THAI_TOKENS",
    "_LATVIAN_TOKENS",
    "_LITHUANIAN_TOKENS",
    "_ESTONIAN_TOKENS",
    "_BANK_TOKENS",
    "_ONLINE_TOKENS",
    "_APM_METHOD_MATCHERS",
    "_APM_VARIANTS",
    "_ADVANCED_CARD_VARIANTS",
    "_QR_BELOW_THRESHOLD",
    "_QR_ABOVE_THRESHOLD",
    "_MICROPAYMENT_VARIANTS",
    "_PAYPAL_CHECKOUT_VARIANTS",
    "_OTHER_COMMERCIAL_VARIANTS",
    "_POS_VARIANTS",
    "_DONATIONS_VARIANTS",
    "_APM_VARIANT_ONLY_VARIANTS",
    "_INVOICE_VARIANTS",
    "_NONPROFIT_VARIANTS",
    "_PAY_LATER_VARIANTS",
    "_QR_VARIANTS",
    "_DISPUTE_VARIANTS",
    "_WITHDRAWAL_VARIANTS",
    "_SEPA_DIRECT_DEBIT_VARIANTS",
    "_FRAUD_PROTECTION_VARIANTS",
    "_RECORDS_REQUEST_VARIANTS",
    "_CARD_VERIFICATION_VARIANTS",
    "_VARIANT_RULES_BY_PRODUCT",
    "_BASE_VARIANTS_BY_PRODUCT",
    "_ADVANCED_CARD_SCHEDULE_KEYWORDS",
    "_SCHEDULE_NAME_FROM_TABLE_MAPPING",
    "_REGION_EXACT",
    "RegionPattern",
    "_REGION_PATTERNS",
    "_REFERENCE_SCHEDULE_KEYWORDS",
    "_REFERENCE_SUFFIX_TO_PRODUCT",
    "_REFERENCE_PRODUCT_SUFFIX",
    "_FIXED_FEE_SCHEDULE_FOR",
    "_FIXED_FEE_INHERITANCE",
    "_FIXED_FEE_SCHEDULE_FALLBACK",
    "_INTERNATIONAL_SURCHARGE_SCHEDULE_FOR",
    "_INTERNATIONAL_SURCHARGE_INHERITANCE",
    "_INTERNATIONAL_SURCHARGE_SCHEDULE_FALLBACK",
    "_MAXIMUM_FEE_SCHEDULE_FALLBACK",
    "_DIRECT_FIXED_FEE_SCHEDULE_PRODUCTS",
    "_STATUS_DEFECT_DIAGNOSTICS",
    "_norm",
    "_NORMALIZED_PRODUCT_ALIASES",
    "_keyword_in_text",
    "_table_text",
    "_table_context_original",
    "_row_cells_text",
    "_text_indicates_percentage",
    "_token_text_indicates_percentage",
    "_first_percentage",
    "_first_money",
    "_cell_money",
    "_row_has_percentage",
    "_row_label",
    "_row_fee_cell",
    "_infer_currency_for_row",
    "_parse_canonical_amount",
    "_cell_looks_like_fee_cell",
    "_last_non_empty_cell_text",
    "_has_likely_numeric_fee_candidate",
    "_first_variant_match",
    "_all_variant_matches",
    "_market_code_from_url",
    "_parse_rate_expression",
    "_tokenize_apm_label",
    "_extract_apm_methods",
    "_is_apm_special_label",
    "_is_international_label",
    "_is_domestic_label",
    "_score_label_against_product",
    "_classify_product",
    "_is_currency_conversion_text",
    "_is_maximum_fee_table",
    "_is_withdrawals_rate_table",
    "_table_has_fixed_fee_rate",
    "_classify_table_category",
    "_is_limit_or_cap_row",
    "_select_category_from_scores",
    "_top_category_candidates",
    "_filter_category_negative_signals",
    "_fallback_category_candidate",
    "_classify_table_by_row_labels",
    "_classify_product_or_apm",
    "_resolve_ambiguous_product",
    "_resolve_missing_product",
    "_resolve_product_id",
    "_is_charity_label",
    "_is_generic_other_commercial_label",
    "_is_generic_apm_label",
    "_extract_country_group_condition",
    "_pricing_plan_for_label",
    "_card_payment_methods_from_label",
    "_service_for_donation_label",
    "_conditions_for_apm",
    "_conditions_for_donations",
    "_service_for_advanced_card",
    "_conditions_for_advanced_card",
    "_conditions_for_pos",
    "_conditions_for_paypal_checkout",
    "_service_for_other_commercial_ach",
    "_conditions_for_other_commercial",
    "_transaction_region_for_variant",
    "_conditions_for_row",
    "_maximum_fee_schedule_for_conditions",
    "_extract_amount_condition",
    "_is_sending_donation_table",
    "_applicable_variants_for_table",
    "_variant_for_apm",
    "_variant_for_advanced_card",
    "_variant_for_qr_code",
    "_variant_for_micropayments",
    "_variant_for_paypal_checkout",
    "_variant_for_other_commercial",
    "_variant_for_pos_transactions",
    "_variant_for_donations",
    "_variant_for_nonprofit",
    "_variant_for_invoice_pay_later",
    "_variant_for_pay_later_consumer",
    "_variant_for_withdrawals",
    "_variant_for_sepa_direct_debit",
    "_variant_for_fraud_protection",
    "_variant_for_records_request",
    "_variant_for_card_verification",
    "_VARIANT_DISPATCH",
    "_variant_id_for_row",
    "_schedule_name_from_table",
    "_signature_key",
    "_schedule_signature_for_row",
    "_schedule_suffix_from_signature",
    "_schedule_id",
    "_select_schedule_id",
    "_signature_from_conditions",
    "_conditions_from_schedule_id",
    "_extract_fixed_fee_schedule",
    "_extract_maximum_fee_schedule",
    "_max_fee_base_id",
    "_max_fee_money",
    "_max_fee_schedule_sources",
    "_merge_max_fee_schedule",
    "_extract_international_surcharge_schedule",
    "_matches_region_pattern",
    "_normalize_region",
    "_fixed_fee_schedule_for",
    "_international_surcharge_schedule_for",
    "_schedule_ids_for_table",
    "_merge_fixed_like_schedules",
    "_merge_international_surcharge_schedules",
    "_collect_fixed_fee_table",
    "_collect_international_surcharge_table",
    "_collect_maximum_fee_table",
    "_collect_schedules",
    "_source_schedule_id",
    "_inheritance_map_for",
    "_source_text_of",
    "_source_text_evidence",
    "_schedule_heading_text",
    "_table_context_evidence",
    "_inheritance_evidence",
    "_create_inherited_schedule",
    "_resolve_schedule",
    "_resolve_schedule_inheritance",
    "_product_family_for_schedule_id",
    "_validate_inheritance_priorities",
    "_merge_provenance_sources",
    "_detect_reference",
    "_reference_product_id",
    "_as_list",
    "_condition_score",
    "_conditions_match_for_reference",
    "_reference_target_id",
    "_reference_candidates",
    "_resolved_source_conditions",
    "_condition_matched_candidates",
    "_build_resolved_rate",
    "_resolve_reference",
    "_provenance",
    "_extract_direct_fixed_amounts",
    "_ExtractedRule",
    "_ignored_rate_row",
    "_build_direct_fixed_rules",
    "_build_standard_rate_rule",
    "_handle_unusable_rate_row",
    "_extract_rules_from_rate_table",
    "_direct_fixed_product_variant",
    "_create_direct_fixed_fee_rules",
    "logger",
    "_derive_status",
    "_rule_identity",
    "_rule_has_rate",
    "_is_reference_source",
    "_fee_components_for_rule",
    "_derive_calculation_status",
    "_rule_fee_signature",
    "_deduplicate_rules",
    "_rule_sort_key",
    "_canonical_json",
    "_resolve_rate_references",
    "_validate_top_level_schedule_references",
    "_validate_nested_schedule_references",
    "_materialize_fee_components",
    "_count_inherited_schedule_references",
    "_count_inherited_schedule_objects",
    "_count_direct_fixed_fees",
    "_diagnostic_counts",
    "_build_coverage_summary",
    "classify_tables",
]
