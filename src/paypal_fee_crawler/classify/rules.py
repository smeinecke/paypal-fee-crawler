from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..models import (
    AmbiguousFeeRow,
    FeeComponent,
    FixedFeeSchedule,
    InternationalSurchargeSchedule,
    Row,
    Source,
    Table,
    TransactionFeeRule,
    UnclassifiedFeeRow,
)
from .apm import _extract_apm_methods
from .conditions import _conditions_for_row, _maximum_fee_schedule_for_conditions
from .patterns import (
    _APM_VARIANT_ONLY_VARIANTS,
    _CATEGORY_SPECIFIC_TABLES,
    _DIRECT_FIXED_FEE_PRODUCTS,
    _DIRECT_FIXED_FEE_SCHEDULE_PRODUCTS,
    _FIXED_FEE_SCHEDULE_FALLBACK,
    _FIXED_FEE_SCHEDULE_FOR,
    _INTERNATIONAL_SURCHARGE_SCHEDULE_FALLBACK,
    _INTERNATIONAL_SURCHARGE_SCHEDULE_FOR,
    _MAXIMUM_FEE_SCHEDULE_FALLBACK,
    _TABLE_CATEGORY_PRODUCT,
)
from .products import _is_limit_or_cap_row, _resolve_product_id
from .references import _provenance
from .schedules import (
    _conditions_from_schedule_id,
    _fixed_fee_schedule_for,
    _international_surcharge_schedule_for,
    _select_schedule_id,
    _signature_from_conditions,
)
from .text_utils import (
    _has_likely_numeric_fee_candidate,
    _infer_currency_for_row,
    _keyword_match,
    _norm,
    _parse_canonical_amount,
    _parse_rate_expression,
    _row_cells_text,
    _row_fee_cell,
    _row_label,
)
from .variants import _variant_id_for_row

logger = logging.getLogger(__name__)


def _extract_direct_fixed_amounts(
    row: Row,
    product_id: str,
    table: Table,
    source: Source | None,
) -> list[tuple[str, str, str]]:
    """Return direct fixed-fee amounts for a row as (amount, currency, variant_id)."""
    fee_text = _row_fee_cell(row)
    if not fee_text:
        return []
    label = _row_label(row)
    inferred_currency = _infer_currency_for_row(row, table, source)
    amounts: list[tuple[str, str, str]] = []
    # Match numeric amounts with an optional trailing ISO currency code.  The
    # pattern supports both thousands-separated and decimal-comma forms so a
    # value like "50,000.00 IDR" is parsed as a single amount.
    pattern = re.compile(
        r"(?P<operator>[+\-])?(?P<amount>\d{1,3}(?:[.,]\d{3})+(?:[.,]\d+)?|\d+(?:[.,]\d+)?)\s*(?P<currency>[A-Za-z]{3})?"
    )
    matches = list(pattern.finditer(fee_text))
    for idx, match in enumerate(matches):
        amount_raw = match.group("amount")
        if not amount_raw:
            continue
        amount = _parse_canonical_amount(amount_raw)
        if amount is None:
            continue
        currency = (match.group("currency") or "").upper() or inferred_currency
        if not currency:
            continue
        next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(fee_text)
        segment = fee_text[match.start() : next_start]
        variant_id = _variant_id_for_row(product_id, label, [], table, fee_text=segment)
        if variant_id is None:
            variant_id = _variant_id_for_row(product_id, label, [], table, fee_text=fee_text) or "standard"
        amounts.append((amount, currency, variant_id))
    # Fallback: use the tokenizer's money/number tokens when regex failed to
    # produce a parseable result.
    if not amounts:
        for cell in reversed(row.cells):
            for token in cell.tokens:
                if token.kind == "money" and token.amount and token.currency:
                    amount = _parse_canonical_amount(token.amount)
                    if amount is None:
                        continue
                    variant_id = _variant_id_for_row(product_id, label, [], table, fee_text=fee_text) or "standard"
                    amounts.append((amount, token.currency, variant_id))
                elif token.kind == "number" and token.value and inferred_currency:
                    amount = _parse_canonical_amount(token.value)
                    if amount is None:
                        continue
                    variant_id = _variant_id_for_row(product_id, label, [], table, fee_text=fee_text) or "standard"
                    amounts.append((amount, inferred_currency, variant_id))
    # Explicit zero-fee rows (e.g. "No Fee") for direct fixed-fee products are
    # still fee information and should be represented as a 0 fixed amount.
    if not amounts:
        fee_norm = _norm(fee_text)
        if _keyword_match(fee_norm, ("no fee", "free", "gratis", "kostenlos", "0.00", "0,00"), word_boundary=False):
            currency = _infer_currency_for_row(row, table, source)
            if currency:
                variant_id = _variant_id_for_row(product_id, label, [], table, fee_text=fee_text) or "standard"
                amounts.append(("0", currency, variant_id))
    return amounts


@dataclass(frozen=True)
class _ExtractedRule:
    product_id: str
    variant_id: str | None
    label: str
    percentage: str | None
    fixed_fee_schedule: str | None
    international_surcharge_schedule: str | None
    maximum_fee_schedule: str | None
    conditions: dict[str, Any]
    table: Table
    row: Row
    row_index: int
    reference: str | None = None
    unknown_apm_methods: list[str] = field(default_factory=list)
    fee_components: list[FeeComponent] = field(default_factory=list)
    # Source schedule ids that an intended schedule may be inherited from when the
    # product-specific schedule is not directly present.
    fixed_fee_schedule_source: str | None = None
    international_surcharge_schedule_source: str | None = None
    maximum_fee_schedule_source: str | None = None
    fixed_expr: str | None = None
    table_category: str | None = None


def _ignored_rate_row(
    row: Row,
    row_index: int,
    label: str,
    table: Table,
    source: Source | None,
    reason: str,
) -> UnclassifiedFeeRow:
    """Return an ignored fee row with the given reason."""
    return UnclassifiedFeeRow(
        normalized_cells=_row_cells_text(row),
        original_label=label,
        source=_provenance(table, row, row_index, source, original_label=label),
        reason=reason,
    )


def _build_direct_fixed_rules(
    row: Row,
    row_index: int,
    product_id: str,
    fallback_variant_id: str,
    label: str,
    methods: list[str],
    table: Table,
    source: Source | None,
    direct_amounts: list[tuple[str, str, str]],
) -> list[_ExtractedRule]:
    """Build direct fixed-fee rules from a row's parsed amounts.

    When multiple currencies apply to the same variant, a ``fee_currency``
    condition is added so the rules have distinct identities.
    """
    rules: list[_ExtractedRule] = []
    variant_currencies: dict[str, set[str]] = {}
    for _, currency, amount_variant_id in direct_amounts:
        variant_currencies.setdefault(amount_variant_id, set()).add(currency)

    for amount, currency, amount_variant_id in direct_amounts:
        if amount_variant_id == "standard" and fallback_variant_id not in (None, "standard"):
            amount_variant_id = fallback_variant_id
        amount_conditions = _conditions_for_row(product_id, amount_variant_id, label, methods=methods, table=table)
        if len(variant_currencies.get(amount_variant_id, set())) > 1:
            amount_conditions["fee_currency"] = currency
        rules.append(
            _ExtractedRule(
                product_id=product_id,
                variant_id=amount_variant_id,
                label=label,
                percentage=None,
                fixed_fee_schedule=None,
                international_surcharge_schedule=None,
                maximum_fee_schedule=None,
                conditions=amount_conditions,
                table=table,
                row=row,
                row_index=row_index,
                reference=None,
                unknown_apm_methods=[],
                fee_components=[FeeComponent(type="fixed_amount", amount=amount, currency=currency)],
            )
        )
    return rules


def _build_standard_rate_rule(
    row: Row,
    row_index: int,
    product_id: str,
    variant_id: str,
    label: str,
    pct: str | None,
    reference: str | None,
    methods: list[str],
    unknown_methods: list[str],
    conditions: dict[str, Any],
    table: Table,
    table_category: str,
    fixed_expr: str | None,
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
) -> _ExtractedRule:
    """Build a standard percentage/reference rule with schedule references."""
    fixed_schedule: str | None = None
    fixed_schedule_source: str | None = None
    if fixed_expr and product_id != "withdrawals":
        fixed_base = _fixed_fee_schedule_for(product_id, variant_id)
        if fixed_base:
            sig = _signature_from_conditions(conditions, fixed_base, product_id)
            fixed_schedule, fixed_schedule_source = _select_schedule_id(
                fixed_base,
                sig,
                fixed_schedules,
                _FIXED_FEE_SCHEDULE_FALLBACK.get(product_id, ()),
                product_base=_FIXED_FEE_SCHEDULE_FOR.get(product_id),
            )

    intl_schedule: str | None = None
    intl_schedule_source: str | None = None
    intl_base = _international_surcharge_schedule_for(product_id, variant_id)
    if intl_base:
        sig = _signature_from_conditions(conditions, intl_base, product_id)
        intl_schedule, intl_schedule_source = _select_schedule_id(
            intl_base,
            sig,
            international_schedules,
            _INTERNATIONAL_SURCHARGE_SCHEDULE_FALLBACK.get(product_id, ()),
            product_base=_INTERNATIONAL_SURCHARGE_SCHEDULE_FOR.get(product_id),
        )

    maximum_fee_schedule: str | None = None
    maximum_fee_schedule_source: str | None = None
    if product_id == "withdrawals" and table_category == "withdrawals_rate_table" and pct is not None:
        max_base = _maximum_fee_schedule_for_conditions(conditions)
        if max_base:
            sig = _signature_from_conditions(conditions, max_base, product_id)
            maximum_fee_schedule, maximum_fee_schedule_source = _select_schedule_id(
                max_base,
                sig,
                maximum_fee_schedules,
                _MAXIMUM_FEE_SCHEDULE_FALLBACK.get(max_base, ()),
                product_base=max_base,
            )

    # Listed-campaign donation campaigns are free.
    if variant_id == "campaign_unlisted":
        pct = "0"
        fixed_schedule = None
        intl_schedule = None

    return _ExtractedRule(
        product_id=product_id,
        variant_id=variant_id,
        label=label,
        percentage=pct,
        fixed_fee_schedule=fixed_schedule,
        international_surcharge_schedule=intl_schedule,
        maximum_fee_schedule=maximum_fee_schedule,
        conditions=conditions,
        table=table,
        row=row,
        row_index=row_index,
        reference=reference,
        unknown_apm_methods=unknown_methods,
        fixed_fee_schedule_source=fixed_schedule_source,
        international_surcharge_schedule_source=intl_schedule_source,
        maximum_fee_schedule_source=maximum_fee_schedule_source,
        fixed_expr=fixed_expr,
        table_category=table_category,
    )


def _handle_unusable_rate_row(
    row: Row,
    row_index: int,
    label: str,
    product_id: str,
    table: Table,
    source: Source | None,
    unclassified: list[UnclassifiedFeeRow],
    ignored: list[UnclassifiedFeeRow],
) -> None:
    """Store a row without a usable percentage/reference in the right bucket."""
    if _has_likely_numeric_fee_candidate(row, table):
        unclassified.append(_ignored_rate_row(row, row_index, label, table, source, "unsupported_fee_shape"))
    else:
        ignored.append(_ignored_rate_row(row, row_index, label, table, source, "no rate or reference"))


def _extract_rules_from_rate_table(
    table: Table,
    table_category: str,
    source: Source | None,
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
) -> tuple[list[_ExtractedRule], list[UnclassifiedFeeRow], list[AmbiguousFeeRow], list[UnclassifiedFeeRow], int, int]:
    rules: list[_ExtractedRule] = []
    unclassified: list[UnclassifiedFeeRow] = []
    ambiguous: list[AmbiguousFeeRow] = []
    ignored: list[UnclassifiedFeeRow] = []
    numeric_fee_candidates = 0
    unclassified_fee_candidates = 0

    default_product = _TABLE_CATEGORY_PRODUCT.get(table_category)
    force_default_product = table_category in _CATEGORY_SPECIFIC_TABLES

    for idx, row in enumerate(table.rows):
        label = _row_label(row)
        fee_text = _row_fee_cell(row)
        is_limit_or_cap = _is_limit_or_cap_row(label, fee_text)
        if _has_likely_numeric_fee_candidate(row, table) and not is_limit_or_cap:
            numeric_fee_candidates += 1
        if not label:
            ignored.append(_ignored_rate_row(row, idx, label, table, source, "empty label"))
            continue

        product_id, reference = _resolve_product_id(
            label,
            row,
            idx,
            table,
            source,
            default_product,
            force_default_product,
            unclassified,
            ambiguous,
            ignored,
        )
        if product_id is None:
            continue

        if is_limit_or_cap:
            ignored.append(_ignored_rate_row(row, idx, label, table, source, "limit or cap"))
            continue

        pct, fixed_expr = _parse_rate_expression(fee_text)
        methods, unknown_methods = _extract_apm_methods(label)
        variant_id = _variant_id_for_row(product_id, label, methods, table, fee_text=fee_text) or "standard"

        if product_id == "alternative_payment_methods" and variant_id in _APM_VARIANT_ONLY_VARIANTS:
            unknown_methods = []

        if pct is None and reference is None and product_id in _DIRECT_FIXED_FEE_PRODUCTS:
            direct_amounts = _extract_direct_fixed_amounts(row, product_id, table, source)
            if direct_amounts:
                rules.extend(
                    _build_direct_fixed_rules(
                        row,
                        idx,
                        product_id,
                        variant_id,
                        label,
                        methods,
                        table,
                        source,
                        direct_amounts,
                    )
                )
                continue

        conditions = _conditions_for_row(product_id, variant_id, label, methods=methods, table=table)

        if pct is None and reference is None:
            _handle_unusable_rate_row(row, idx, label, product_id, table, source, unclassified, ignored)
            continue

        rules.append(
            _build_standard_rate_rule(
                row,
                idx,
                product_id,
                variant_id,
                label,
                pct,
                reference,
                methods,
                unknown_methods,
                conditions,
                table,
                table_category,
                fixed_expr,
                fixed_schedules,
                international_schedules,
                maximum_fee_schedules,
            )
        )
    unclassified_fee_candidates = sum(
        1 for r in unclassified if r.reason in ("unclassified_fee_candidate", "unsupported_fee_shape")
    )
    return rules, unclassified, ambiguous, ignored, numeric_fee_candidates, unclassified_fee_candidates


def _direct_fixed_product_variant(schedule_id: str, direct_bases: set[str]) -> tuple[str | None, str | None]:
    """Return the product id and variant id for a direct fixed-fee schedule id."""
    base_part = schedule_id.split("__", 1)[0]
    if base_part in direct_bases:
        return base_part, "standard"
    for base in sorted(direct_bases, key=len, reverse=True):
        if base_part.startswith(base + "_"):
            variant = base_part[len(base) + 1 :]
            return base, variant
    return None, None


def _create_direct_fixed_fee_rules(
    fixed_schedules: dict[str, FixedFeeSchedule],
    referenced_schedules: set[str],
) -> list[TransactionFeeRule]:
    """Create calculable rules for schedules that are not referenced by a rate table.

    Chargebacks, disputes, refunds, card verification and withdrawals are
    expressed as direct monetary amounts per currency. They do not have a
    percentage-based rate table, so they cannot be represented by the normal
    rate-table extraction path.
    """
    rules: list[TransactionFeeRule] = []
    direct_bases = set(_DIRECT_FIXED_FEE_SCHEDULE_PRODUCTS.values())
    for schedule_id, schedule in fixed_schedules.items():
        if schedule_id in referenced_schedules:
            continue
        base_part = schedule_id.split("__", 1)[0]
        product_id, variant_id = _direct_fixed_product_variant(base_part, direct_bases)
        if not product_id:
            continue
        if not schedule.entries:
            continue
        provenance = schedule.sources[0] if schedule.sources else None
        conditions = _conditions_from_schedule_id(schedule_id)
        if product_id == "withdrawals" and variant_id and variant_id != "standard":
            conditions["withdrawal_method"] = variant_id
        elif product_id == "disputes" and variant_id and variant_id != "standard":
            conditions["volume_status"] = variant_id
        rules.append(
            TransactionFeeRule(
                id=product_id,
                variant_id=variant_id or "standard",
                label=provenance.section_heading if provenance else None,
                percentage=None,
                fixed_fee_schedule=schedule_id,
                international_surcharge_schedule=None,
                conditions=conditions,
                rate_reference=None,
                source=provenance,
                calculation_status="calculable",
                fee_components=[],
            )
        )
    return rules
