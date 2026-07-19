from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Any

from ..models import (
    Diagnostic,
    FixedFeeSchedule,
    InternationalSurchargeSchedule,
    InternationalSurchargeScheduleEntry,
    Provenance,
    Row,
    Source,
    Table,
    TransactionFeeRule,
)
from ..normalize import normalize_decimal_string
from ..pricing_tokens import CURRENCY_CODES

if TYPE_CHECKING:
    from .rules import _ExtractedRule
from .conditions import _extract_country_group_condition
from .patterns import (
    _ADVANCED_CARD_SCHEDULE_KEYWORDS,
    _BASE_VARIANTS_BY_PRODUCT,
    _CLASSIFIER_VERSION,
    _DIRECT_FIXED_FEE_SCHEDULE_PRODUCTS,
    _FIXED_FEE_INHERITANCE,
    _FIXED_FEE_SCHEDULE_FOR,
    _INTERNATIONAL_SURCHARGE_INHERITANCE,
    _INTERNATIONAL_SURCHARGE_SCHEDULE_FOR,
    _REGION_EXACT,
    _REGION_PATTERNS,
    _SCHEDULE_NAME_FROM_TABLE_MAPPING,
    RegionPattern,
)
from .products import _classify_table_category, _is_limit_or_cap_row
from .text_utils import (
    _cell_money,
    _first_percentage,
    _keyword_match,
    _norm,
    _row_fee_cell,
    _row_label,
    _table_context_original,
    _table_text,
)
from .variants import _applicable_variants_for_table

logger = logging.getLogger(__name__)


def _schedule_name_from_table(table: Table, default: str | None) -> str:
    text = _table_text(table)
    if _keyword_match(text, _ADVANCED_CARD_SCHEDULE_KEYWORDS, word_boundary=False):
        return "advanced_card_payments"
    mapping = _SCHEDULE_NAME_FROM_TABLE_MAPPING
    for name, keywords in mapping.items():
        if _keyword_match(text, keywords, word_boundary=False):
            return name
    return default or "commercial"


def _signature_key(signature: dict[str, Any]) -> frozenset[tuple[str, Any]]:
    """Return a hashable representation of a schedule signature for grouping."""
    items: list[tuple[str, Any]] = []
    for k, v in signature.items():
        if isinstance(v, list):
            v = tuple(sorted(v))
        elif isinstance(v, dict):
            v = tuple(sorted((kk, tuple(sorted(vv)) if isinstance(vv, list) else vv) for kk, vv in v.items()))
        items.append((k, v))
    return frozenset(items)


def _schedule_signature_for_row(
    row: Row,
    base_name: str,
    table_text: str = "",
    use_row_label: bool = True,
) -> dict[str, Any]:
    """Return applicability dimensions encoded in a schedule table row and context.

    Schedule tables usually contain currency names as row labels, so market/amount
    applicability is read from the table heading/caption by default.  Rate-table
    rows (``use_row_label=True``) may encode applicability in the label itself.
    """
    label = _row_label(row) if use_row_label else ""
    combined_text = f"{label} {table_text}".strip()
    sig: dict[str, Any] = {}
    market_condition = _extract_country_group_condition(label) or _extract_country_group_condition(table_text)
    if market_condition:
        sig["applies_to_markets"] = market_condition["applies_to_markets"]
    # QR threshold variants are derived from the table caption via
    # _applicable_variants_for_table, so the amount_tier is encoded in the
    # base schedule id and should not be duplicated in the signature.
    if base_name == "advanced_card_payments" and "interchange" in _norm(combined_text):
        if "plus plus" in _norm(combined_text) or "++" in combined_text:
            sig["pricing_plan"] = "interchange_plus_plus"
        else:
            sig["pricing_plan"] = "interchange_plus"
    return sig


def _schedule_suffix_from_signature(signature: dict[str, Any]) -> str:
    """Canonical string suffix for a schedule applicability signature."""
    parts: list[str] = []
    for key in sorted(signature):
        value = signature[key]
        if key == "applies_to_markets":
            if isinstance(value, list):
                if value == ["all_other_markets"] or not value:
                    continue
                parts.append("applies_to_markets=" + "_".join(sorted(value)).lower())
            else:
                parts.append(f"applies_to_markets={_norm(str(value))}")
        else:
            parts.append(f"{key}={_norm(str(value))}")
    return "__".join(parts)


def _schedule_id(base_id: str, signature: dict[str, Any]) -> str:
    """Build a schedule id from a base id and an applicability signature."""
    suffix = _schedule_suffix_from_signature(signature)
    if not suffix:
        return base_id
    return f"{base_id}__{suffix}"


def _select_schedule_id(
    base_id: str,
    signature: dict[str, Any],
    available: dict[str, Any],
    fallback_bases: tuple[str, ...] = (),
    product_base: str | None = None,
) -> tuple[str | None, str | None]:
    """Return the schedule id a rule should reference and an optional inheritance source.

    Resolution priority is:

    1. Exact variant-specific schedule (with the full signature suffix).
    2. Direct product-family base schedule (with the same suffix or as a base).
    3. Explicitly proven cross-product inheritance from ``fallback_bases``.
    4. ``None`` if no usable schedule exists.

    The direct product-family base is only consulted when it differs from the
    variant-specific ``base_id`` so that variant-specific schedules are tried
    first, but a generic product schedule always wins over cross-product
    inheritance.
    """
    suffix = _schedule_suffix_from_signature(signature)
    candidates: list[str] = []
    if suffix:
        candidates.append(f"{base_id}__{suffix}")
    candidates.append(base_id)
    if product_base and product_base != base_id:
        if suffix:
            candidates.append(f"{product_base}__{suffix}")
        candidates.append(product_base)
    intended = candidates[0] if candidates else None
    for candidate in candidates:
        if candidate in available:
            return candidate, None
    # No existing schedule. Report the intended id and the first available
    # fallback source schedule so that explicit inheritance can be attempted.
    if fallback_bases:
        for fallback in fallback_bases:
            if suffix:
                suffixed = f"{fallback}__{suffix}"
                if suffixed in available:
                    return intended, suffixed
                if fallback in available:
                    return intended, fallback
            elif fallback in available:
                return intended, fallback
    return None, None


def _signature_from_conditions(conditions: dict[str, Any], base_id: str, product_id: str) -> dict[str, Any]:
    """Build a schedule applicability signature from rule conditions."""
    sig: dict[str, Any] = {}
    markets = conditions.get("applies_to_markets")
    if markets:
        sig["applies_to_markets"] = markets
    amount = conditions.get("amount")
    if isinstance(amount, dict):
        op = amount.get("operator")
        if op in {"lt", "lte", "under", "below", "less than", "up to"}:
            sig["amount_tier"] = "below_threshold"
        elif op in {"gt", "gte", "above", "over", "greater than", "at least", "mindestens"}:
            sig["amount_tier"] = "above_threshold"
    pricing_plan = conditions.get("pricing_plan")
    if pricing_plan and not base_id.endswith(str(pricing_plan)):
        sig["pricing_plan"] = pricing_plan
    return sig


def _conditions_from_schedule_id(schedule_id: str) -> dict[str, Any]:
    """Parse an applicability suffix from a schedule id into conditions."""
    conditions: dict[str, Any] = {}
    if "__" not in schedule_id:
        return conditions
    _, suffix = schedule_id.split("__", 1)
    for part in suffix.split("__"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if key == "applies_to_markets":
            if value == "all_other_markets":
                conditions[key] = ["all_other_markets"]
            else:
                conditions[key] = sorted(value.split("_"))
        elif key == "amount_tier" or key == "pricing_plan" or key == "transaction_region":
            conditions[key] = value
    return conditions


def _extract_fixed_fee_schedule(
    table: Table, base_name: str, source: Source | None = None
) -> dict[str, FixedFeeSchedule]:
    """Extract fixed-fee schedules grouped by row applicability signature."""
    table_text = _table_context_original(table)
    groups: dict[frozenset[tuple[str, Any]], dict[str, str]] = {}
    group_keys: dict[frozenset[tuple[str, Any]], dict[str, Any]] = {}
    # Determine which row cells are charge/fee columns. The first column is
    # normally the label/currency name. Any cells beyond the header count are
    # stray cells (e.g. footnotes or converted amounts) and must be ignored.
    header_count = len(table.headers)
    if header_count:
        charge_indices = [i for i in range(1, header_count) if table.headers[i].text.strip()]
    else:
        charge_indices: list[int] = []
    for row in table.rows:
        if _is_limit_or_cap_row(_row_label(row), _row_fee_cell(row)):
            continue
        signature = _schedule_signature_for_row(row, base_name, table_text, use_row_label=False)
        key = _signature_key(signature)
        group_keys.setdefault(key, signature)
        amounts = groups.setdefault(key, {})
        cells = row.cells
        iterable = (
            (cells[i] for i in charge_indices if i < len(cells)) if charge_indices else cells[1:]
        )  # skip the label cell
        for cell in iterable:
            if not cell.text.strip():
                continue
            money = _cell_money(cell)
            if money:
                amounts[money[0]] = money[1]
                continue
            # Some cells contain templated placeholders like {{...}}; skip them.
            if "{{" in cell.text:
                continue
            # Fallback: parse an explicit "amount CUR" text.
            parts = cell.text.strip().split()
            if len(parts) >= 2 and parts[-1].upper() in CURRENCY_CODES:
                with contextlib.suppress(ValueError):
                    amounts[parts[-1].upper()] = normalize_decimal_string(parts[0])
    if not groups:
        return {}

    # Preserve per-fragment provenance.  Rows are already tagged with their
    # original document/component id by the components extractor.
    sources = []
    for doc_id in table.source_table_ids or ([table.document_id] if table.document_id else []):
        sources.append(
            Provenance(
                requested_url=source.requested_url if source else None,
                canonical_url=source.canonical_url if source else None,
                page_id=source.page_id if source else None,
                page_title=source.page_title if source else None,
                document_id=doc_id,
                component_id=table.component_id,
                table_id=table.table_id,
                section_heading=table.caption or (table.section_path[-1] if table.section_path else None),
                classifier_version=_CLASSIFIER_VERSION,
            )
        )
    result: dict[str, FixedFeeSchedule] = {}
    for key, amounts in groups.items():
        if not amounts:
            continue
        suffix = _schedule_suffix_from_signature(group_keys[key])
        result[suffix] = FixedFeeSchedule(entries=amounts, sources=sources)
    return result


def _extract_maximum_fee_schedule(table: Table, source: Source | None = None) -> dict[str, FixedFeeSchedule]:
    """Extract per-region, per-applicability maximum-fee-cap schedules.

    Tables such as "Fee and maximum fee cap for PayPal Payouts" contain a
    currency column and several columns for different max-fee caps. Each cap
    column becomes a separate ``payouts_<region>`` schedule, further split by
    the row's market group or amount tier.
    """
    schedules: dict[str, FixedFeeSchedule] = {}
    if not table.headers:
        return schedules

    table_text = _table_context_original(table)
    for col_idx in range(1, len(table.headers)):
        base_id = _max_fee_base_id(table.headers[col_idx].text)
        if not base_id:
            continue

        groups: dict[frozenset[tuple[str, Any]], dict[str, str]] = {}
        group_keys: dict[frozenset[tuple[str, Any]], dict[str, Any]] = {}
        for row in table.rows:
            cells = [c for c in row.cells if c.text.strip()]
            if col_idx >= len(cells):
                continue
            money = _max_fee_money(cells[0], cells[col_idx])
            if not money:
                continue
            currency_code, amount = money
            signature = _schedule_signature_for_row(row, base_id, table_text, use_row_label=False)
            key = _signature_key(signature)
            group_keys.setdefault(key, signature)
            amounts = groups.setdefault(key, {})
            amounts[currency_code] = amount

        if not groups:
            continue

        sources = _max_fee_schedule_sources(table, source)
        for key, entries in groups.items():
            if not entries:
                continue
            schedule_id = _schedule_id(base_id, group_keys[key])
            _merge_max_fee_schedule(schedules, schedule_id, entries, sources)

    return schedules


def _max_fee_base_id(header: str) -> str | None:
    """Map a maximum-fee schedule header to its base schedule id."""
    header_norm = _norm(header)
    if "maximum fee cap" not in header_norm and "max fee cap" not in header_norm:
        return None
    if "us" in header_norm:
        return "payouts_us"
    if "domestic" in header_norm:
        return "payouts_domestic"
    if "international" in header_norm:
        return "payouts_international"
    return None


def _max_fee_money(currency_cell: Any, amount_cell: Any) -> tuple[str, str] | None:
    """Extract (currency_code, amount) from a maximum-fee schedule row."""
    money = _cell_money(amount_cell)
    if not money:
        parts = amount_cell.text.strip().split()
        if len(parts) >= 2 and parts[-1].upper() in CURRENCY_CODES:
            with contextlib.suppress(ValueError):
                money = (parts[-1].upper(), normalize_decimal_string(parts[0]))
        else:
            return None
    if not money:
        return None
    currency = _cell_money(currency_cell)
    if currency:
        return currency[0], money[1]
    return money


def _max_fee_schedule_sources(table: Table, source: Source | None) -> list[Provenance]:
    """Build provenance sources for a maximum-fee schedule."""
    sources = []
    for doc_id in table.source_table_ids or ([table.document_id] if table.document_id else []):
        sources.append(
            Provenance(
                requested_url=source.requested_url if source else None,
                canonical_url=source.canonical_url if source else None,
                page_id=source.page_id if source else None,
                page_title=source.page_title if source else None,
                document_id=doc_id,
                component_id=table.component_id,
                table_id=table.table_id,
                section_heading=table.caption or (table.section_path[-1] if table.section_path else None),
                classifier_version=_CLASSIFIER_VERSION,
            )
        )
    return sources


def _merge_max_fee_schedule(
    schedules: dict[str, FixedFeeSchedule],
    schedule_id: str,
    entries: dict[str, str],
    sources: list[Provenance],
) -> None:
    """Merge a group of maximum-fee entries into an existing or new schedule."""
    existing = schedules.get(schedule_id)
    if existing:
        merged_entries = dict(existing.entries)
        for currency, amount in entries.items():
            if currency not in merged_entries:
                merged_entries[currency] = amount
        schedules[schedule_id] = FixedFeeSchedule(
            entries=merged_entries,
            sources=existing.sources + sources,
            origin=existing.origin,
            inherited_from=existing.inherited_from,
            inheritance_reason=existing.inheritance_reason,
            inherited_sources=existing.inherited_sources,
        )
    else:
        schedules[schedule_id] = FixedFeeSchedule(entries=entries, sources=sources)


def _extract_international_surcharge_schedule(
    table: Table, base_name: str, source: Source | None = None
) -> dict[str, InternationalSurchargeSchedule]:
    """Extract international-surcharge schedules grouped by applicability signature."""
    table_text = _table_context_original(table)
    groups: dict[frozenset[tuple[str, Any]], list[InternationalSurchargeScheduleEntry]] = {}
    group_keys: dict[frozenset[tuple[str, Any]], dict[str, Any]] = {}
    fallback: list[tuple[dict[str, Any], str, str]] = []
    for row in table.rows:
        label = _row_label(row)
        if _is_limit_or_cap_row(label, _row_fee_cell(row)):
            continue
        pct = _first_percentage(row)
        region = _normalize_region(label)
        if pct is None:
            # "No fee" / "Free" entries represent a 0% surcharge.
            fee_text = _norm(_row_fee_cell(row))
            no_fee_phrases = (
                "no fee",
                "free",
                "nessuna tariffa",
                "nessun costo",
                "ei palkkiota",
                "ei maksua",
                "bez poplatku",
                "bez poplatkov",
                "geen kosten",
                "ingen avgift",
                "ingen gebyr",
                "χωρίς χρέωση",
                "χωρισ χρεωση",
                "díjmentes",
                "nincs díj",
                "0%",
                "0,00%",
                "0.00%",
            )
            if _keyword_match(fee_text, no_fee_phrases, word_boundary=False):
                pct = "0"
            else:
                continue
        signature = _schedule_signature_for_row(row, base_name, table_text, use_row_label=False)
        if region is None:
            # Some region-less tables (e.g. Brazil) list transaction types instead of
            # payer regions. Keep these as a fallback in case no region is recognized.
            if label:
                fallback.append((signature, label, pct))
            continue
        key = _signature_key(signature)
        group_keys.setdefault(key, signature)
        group_entries = groups.setdefault(key, [])
        # Avoid duplicate regions within the same applicability group.
        if any(e.payer_region == region for e in group_entries):
            continue
        group_entries.append(InternationalSurchargeScheduleEntry(payer_region=region, percentage_points=pct))
    if not groups and fallback:
        # No recognized region rows: treat the first percentage row as a generic
        # "OTHER" international surcharge. This is typically a region-less rate.
        signature, label, pct = fallback[0]
        key = _signature_key(signature)
        group_keys[key] = signature
        groups[key] = [InternationalSurchargeScheduleEntry(payer_region="OTHER", percentage_points=pct)]
    if not groups:
        return {}

    sources = []
    for doc_id in table.source_table_ids or ([table.document_id] if table.document_id else []):
        sources.append(
            Provenance(
                requested_url=source.requested_url if source else None,
                canonical_url=source.canonical_url if source else None,
                page_id=source.page_id if source else None,
                page_title=source.page_title if source else None,
                document_id=doc_id,
                component_id=table.component_id,
                table_id=table.table_id,
                section_heading=table.caption or (table.section_path[-1] if table.section_path else None),
                classifier_version=_CLASSIFIER_VERSION,
            )
        )
    result: dict[str, InternationalSurchargeSchedule] = {}
    for key, entries in groups.items():
        suffix = _schedule_suffix_from_signature(group_keys[key])
        result[suffix] = InternationalSurchargeSchedule(entries=entries, sources=sources)
    return result


def _matches_region_pattern(text: str, pattern: RegionPattern) -> bool:
    if isinstance(pattern, str):
        return pattern in text
    return all(part in text for part in pattern)


def _normalize_region(text: str) -> str | None:
    t = _norm(text)
    if not t:
        return None
    if t in _REGION_EXACT:
        return _REGION_EXACT[t]
    for region, patterns in _REGION_PATTERNS:
        if any(_matches_region_pattern(t, p) for p in patterns):
            return region
    return None


_SCHEDULE_LOOKUP_TABLES: dict[str, dict[str, str | None]] = {
    "fixed_fee": _FIXED_FEE_SCHEDULE_FOR,
    "international_surcharge": _INTERNATIONAL_SURCHARGE_SCHEDULE_FOR,
}


def _schedule_for(schedule_type: str, product_id: str, variant_id: str | None = None) -> str | None:
    """Generic schedule lookup over a ``{product_id: base_schedule_id}`` table."""
    table = _SCHEDULE_LOOKUP_TABLES.get(schedule_type)
    if table is None:
        return None
    base = table.get(product_id)
    if base is None:
        return None
    if variant_id is None or variant_id == "standard":
        return base
    if variant_id in _BASE_VARIANTS_BY_PRODUCT.get(product_id, frozenset()):
        return base
    return f"{base}_{variant_id}"


def _fixed_fee_schedule_for(product_id: str, variant_id: str | None = None) -> str | None:
    """Return the fixed-fee schedule name for a product and variant, or None if no fixed fee applies."""
    return _schedule_for("fixed_fee", product_id, variant_id)


def _international_surcharge_schedule_for(product_id: str, variant_id: str | None = None) -> str | None:
    """Return the international surcharge schedule name for a product and variant, or None."""
    return _schedule_for("international_surcharge", product_id, variant_id)


def _schedule_ids_for_table(
    base_name: str,
    applicable_variants: list[str],
    existing_names: set[str],
    product_is_direct: bool = False,
) -> list[str]:
    """Determine schedule ids to create for a fixed/international surcharge table.

    If the table is generic (no variants) or describes a base variant, the
    base name is used. Otherwise variant-specific ids are created. Direct
    products (withdrawals, disputes) never create a base schedule when a
    variant is present.
    """
    if not applicable_variants:
        return [base_name]
    base_variants = _BASE_VARIANTS_BY_PRODUCT.get(base_name, frozenset())
    has_base_variant = bool(set(applicable_variants) & base_variants)
    if has_base_variant and base_name not in existing_names and not product_is_direct:
        return [base_name]
    return [f"{base_name}_{variant}" for variant in applicable_variants]


def _merge_fixed_like_schedules(
    existing: FixedFeeSchedule,
    schedule: FixedFeeSchedule,
    name: str,
    schedule_type: str,
) -> tuple[FixedFeeSchedule, list[Diagnostic]]:
    """Merge a fixed-like schedule into an existing one, reporting conflicts."""
    merged_entries = dict(existing.entries)
    merged_sources = list(existing.sources)
    diagnostics: list[Diagnostic] = []
    for s in schedule.sources:
        if s not in merged_sources:
            merged_sources.append(s)
    for currency, amount in schedule.entries.items():
        if currency in merged_entries:
            if merged_entries[currency] != amount:
                diagnostics.append(
                    Diagnostic(
                        type="conflicting_schedule_entry",
                        schedule_type=schedule_type,
                        schedule_id=name,
                        normalized_key=currency,
                        values=[merged_entries[currency], amount],
                        sources=_merge_provenance_sources(existing.sources, schedule.sources),
                    )
                )
            # Keep first value; do not overwrite.
        else:
            merged_entries[currency] = amount

    # Preserve inherited provenance; direct + direct stays direct.
    merged_inherited_sources = list(existing.inherited_sources)
    for s in schedule.inherited_sources:
        if s not in merged_inherited_sources:
            merged_inherited_sources.append(s)
    origin = existing.origin
    inherited_from = existing.inherited_from
    inheritance_reason = existing.inheritance_reason
    if schedule.origin == "inherited":
        origin = "inherited"
        inherited_from = schedule.inherited_from or inherited_from
        inheritance_reason = schedule.inheritance_reason or inheritance_reason
        for s in schedule.sources:
            if s not in merged_inherited_sources:
                merged_inherited_sources.append(s)
    return FixedFeeSchedule(
        entries=merged_entries,
        sources=merged_sources,
        origin=origin,
        inherited_from=inherited_from,
        inheritance_reason=inheritance_reason,
        inherited_sources=merged_inherited_sources,
    ), diagnostics


def _merge_international_surcharge_schedules(
    existing: InternationalSurchargeSchedule,
    schedule: InternationalSurchargeSchedule,
    name: str,
) -> tuple[InternationalSurchargeSchedule, list[Diagnostic]]:
    """Merge an international surcharge schedule into an existing one."""
    merged_entries = list(existing.entries)
    seen = {e.payer_region: e for e in merged_entries}
    merged_sources = list(existing.sources)
    diagnostics: list[Diagnostic] = []
    for s in schedule.sources:
        if s not in merged_sources:
            merged_sources.append(s)
    for entry in schedule.entries:
        if entry.payer_region in seen:
            if seen[entry.payer_region].percentage_points != entry.percentage_points:
                diagnostics.append(
                    Diagnostic(
                        type="conflicting_schedule_entry",
                        schedule_type="international_surcharge",
                        schedule_id=name,
                        normalized_key=entry.payer_region,
                        values=[
                            seen[entry.payer_region].percentage_points or "",
                            entry.percentage_points or "",
                        ],
                        sources=_merge_provenance_sources(existing.sources, schedule.sources),
                    )
                )
            # Keep first value; do not overwrite.
        else:
            merged_entries.append(entry)
            seen[entry.payer_region] = entry

    merged_inherited_sources = list(existing.inherited_sources)
    for s in schedule.inherited_sources:
        if s not in merged_inherited_sources:
            merged_inherited_sources.append(s)
    origin = existing.origin
    inherited_from = existing.inherited_from
    inheritance_reason = existing.inheritance_reason
    if schedule.origin == "inherited":
        origin = "inherited"
        inherited_from = schedule.inherited_from or inherited_from
        inheritance_reason = schedule.inheritance_reason or inheritance_reason
        for s in schedule.sources:
            if s not in merged_inherited_sources:
                merged_inherited_sources.append(s)
    return InternationalSurchargeSchedule(
        entries=merged_entries,
        sources=merged_sources,
        origin=origin,
        inherited_from=inherited_from,
        inheritance_reason=inheritance_reason,
        inherited_sources=merged_inherited_sources,
    ), diagnostics


def _collect_fixed_fee_table(
    table: Table,
    source: Source | None,
    fixed: dict[str, FixedFeeSchedule],
    direct_products: set[str],
) -> list[Diagnostic]:
    """Collect fixed-fee schedules from a single table into ``fixed``."""
    base_name = _schedule_name_from_table(table, "commercial")
    schedules_by_sig = _extract_fixed_fee_schedule(table, base_name, source=source)
    if not schedules_by_sig:
        return []
    applicable_variants = _applicable_variants_for_table(table, base_name)
    product_is_direct = base_name in direct_products
    diagnostics: list[Diagnostic] = []
    for base_id in _schedule_ids_for_table(base_name, applicable_variants, set(fixed.keys()), product_is_direct):
        for suffix, schedule in schedules_by_sig.items():
            name = f"{base_id}__{suffix}" if suffix else base_id
            existing = fixed.get(name)
            if existing:
                merged, new_diagnostics = _merge_fixed_like_schedules(existing, schedule, name, "fixed_fee")
                diagnostics.extend(new_diagnostics)
                fixed[name] = merged
            else:
                fixed[name] = schedule
    return diagnostics


def _collect_international_surcharge_table(
    table: Table,
    source: Source | None,
    international: dict[str, InternationalSurchargeSchedule],
) -> list[Diagnostic]:
    """Collect international-surcharge schedules from a single table."""
    base_name = _schedule_name_from_table(table, "commercial")
    schedules_by_sig = _extract_international_surcharge_schedule(table, base_name, source=source)
    if not schedules_by_sig:
        return []
    applicable_variants = _applicable_variants_for_table(table, base_name)
    diagnostics: list[Diagnostic] = []
    for base_id in _schedule_ids_for_table(base_name, applicable_variants, set(international.keys())):
        for suffix, schedule in schedules_by_sig.items():
            name = f"{base_id}__{suffix}" if suffix else base_id
            existing = international.get(name)
            if existing:
                merged, new_diagnostics = _merge_international_surcharge_schedules(existing, schedule, name)
                diagnostics.extend(new_diagnostics)
                international[name] = merged
            else:
                international[name] = schedule
    return diagnostics


def _collect_maximum_fee_table(
    table: Table,
    source: Source | None,
    maximum: dict[str, FixedFeeSchedule],
) -> list[Diagnostic]:
    """Collect maximum-fee schedules from a single table."""
    diagnostics: list[Diagnostic] = []
    for name, schedule in _extract_maximum_fee_schedule(table, source=source).items():
        existing = maximum.get(name)
        if existing:
            merged, new_diagnostics = _merge_fixed_like_schedules(existing, schedule, name, "maximum_fee")
            diagnostics.extend(new_diagnostics)
            maximum[name] = merged
        else:
            maximum[name] = schedule
    return diagnostics


def _collect_schedules(
    tables: list[Table],
    source: Source | None = None,
    table_categories: dict[int, str | None] | None = None,
) -> tuple[
    dict[str, FixedFeeSchedule],
    dict[str, InternationalSurchargeSchedule],
    dict[str, FixedFeeSchedule],
    list[Diagnostic],
]:
    """Extract fixed-fee, international-surcharge and maximum-fee schedules.

    Schedules are keyed by product name. If two tables map to the same product
    (e.g. "Fixed fee by received currency" and "Currency fixed fees" both for
    commercial), their entries are merged and sources are combined. Conflicting
    duplicate keys are reported as diagnostics and the first encountered value
    is kept.
    """
    fixed: dict[str, FixedFeeSchedule] = {}
    international: dict[str, InternationalSurchargeSchedule] = {}
    maximum: dict[str, FixedFeeSchedule] = {}
    diagnostics: list[Diagnostic] = []

    direct_products = set(_DIRECT_FIXED_FEE_SCHEDULE_PRODUCTS.values())

    for table in tables:
        category = table_categories[id(table)] if table_categories else _classify_table_category(table)
        if category == "fixed_fee_table":
            diagnostics.extend(_collect_fixed_fee_table(table, source, fixed, direct_products))
        elif category == "international_surcharge_table":
            diagnostics.extend(_collect_international_surcharge_table(table, source, international))
        elif category == "maximum_fee_table":
            diagnostics.extend(_collect_maximum_fee_table(table, source, maximum))

    return fixed, international, maximum, diagnostics


def _source_schedule_id(source_base: str, intended_id: str, schedules: dict[str, Any]) -> str | None:
    """Choose the most specific source schedule matching the intended id's suffix."""
    if "__" in intended_id:
        suffix = intended_id.split("__", 1)[1]
        suffixed = f"{source_base}__{suffix}"
        if suffixed in schedules:
            return suffixed
    return source_base if source_base in schedules else None


def _inheritance_map_for(schedule_type: str) -> dict[str, str]:
    """Return the inheritance map for the given schedule type."""
    if schedule_type == "fixed_fee":
        return _FIXED_FEE_INHERITANCE
    if schedule_type == "international_surcharge":
        return _INTERNATIONAL_SURCHARGE_INHERITANCE
    return {}


def _source_text_of(extracted: _ExtractedRule | None) -> str:
    """Return the lowercased fixed-fee expression from the extracted row, if any."""
    return (extracted.fixed_expr or "").lower() if extracted else ""


def _source_text_evidence(source_text: str, source_base: str) -> str | None:
    """Return an evidence string when the row text names the source schedule."""
    if not source_text:
        return None
    if (
        source_base == "commercial"
        and "commercial" in source_text
        and ("transaction" in source_text or "fixed fee" in source_text)
    ):
        return "source text references commercial fixed fee"
    if source_base == "online_card_payments" and "online card" in source_text:
        return "source text references online card fixed fee"
    return None


def _schedule_heading_text(schedules: dict[str, Any], source_schedule_id: str) -> str:
    """Return the lowercased section heading of the source schedule, if available."""
    source_schedule = schedules.get(source_schedule_id)
    if source_schedule and source_schedule.sources:
        return (source_schedule.sources[0].section_heading or "").lower()
    return ""


def _table_context_evidence(
    source_base: str,
    schedules: dict[str, Any],
    source_schedule_id: str,
) -> str | None:
    """Return an evidence string when the source schedule's table context supports inheritance."""
    heading = _schedule_heading_text(schedules, source_schedule_id)
    if source_base == "online_card_payments" and "online card" in heading:
        return "table context references online card fixed fee"
    if source_base == "commercial" and "commercial transactions" in heading:
        return "table context references commercial fixed fee"
    return None


def _inheritance_evidence(
    rule: TransactionFeeRule,
    extracted: _ExtractedRule | None,
    source_base: str,
    source_schedule_id: str,
    schedule_type: str,
    schedules: dict[str, Any],
) -> str | None:
    """Return a human-readable evidence string when inheritance is allowed."""
    inheritance_map = _inheritance_map_for(schedule_type)

    # The inheritance map is itself an explicit documented product rule.  Even
    # for mapped products we still require corroborating source text or table
    # context so there is no implicit cross-product fallback.
    mapped_source = inheritance_map.get(rule.id)
    if mapped_source and mapped_source != source_base:
        return None

    # Maximum-fee schedules have their own explicit fallback map and do not
    # require source-text corroboration.
    if schedule_type == "maximum_fee" and source_base:
        return f"explicit maximum-fee fallback from {source_base}"

    source_text = _source_text_of(extracted)

    reason = _source_text_evidence(source_text, source_base)
    if reason:
        return reason

    reason = _table_context_evidence(source_base, schedules, source_schedule_id)
    if reason:
        return reason

    # When the product is explicitly mapped and no contradictory evidence
    # exists, the map itself documents the inheritance.
    if mapped_source == source_base:
        return f"explicit product inheritance from {source_base}"

    return None


def _create_inherited_schedule(
    schedule_id: str,
    source_schedule_id: str,
    schedules: dict[str, Any],
    rule_source: Provenance | None,
    reason: str,
) -> None:
    source = schedules[source_schedule_id]
    inherited_sources = list(source.sources)
    sources = list(source.sources)
    if rule_source and rule_source not in sources:
        sources.append(rule_source)
    schedules[schedule_id] = source.model_copy(
        update={
            "origin": "inherited",
            "inherited_from": source_schedule_id,
            "inheritance_reason": reason,
            "inherited_sources": inherited_sources,
            "sources": sources,
        }
    )


def _resolve_schedule(
    schedule_id: str,
    source_id: str | None,
    schedule_type: str,
    schedules: dict[str, Any],
    inheritance_map: dict[str, str],
    rule: TransactionFeeRule,
    extracted: _ExtractedRule | None,
    diagnostics: list[Diagnostic],
) -> str | None:
    """Return the actual source schedule id if inheritance is allowed, else None."""
    if not source_id:
        return None
    source_base = source_id.split("__", 1)[0]
    expected_source = inheritance_map.get(rule.id)
    if expected_source and source_base != expected_source:
        return None
    actual_source = _source_schedule_id(source_base, schedule_id, schedules)
    if not actual_source:
        return None
    reason = _inheritance_evidence(rule, extracted, source_base, actual_source, schedule_type, schedules)
    if not reason:
        return None
    _create_inherited_schedule(schedule_id, actual_source, schedules, rule.source, reason)
    diagnostics.append(
        Diagnostic(
            type="inherited_schedule",
            rule_id=rule.id,
            schedule_type=schedule_type,
            expected_schedule=schedule_id,
            inherited_from=actual_source,
            sources=[rule.source] if rule.source else [],
        )
    )
    return actual_source


def _resolve_schedule_inheritance(
    extracted_rules: list[_ExtractedRule],
    rules: list[TransactionFeeRule],
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
    diagnostics: list[Diagnostic],
) -> None:
    """Create inherited schedules for product rules whose own schedule is missing.

    Inheritance is only performed when the source text/table context contains an
    explicit reference to the source schedule family, or when the product is in
    the explicit inheritance map and the source text/table context supports it.
    Every inherited schedule records its source schedule and provenance from
    both the requesting rule and the source schedule.
    """

    _schedule_attrs: list[tuple[str, dict[str, Any], dict[str, str]]] = [
        ("fixed_fee", fixed_schedules, _FIXED_FEE_INHERITANCE),
        ("international_surcharge", international_schedules, _INTERNATIONAL_SURCHARGE_INHERITANCE),
        ("maximum_fee", maximum_fee_schedules, {}),
    ]

    # Collect all missing schedule references and create inherited schedules
    # before updating any rule.  Base schedules are created first (shorter ids)
    # so variant-specific ids can reuse an inherited base schedule.
    refs: list[
        tuple[str, str | None, str, dict[str, Any], dict[str, str], TransactionFeeRule, _ExtractedRule | None]
    ] = []
    for i, rule in enumerate(rules):
        extracted = extracted_rules[i] if i < len(extracted_rules) else None
        for schedule_type, schedules, inheritance_map in _schedule_attrs:
            schedule_id = getattr(rule, f"{schedule_type}_schedule")
            source_id = getattr(extracted, f"{schedule_type}_schedule_source", None)
            if schedule_id and schedule_id not in schedules:
                refs.append((schedule_id, source_id, schedule_type, schedules, inheritance_map, rule, extracted))

    # Deduplicate by (schedule_type, schedule_id), preferring an entry with a
    # concrete source id so we can resolve the source schedule.
    seen: dict[
        tuple[str, str],
        tuple[str, str | None, str, dict[str, Any], dict[str, str], TransactionFeeRule, _ExtractedRule | None],
    ] = {}
    for ref in refs:
        key = (ref[2], ref[0])
        existing = seen.get(key)
        if existing is None or (existing[1] is None and ref[1] is not None):
            seen[key] = ref
    unique_refs = sorted(seen.values(), key=lambda r: (len(r[0]), r[0]))

    created: set[tuple[str, str]] = set()
    schedule_source: dict[tuple[str, str], str | None] = {}
    for schedule_id, source_id, schedule_type, schedules, inheritance_map, rule, extracted in unique_refs:
        source = _resolve_schedule(
            schedule_id, source_id, schedule_type, schedules, inheritance_map, rule, extracted, diagnostics
        )
        if source:
            created.add((schedule_type, schedule_id))
        schedule_source[(schedule_type, schedule_id)] = source

    # Update every rule to use an inherited schedule when one was created, or
    # report it as missing.
    for i, rule in enumerate(rules):
        extracted = extracted_rules[i] if i < len(extracted_rules) else None
        updates: dict[str, Any] = {}
        for schedule_type, schedules, _ in _schedule_attrs:
            attr = f"{schedule_type}_schedule"
            schedule_id = getattr(rule, attr)
            if not schedule_id:
                continue
            if schedule_id in schedules:
                continue
            if (schedule_type, schedule_id) in created:
                continue
            diagnostics.append(
                Diagnostic(
                    type="missing_required_schedule",
                    rule_id=rule.id,
                    schedule_type=schedule_type,
                    expected_schedule=schedule_id,
                    sources=[rule.source] if rule.source else [],
                )
            )
            updates[attr] = None
        if updates:
            rules[i] = rule.model_copy(update=updates)


def _product_family_for_schedule_id(schedule_id: str, schedule_type: str = "fixed_fee") -> str:
    """Return the product-family base name for a schedule id.

    For fixed-fee and international-surcharge schedules the family is the
    longest known product base that matches the id's prefix (e.g.
    ``micropayments_domestic`` -> ``micropayments``).  Maximum-fee schedules
    have no variant prefix, so the family is the base id before any ``__``
    applicability suffix.
    """
    base = schedule_id.split("__", 1)[0]
    if schedule_type == "maximum_fee":
        return base
    product_bases: set[str] = set()
    for value in _FIXED_FEE_SCHEDULE_FOR.values():
        if value:
            product_bases.add(value)
    for value in _INTERNATIONAL_SURCHARGE_SCHEDULE_FOR.values():
        if value:
            product_bases.add(value)
    candidates = [pb for pb in product_bases if base == pb or base.startswith(pb + "_")]
    if candidates:
        return max(candidates, key=len)
    return base


def _validate_inheritance_priorities(
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    maximum_fee_schedules: dict[str, FixedFeeSchedule],
    diagnostics: list[Diagnostic],
) -> None:
    """Flag inherited schedules that selected a cross-product fallback while a
    direct product-family schedule was available.

    The validation mirrors the resolution priority enforced by
    ``_select_schedule_id``: a variant-specific schedule may fall back to its
    own product base, but it must never fall back to a different product family
    when its own product base exists.
    """
    schedule_groups: list[tuple[str, dict[str, Any]]] = [
        ("fixed_fee", fixed_schedules),
        ("international_surcharge", international_schedules),
        ("maximum_fee", maximum_fee_schedules),
    ]
    for schedule_type, schedules in schedule_groups:
        invalid: list[str] = []
        for schedule_id, schedule in schedules.items():
            if schedule.origin != "inherited":
                continue
            target_family = _product_family_for_schedule_id(schedule_id, schedule_type)
            source_id = schedule.inherited_from or ""
            source_family = _product_family_for_schedule_id(source_id, schedule_type)
            if target_family == source_family:
                continue
            # Product-base schedules (possibly with an applicability suffix) may
            # be explicitly inherited from another product family; only
            # variant-specific schedules are checked for bypassing their own base.
            base_part = schedule_id.split("__", 1)[0]
            if base_part == target_family:
                continue
            suffix = ""
            if "__" in schedule_id:
                suffix = schedule_id.split("__", 1)[1]
            direct_candidates: list[str] = []
            if suffix:
                direct_candidates.append(f"{target_family}__{suffix}")
            direct_candidates.append(target_family)
            for direct in direct_candidates:
                if direct in schedules:
                    diagnostics.append(
                        Diagnostic(
                            type="inappropriate_inheritance",
                            rule_id=target_family,
                            schedule_type=schedule_type,
                            schedule_id=schedule_id,
                            inherited_from=source_id,
                            expected_schedule=direct,
                            sources=schedule.sources,
                        )
                    )
                    invalid.append(schedule_id)
                    break
        for schedule_id in invalid:
            del schedules[schedule_id]


def _merge_provenance_sources(*source_lists: list[Provenance]) -> list[Provenance]:
    """Combine multiple provenance lists without duplicates."""
    merged: list[Provenance] = []
    for sources in source_lists:
        for s in sources:
            if s not in merged:
                merged.append(s)
    return merged
