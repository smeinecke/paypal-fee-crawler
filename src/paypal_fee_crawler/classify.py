"""Derive product-specific transaction fee rules from normalized PayPal tables.

The classifier works at the row level: a single PayPal table may contain several
independent payment products, and each relevant fee row becomes a separate
``TransactionFeeRule``.  Fixed-fee and international-surcharge schedules are kept
separate per product or product family so that an HTTP fee calculator can select
the schedule that applies to a given rule.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any

from .models import (
    AmbiguousFeeRow,
    CurrencyConversion,
    DerivedFeeResult,
    FixedFeeSchedule,
    InternationalSurchargeSchedule,
    InternationalSurchargeScheduleEntry,
    Provenance,
    RateReference,
    ResolvedRate,
    Row,
    Source,
    Table,
    TransactionFeeRule,
    UnclassifiedFeeRow,
)
from .normalize import clean_text, normalize_decimal_string
from .pricing_tokens import CURRENCY_CODES

logger = logging.getLogger(__name__)

CLASSIFIER_VERSION = "rules-v1"
_CLASSIFIER_VERSION = CLASSIFIER_VERSION

# ---------------------------------------------------------------------------
# Language-aware product aliases.  More specific / longer aliases are listed
# first so that substring scoring naturally prefers them.
# ---------------------------------------------------------------------------

_PRODUCT_ALIASES: dict[str, tuple[str, ...]] = {
    "paypal_checkout": (
        "paypal checkout",
        "paypal-zahlung",
        "paypal bezahlen",
        "paypal checkout-transaktionen",
    ),
    "goods_and_services": (
        "sending and receiving money for goods and services",
        "geld für waren und dienstleistungen senden/empfangen",
        "geld für waren und dienstleistungen",
        "waren und dienstleistungen",
        "goods and services",
        "goods & services",
        "goods or services",
    ),
    "advanced_card_payments": (
        "advanced credit and debit card payments",
        "erweiterte kredit- und debitkartenzahlungen",
        "zahlungen mit kredit- und debitkarten mit erweiterten funktionen",
        "kredit- und debitkarten mit erweiterten funktionen",
        "advanced card",
        "erweiterte kartenzahlung",
    ),
    "other_commercial": (
        "all other commercial transactions",
        "alle anderen geschäftlichen transaktionen",
        "sonstige gewerbliche transaktionen",
        "other commercial",
        "sonstige geschäftliche",
        "other commercial transactions",
    ),
    "alternative_payment_methods": (
        "alle anderen alternativen zahlungsmethoden",
        "alternative zahlungsmethode",
        "alternative zahlungsmethoden",
        "alternative payment method",
        "alternative payment methods",
        "all other alternative payment methods",
        "apm-transaktionsgebühren",
        "apm",
    ),
    "guest_checkout": (
        "zahlung eines nutzers unserer bedingungen für zahlungen ohne paypal-konto",
        "zahlungen ohne paypal-konto",
        "payments without a paypal account",
        "zahlung ohne paypal-konto",
        "guest checkout",
    ),
    "invoice_pay_later": (
        "rechnungskauf mit ratepay",
        "ratepay",
        "invoice payments",
        "pay later",
        "ratenzahlungsangebote",
        "rechnungskauf",
    ),
    "qr_code_payments": (
        "qr-code-transaktionen",
        "qr-code transactions",
        "qr-code-zahlungen",
        "qr code transactions",
        "qr-code",
        "qr code",
    ),
    "donations": (
        "paypal-spendenaktionen",
        "spendenaktionen",
        "spende",
        "donation",
        "charity donation",
    ),
    "nonprofit": (
        "gemeinnützige organisationen",
        "gemeinnützig",
        "gemeinnutzig",
        "nonprofit organisation",
        "nonprofit",
        "non-profit",
    ),
    "micropayments": (
        "mikrozahlung",
        "micropayment",
        "kleinbetragszahlung",
    ),
    "chargebacks": (
        "rückbuchungsgebühren",
        "rückbuchung",
        "rückabwicklung",
        "chargeback",
    ),
    "disputes": (
        "konfliktgebühren",
        "konfliktgebühr",
        "streitfall",
        "dispute",
    ),
    "refunds": (
        "rückerstattung",
        "rückzahlung",
        "refund",
    ),
    "currency_conversion": (
        "währungsumrechnung",
        "umrechnung",
        "currency conversion",
        "wechselkurs",
    ),
    "withdrawals": (
        "guthaben von einem paypal-geschäftskonto abbuchen",
        "abbuchen",
        "auszahlung",
        "withdrawal",
        "payout",
    ),
    "card_verification": (
        "kartenverifizierung",
        "karten verifizierung",
        "card verification",
        "kartenbestätigung",
    ),
    "pay_later_consumer": (
        "paypal-ratenzahlungsangebote",
        "ratenzahlungsangebote",
        "pay in 4",
        "pay later",
    ),
    "pos_transactions": (
        "point of sale",
        "paypal point of sale",
        "präsenter karte",
        "card present",
    ),
}

# Order used for stable output.
_PRODUCT_ORDER = list(_PRODUCT_ALIASES)

# ---------------------------------------------------------------------------
# Table category detection
# ---------------------------------------------------------------------------

_TABLE_CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "commercial_rate_table": (
        "standardgebühr beim empfang von inlandstransaktionen",
        "empfangen von inlandstransaktionen",
        "standard transaction fees",
        "commercial transaction fees",
        "geschäftlichen transaktionen",
    ),
    "online_card_rate_table": (
        "paypal-dienste für online-kartenzahlungen",
        "paypal-dienste für online-zahlungen",
        "online card payments",
        "online-kartenzahlungen",
        "online card",
    ),
    "goods_and_services_rate_table": (
        "geld für waren und dienstleistungen",
        "goods and services",
    ),
    "donation_rate_table": (
        "empfang von inlandsspenden",
        "donation",
        "spenden",
    ),
    "nonprofit_rate_table": (
        "gemeinnützige organisationen",
        "nonprofit",
        "non-profit",
    ),
    "apm_rate_table": (
        "alternative zahlungsmethode",
        "alternative payment method",
        "apm-transaktionen",
        "apm",
    ),
    "pos_rate_table": (
        "point of sale",
        "paypal point of sale",
        "präsenter karte",
    ),
    "micropayment_rate_table": (
        "mikrozahlung",
        "micropayment",
    ),
    "fixed_fee_table": (
        "festgebühr",
        "fixed fee",
    ),
    "international_surcharge_table": (
        "zusätzliche prozentuale gebühr",
        "prozentuale zusatzgebühr",
        "international surcharge",
        "international",
        "ausland",
        "zusatzgebühr",
    ),
    "currency_conversion_table": (
        "währungsumrechnung",
        "umrechnung",
        "currency conversion",
        "guthaben umrechnen",
    ),
}

# When a caption contains one of these negative signals, it is not treated as a
# rate table even if it also contains commercial keywords.
_TABLE_NEGATIVE_SIGNALS: dict[str, tuple[str, ...]] = {
    "commercial_rate_table": (
        "spende",
        "donation",
        "gemeinnützig",
        "nonprofit",
        "mikrozahlung",
        "micropayment",
        "alternative zahlungsmethode",
        "alternative payment",
        "online-kartenzahlungen",
        "online card",
        "point of sale",
        "qr-code",
        "qr code",
    ),
}

# Map a table category to the default schedule name used by its rows.
_TABLE_CATEGORY_SCHEDULE: dict[str, str] = {
    "commercial_rate_table": "commercial",
    "online_card_rate_table": "online_card_payments",
    "goods_and_services_rate_table": "goods_and_services",
    "donation_rate_table": "donations",
    "nonprofit_rate_table": "nonprofit",
    "apm_rate_table": "alternative_payment_methods",
    "pos_rate_table": "pos_transactions",
    "micropayment_rate_table": "micropayments",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _norm(text: str | None) -> str:
    return clean_text(text or "").lower()


def _table_text(table: Table) -> str:
    parts = list(table.section_path or []) + [table.caption or ""]
    for header in table.headers:
        parts.append(header.text)
    return _norm(" ".join(parts))


def _row_text(row: Row) -> str:
    return _norm(" ".join(c.text for c in row.cells))


def _row_cells_text(row: Row) -> list[str]:
    return [c.text for c in row.cells]


def _first_percentage(row: Row) -> str | None:
    for cell in row.cells:
        for token in cell.tokens:
            if token.kind == "percentage" and token.value:
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


def _first_money_text(row: Row) -> str | None:
    for cell in row.cells:
        for token in cell.tokens:
            if token.kind == "money" and token.amount and token.currency:
                return f"{token.amount} {token.currency}"
    return None


def _extract_all_money(row: Row) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for cell in row.cells:
        for token in cell.tokens:
            if token.kind == "money" and token.amount and token.currency:
                key = (token.currency, token.amount)
                if key not in seen:
                    out.append(key)
                    seen.add(key)
    return out


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


# ---------------------------------------------------------------------------
# Product classification
# ---------------------------------------------------------------------------


def _score_label_against_product(label: str, aliases: tuple[str, ...]) -> int:
    normalized = _norm(label)
    best = 0
    for alias in aliases:
        alias_norm = _norm(alias)
        if alias_norm == normalized:
            return max(best, len(alias_norm) * 10)
        if alias_norm in normalized:
            best = max(best, len(alias_norm))
    return best


def _classify_product(label: str) -> tuple[str | None, list[str]]:
    """Return the best matching product ID and any ambiguous alternatives."""
    scores: dict[str, int] = {}
    for product_id, aliases in _PRODUCT_ALIASES.items():
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


# ---------------------------------------------------------------------------
# Table category classification
# ---------------------------------------------------------------------------


# Map a product id to the rate-table category that owns it, used as a fallback
# when the table caption is too generic to classify from metadata alone.
_PRODUCT_CATEGORY_MAP: dict[str, str] = {
    "alternative_payment_methods": "apm_rate_table",
    "advanced_card_payments": "online_card_rate_table",
    "pos_transactions": "pos_rate_table",
    "micropayments": "micropayment_rate_table",
    "donations": "donation_rate_table",
    "nonprofit": "nonprofit_rate_table",
    "goods_and_services": "goods_and_services_rate_table",
}


def _classify_table_category(table: Table) -> str | None:
    text = _table_text(table)
    # Explicit schedule-type captions are authoritative and win over product
    # rate-table keywords such as "commercial transactions" or "donations".
    if "festgebühr" in text:
        return "fixed_fee_table"
    if "prozentuale zusatzgebühr" in text or "zusätzliche prozentuale gebühr" in text:
        return "international_surcharge_table"
    if "währungsumrechnung" in text or "umrechnung des guthabens" in text:
        return "currency_conversion_table"

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
        # Fallback: infer the category from product-specific row labels when the
        # caption/headers are too generic (e.g. APM tables titled "Receiving
        # Inland Transactions").
        return _classify_table_by_row_labels(table)
    max_score = max(scores.values())
    candidates = [cat for cat, sc in scores.items() if sc == max_score]
    # Apply negative signals: a commercial rate table should not also be a more
    # specific product table.
    if "commercial_rate_table" in candidates:
        negatives = _TABLE_NEGATIVE_SIGNALS.get("commercial_rate_table", ())
        for neg in negatives:
            if _norm(neg) in text:
                candidates.remove("commercial_rate_table")
                break
    if not candidates:
        # If the commercial candidate was the only one and got removed, fall back
        # to the next-highest-scoring category or to row-label inference.
        remaining = {cat: sc for cat, sc in scores.items() if cat != "commercial_rate_table"}
        if remaining:
            next_score = max(remaining.values())
            candidates = [cat for cat, sc in remaining.items() if sc == next_score]
            if len(candidates) == 1:
                return candidates[0]
        return _classify_table_by_row_labels(table)
    return candidates[0]


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
    return max(category_counts.items(), key=lambda kv: kv[1])[0]


# ---------------------------------------------------------------------------
# Schedule naming
# ---------------------------------------------------------------------------


def _schedule_name_from_table(table: Table, default: str | None) -> str:
    text = _table_text(table)
    mapping = {
        "goods_and_services": (
            "geld für waren und dienstleistungen",
            "waren und dienstleistungen",
            "goods and services",
        ),
        "donations": ("spende", "donation"),
        "nonprofit": ("gemeinnützig", "nonprofit", "non-profit"),
        "micropayments": ("mikrozahlung", "micropayment"),
        "alternative_payment_methods": (
            "alternative zahlungsmethode",
            "alternative payment",
            "apm",
        ),
        "online_card_payments": (
            "online-kartenzahlungen",
            "online card",
            "online card payments",
        ),
        "pos_transactions": ("point of sale", "präsenter karte"),
        "commercial": (
            "geschäftlichen transaktionen",
            "commercial transaction",
            "commercial",
        ),
    }
    for name, keywords in mapping.items():
        for kw in keywords:
            if _norm(kw) in text:
                return name
    return default or "commercial"


# ---------------------------------------------------------------------------
# Schedule extraction
# ---------------------------------------------------------------------------


def _extract_fixed_fee_schedule(table: Table) -> FixedFeeSchedule | None:
    amounts: dict[str, str] = {}
    for row in table.rows:
        cells = [c for c in row.cells if c.text.strip()]
        # Fixed-fee tables are laid out as (currency label, amount) pairs, sometimes
        # with two pairs per row.
        for i in range(0, len(cells) - 1, 2):
            amount_cell = cells[i + 1]
            money = _cell_money(amount_cell)
            if money:
                amounts[money[0]] = money[1]
                continue
            # Some cells contain templated placeholders like {{...}}; try to infer
            # the currency code from the placeholder key and skip the amount.
            if "{{" in amount_cell.text:
                continue
            # Fallback: parse an explicit "amount CUR" text.
            parts = amount_cell.text.strip().split()
            if len(parts) >= 2 and parts[-1].upper() in CURRENCY_CODES:
                with contextlib.suppress(ValueError):
                    amounts[parts[-1].upper()] = normalize_decimal_string(parts[0])
    if not amounts:
        return None
    return FixedFeeSchedule(**amounts)


def _extract_international_surcharge_schedule(table: Table) -> InternationalSurchargeSchedule | None:
    entries: list[InternationalSurchargeScheduleEntry] = []
    seen: set[str] = set()
    for row in table.rows:
        pct = _first_percentage(row)
        label = _row_label(row)
        region = _normalize_region(label)
        if region is None:
            continue
        if pct is None:
            continue
        if region in seen:
            continue
        seen.add(region)
        entries.append(InternationalSurchargeScheduleEntry(payer_region=region, percentage_points=pct))
    if not entries:
        return None
    return InternationalSurchargeSchedule(entries=entries)


def _normalize_region(text: str) -> str | None:
    t = _norm(text)
    if not t:
        return None
    exact = {"eu": "EEA", "gb": "GB", "uk": "GB", "us": "US_CA"}
    if t in exact:
        return exact[t]
    if "europäischer wirtschaftsraum" in t or "ewr" in t or "eea" in t or "e.u" in t:
        return "EEA"
    if "vereinigtes königreich" in t or "großbritannien" in t or "united kingdom" in t or "britain" in t:
        return "GB"
    if "usa" in t or "united states" in t or "u.s" in t or "canada" in t:
        return "US_CA"
    if ("all" in t and "other" in t) or "rest" in t or "restante" in t:
        return "OTHER"
    return None


# ---------------------------------------------------------------------------
# Reference detection and resolution
# ---------------------------------------------------------------------------

_REFERENCE_SCHEDULE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "online_card_payments": (
        "online-kartenzahlungen",
        "online card payments",
        "erweiterte kredit- und debitkartenzahlungen",
        "advanced credit and debit card",
        "online card",
    ),
    "alternative_payment_methods": (
        "alternative zahlungsmethode",
        "alternative payment methods",
        "alternative payment",
        "apm",
    ),
    "goods_and_services": (
        "waren und dienstleistungen",
        "goods and services",
    ),
    "donations": (
        "spenden",
        "donation",
    ),
    "nonprofit": (
        "gemeinnützig",
        "nonprofit",
    ),
}

# Map a reference suffix to the product id it denotes.
_REFERENCE_SUFFIX_TO_PRODUCT: dict[str, str] = {
    "advanced": "advanced_card_payments",
}

# Inverse: map a product id to the reference suffix used in a qualified reference.
_REFERENCE_PRODUCT_SUFFIX: dict[str, str] = {v: k for k, v in _REFERENCE_SUFFIX_TO_PRODUCT.items()}


def _detect_reference(row: Row, product_id: str | None) -> str | None:
    """Detect when a row does not contain a numeric rate but refers to another schedule."""
    if _row_has_percentage(row):
        return None
    fee_text = _norm(_row_fee_cell(row))
    if not fee_text or "{{" in fee_text:
        return None
    # A reference is a textual pointer; if it already contains money, it is
    # likely a flat-fee rule, not a reference.
    if _first_money(row):
        return None
    for schedule_name, keywords in _REFERENCE_SCHEDULE_KEYWORDS.items():
        for kw in keywords:
            if _norm(kw) in fee_text:
                suffix = _REFERENCE_PRODUCT_SUFFIX.get(product_id or "", "")
                if suffix:
                    return f"{schedule_name}.{suffix}"
                return schedule_name
    return None


def _resolve_reference(
    reference: str,
    rules: list[TransactionFeeRule],
) -> ResolvedRate | None:
    """Resolve a textual reference to a concrete percentage and schedule names."""
    # References may be qualified with a product suffix, e.g. "online_card_payments.advanced".
    target_id: str
    if "." in reference:
        _, suffix = reference.split(".", 1)
        target_id = _REFERENCE_SUFFIX_TO_PRODUCT.get(suffix, suffix)
    else:
        target_id = reference

    for rule in rules:
        if rule.id == target_id and rule.percentage is not None:
            return ResolvedRate(
                percentage=rule.percentage,
                fixed_fee_schedule=rule.fixed_fee_schedule,
                international_surcharge_schedule=rule.international_surcharge_schedule,
            )
    # Fallback: find a rule whose label matches the reference product aliases.
    aliases = _PRODUCT_ALIASES.get(target_id, ())
    for rule in rules:
        if rule.label and any(_norm(a) in _norm(rule.label) for a in aliases):
            return ResolvedRate(
                percentage=rule.percentage,
                fixed_fee_schedule=rule.fixed_fee_schedule,
                international_surcharge_schedule=rule.international_surcharge_schedule,
            )
    return None


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


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
        document_id=table.document_id,
        component_id=table.component_id,
        table_id=table.table_id,
        row_id=row.row_id,
        row_index=row_index,
        section_heading=section_heading or (table.section_path[-1] if table.section_path else table.caption),
        original_label=original_label,
        classifier_version=_CLASSIFIER_VERSION,
    )


# ---------------------------------------------------------------------------
# Core classifier
# ---------------------------------------------------------------------------


def _parse_rate_expression(fee_text: str) -> tuple[str | None, str | None]:
    """Parse a German/English percentage + fixed-fee expression.

    Returns (percentage, fixed_fee_currency_amount_text).
    """
    import re as _re

    pct: str | None = None
    # Find a percentage token anywhere in the text.
    for match in _re.finditer(r"([0-9]+(?:[.,][0-9]+)?)\s*%", fee_text):
        pct = normalize_decimal_string(match.group(1))
        break
    # Money amount is everything after the plus/extra token, if present.
    fixed: str | None = None
    plus_match = _re.search(r"[+]\s*(.+)", fee_text)
    if plus_match:
        fixed = plus_match.group(1).strip()
    return pct, fixed


@dataclass(frozen=True)
class _ExtractedRule:
    product_id: str
    label: str
    percentage: str | None
    default_schedule: str
    table: Table
    row: Row
    row_index: int
    reference: str | None = None


def _extract_rules_from_rate_table(
    table: Table,
    table_category: str,
    source: Source | None,
) -> tuple[list[_ExtractedRule], list[UnclassifiedFeeRow], list[AmbiguousFeeRow]]:
    rules: list[_ExtractedRule] = []
    unclassified: list[UnclassifiedFeeRow] = []
    ambiguous: list[AmbiguousFeeRow] = []
    default_schedule = _TABLE_CATEGORY_SCHEDULE.get(table_category, "commercial")

    for idx, row in enumerate(table.rows):
        label = _row_label(row)
        if not label:
            continue
        product_id, ambiguous_candidates = _classify_product(label)
        if ambiguous_candidates:
            ambiguous.append(
                AmbiguousFeeRow(
                    normalized_cells=_row_cells_text(row),
                    original_label=label,
                    source=_provenance(table, row, idx, source, original_label=label),
                    candidates=ambiguous_candidates,
                )
            )
            continue
        if product_id is None:
            # Rate tables sometimes contain header-like rows; skip empty/short rows.
            if len(label) > 3 and _row_has_percentage(row):
                unclassified.append(
                    UnclassifiedFeeRow(
                        normalized_cells=_row_cells_text(row),
                        original_label=label,
                        source=_provenance(table, row, idx, source, original_label=label),
                        reason="no product alias matched",
                    )
                )
            continue
        fee_text = _row_fee_cell(row)
        pct, _fixed = _parse_rate_expression(fee_text)
        reference = _detect_reference(row, product_id)
        rules.append(
            _ExtractedRule(
                product_id=product_id,
                label=label,
                percentage=pct,
                default_schedule=_override_schedule(product_id, default_schedule),
                table=table,
                row=row,
                row_index=idx,
                reference=reference,
            )
        )
    return rules, unclassified, ambiguous


def _override_schedule(product_id: str, default: str) -> str:
    """Some products always carry their own schedule name."""
    overrides = {
        "goods_and_services": "goods_and_services",
        "donations": "donations",
        "nonprofit": "nonprofit",
        "micropayments": "micropayments",
        "alternative_payment_methods": "alternative_payment_methods",
        "advanced_card_payments": "online_card_payments",
        "pos_transactions": "pos_transactions",
        "qr_code_payments": "commercial",
        "guest_checkout": "commercial",
        "invoice_pay_later": "commercial",
        "other_commercial": "commercial",
        "paypal_checkout": "commercial",
    }
    return overrides.get(product_id, default)


# ---------------------------------------------------------------------------
# Schedule assembly
# ---------------------------------------------------------------------------


def _collect_schedules(
    tables: list[Table],
) -> tuple[dict[str, FixedFeeSchedule], dict[str, InternationalSurchargeSchedule]]:
    fixed: dict[str, FixedFeeSchedule] = {}
    international: dict[str, InternationalSurchargeSchedule] = {}
    for table in tables:
        category = _classify_table_category(table)
        if category == "fixed_fee_table":
            schedule = _extract_fixed_fee_schedule(table)
            if schedule:
                name = _schedule_name_from_table(table, "commercial")
                fixed[name] = schedule
        elif category == "international_surcharge_table":
            schedule = _extract_international_surcharge_schedule(table)
            if schedule:
                name = _schedule_name_from_table(table, "commercial")
                international[name] = schedule
    return fixed, international


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def _derive_status(
    rules: list[TransactionFeeRule],
    unclassified: list[UnclassifiedFeeRow],
    ambiguous: list[AmbiguousFeeRow],
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
) -> str:
    if not rules:
        return "unclassified"
    if ambiguous or unclassified:
        return "partial"
    # A complete result should expose the core commercial rules for a market.
    has_commercial = any(r.id in {"paypal_checkout", "goods_and_services", "other_commercial"} for r in rules)
    if has_commercial and bool(fixed_schedules):
        return "complete"
    return "partial"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_tables(tables: list[Table], source: Source | None = None) -> DerivedFeeResult:
    """Derive product-specific transaction fee rules from normalized tables."""
    fixed_schedules, international_schedules = _collect_schedules(tables)

    extracted_rules: list[_ExtractedRule] = []
    unclassified_rows: list[UnclassifiedFeeRow] = []
    ambiguous_rows: list[AmbiguousFeeRow] = []

    for table in tables:
        category = _classify_table_category(table)
        if category in _TABLE_CATEGORY_SCHEDULE or category in {
            "commercial_rate_table",
            "online_card_rate_table",
            "goods_and_services_rate_table",
            "donation_rate_table",
            "nonprofit_rate_table",
            "apm_rate_table",
            "pos_rate_table",
            "micropayment_rate_table",
        }:
            rules, uncls, ambig = _extract_rules_from_rate_table(table, category, source)
            extracted_rules.extend(rules)
            unclassified_rows.extend(uncls)
            ambiguous_rows.extend(ambig)

    # First pass: build TransactionFeeRule objects without resolving references so
    # that all candidate target rules exist for the second pass.
    unresolved_rules: list[TransactionFeeRule] = []
    for extracted in extracted_rules:
        schedule = extracted.default_schedule
        unresolved_rules.append(
            TransactionFeeRule(
                id=extracted.product_id,
                label=extracted.label,
                percentage=extracted.percentage,
                fixed_fee_schedule=schedule,
                international_surcharge_schedule=schedule,
                rate_reference=None,
                source=_provenance(
                    extracted.table,
                    extracted.row,
                    extracted.row_index,
                    source,
                    original_label=extracted.label,
                ),
            )
        )

    # Second pass: resolve textual references against all collected rules.
    for extracted in extracted_rules:
        if not extracted.reference:
            continue
        # Update the matching rule with the resolved reference.
        for rule in unresolved_rules:
            if rule.source == _provenance(
                extracted.table,
                extracted.row,
                extracted.row_index,
                source,
                original_label=extracted.label,
            ):
                resolved = _resolve_reference(extracted.reference, unresolved_rules)
                percentage = rule.percentage
                if resolved and resolved.percentage and percentage is None:
                    percentage = resolved.percentage
                new_rule = rule.model_copy(
                    update={
                        "rate_reference": RateReference(
                            reference=extracted.reference, resolved_rate=resolved
                        ),
                        "percentage": percentage,
                    }
                )
                idx = unresolved_rules.index(rule)
                unresolved_rules[idx] = new_rule
                break

    # Deduplicate by product id: prefer the first rule with a rate (or reference),
    # which for Germany is typically the commercial rate-table row.
    seen_ids: set[str] = set()
    transaction_rules: list[TransactionFeeRule] = []
    for rule in unresolved_rules:
        if rule.id in seen_ids:
            continue
        seen_ids.add(rule.id)
        transaction_rules.append(rule)

    # Stable ordering: by product ID order, then by label.
    order = {pid: idx for idx, pid in enumerate(_PRODUCT_ORDER)}
    transaction_rules.sort(key=lambda r: (order.get(r.id, 999), r.label or ""))

    # Currency conversion.
    currency_conversion = None
    for table in tables:
        if _classify_table_category(table) == "currency_conversion_table":
            for row in table.rows:
                pct = _first_percentage(row)
                if pct:
                    currency_conversion = CurrencyConversion(spread_percentage=pct)
                    break
            if currency_conversion:
                break

    status = _derive_status(
        transaction_rules,
        unclassified_rows,
        ambiguous_rows,
        fixed_schedules,
        international_schedules,
    )

    return DerivedFeeResult(
        status=status,
        transaction_fee_rules=transaction_rules,
        fixed_fee_schedules=fixed_schedules,
        international_surcharge_schedules=international_schedules,
        currency_conversion=currency_conversion,
        unclassified_fee_rows=unclassified_rows,
        ambiguous_rows=ambiguous_rows,
    )
