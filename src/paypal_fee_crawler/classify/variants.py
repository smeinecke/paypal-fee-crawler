from __future__ import annotations

import logging
import re
from collections.abc import Callable

from ..models import (
    Table,
)
from .apm import _is_domestic_label, _is_international_label
from .conditions import _is_charity_label, _is_generic_apm_label, _is_generic_other_commercial_label
from .patterns import (
    _ADVANCED_CARD_VARIANTS,
    _APM_SPECIAL_METHOD_IDS,
    _APM_VARIANTS,
    _CARD_VERIFICATION_VARIANTS,
    _DIRECT_FIXED_FEE_PRODUCTS,
    _DONATIONS_VARIANTS,
    _FRAUD_PROTECTION_VARIANTS,
    _INVOICE_VARIANTS,
    _MICROPAYMENT_VARIANTS,
    _NONPROFIT_VARIANTS,
    _OTHER_COMMERCIAL_VARIANTS,
    _PAY_LATER_VARIANTS,
    _PAYPAL_CHECKOUT_VARIANTS,
    _POS_VARIANTS,
    _QR_ABOVE_THRESHOLD,
    _QR_BELOW_THRESHOLD,
    _RECORDS_REQUEST_VARIANTS,
    _SEPA_DIRECT_DEBIT_VARIANTS,
    _VARIANT_RULES_BY_PRODUCT,
    _WITHDRAWAL_VARIANTS,
)
from .text_utils import _all_variant_matches, _first_variant_match, _keyword_match, _norm, _table_text

logger = logging.getLogger(__name__)


def _is_sending_donation_table(table_text: str) -> bool:
    return _keyword_match(table_text, ("sending", "senden", "envoi", "envío", "invio", "wysyłka"), word_boundary=False)


def _applicable_variants_for_table(table: Table, base_name: str) -> list[str]:
    """Return the variant ids explicitly named in a schedule table caption."""
    rules = _VARIANT_RULES_BY_PRODUCT.get(base_name)
    if not rules:
        return []
    text = _table_text(table)
    return _all_variant_matches(text, rules)


type _VariantRules = tuple[tuple[tuple[str, ...], str], ...]


def _resolve_variant_lookup(
    product_id: str,
    label: str,
    norm_label: str,
    table_text: str,
    combined: str,
    methods: list[str],
    is_intl: bool,
    is_dom: bool,
) -> str | None:
    """Resolve a variant from a static keyword table, trying fields in order."""
    fields, rules, default = _SIMPLE_VARIANT_LOOKUPS[product_id]
    text_by_field = {
        "label": label,
        "norm_label": norm_label,
        "table_text": table_text,
        "combined": combined,
    }
    for field in fields:
        variant = _first_variant_match(text_by_field[field], rules)
        if variant:
            return variant
    return default


_SIMPLE_VARIANT_LOOKUPS: dict[str, tuple[tuple[str, ...], _VariantRules, str | None]] = {
    "pos_transactions": (("norm_label",), _POS_VARIANTS, "standard"),
    "invoice_pay_later": (("norm_label",), _INVOICE_VARIANTS, "standard"),
    "pay_later_consumer": (("norm_label",), _PAY_LATER_VARIANTS, "standard"),
    "sepa_direct_debit": (("combined",), _SEPA_DIRECT_DEBIT_VARIANTS, None),
    "fraud_protection": (("combined",), _FRAUD_PROTECTION_VARIANTS, "advanced"),
    "records_request": (("combined",), _RECORDS_REQUEST_VARIANTS, "standard"),
    "card_verification": (("combined",), _CARD_VERIFICATION_VARIANTS, "standard"),
    "withdrawals": (("norm_label", "combined"), _WITHDRAWAL_VARIANTS, "standard"),
    "other_commercial": (("norm_label",), _OTHER_COMMERCIAL_VARIANTS, "standard"),
}


def _variant_for_apm(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    if any(m in _APM_SPECIAL_METHOD_IDS for m in methods):
        return "special"
    variant = _first_variant_match(norm_label, _APM_VARIANTS)
    if variant:
        return variant
    if _is_generic_apm_label(label):
        return "default"
    return "default"


def _variant_for_advanced_card(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    if _is_charity_label(combined):
        return "donations"
    if "american express" in norm_label or "americanexpress" in norm_label or "amex" in norm_label:
        return "american_express"
    return _first_variant_match(norm_label, _ADVANCED_CARD_VARIANTS) or "standard"


def _variant_for_qr_code(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    if _first_variant_match(norm_label, _QR_BELOW_THRESHOLD):
        return "below_threshold"
    if _first_variant_match(norm_label, _QR_ABOVE_THRESHOLD):
        return "above_threshold"
    return "standard"


def _variant_for_micropayments(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    variant = _first_variant_match(norm_label, _MICROPAYMENT_VARIANTS)
    if variant:
        return variant
    if is_intl:
        return "international"
    if is_dom:
        return "domestic"
    return "standard"


def _variant_for_paypal_checkout(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    if _is_charity_label(label):
        return "donations"
    parts = re.split(r"[^a-z0-9]+", norm_label)
    if "venmo" in parts and not any(
        i > 0 and parts[i - 1] in {"or", "and", "ou"} for i in range(1, len(parts)) if parts[i] == "venmo"
    ):
        return "venmo"
    variant = _first_variant_match(norm_label, _PAYPAL_CHECKOUT_VARIANTS)
    if variant:
        return variant
    if is_intl:
        return "international"
    if is_dom:
        return "domestic"
    return "standard"


def _variant_for_other_commercial(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    if _is_generic_other_commercial_label(label):
        return "standard"
    return _resolve_variant_lookup(
        "other_commercial", label, norm_label, table_text, combined, methods, is_intl, is_dom
    )


def _variant_for_pos_transactions(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    return _resolve_variant_lookup(
        "pos_transactions", label, norm_label, table_text, combined, methods, is_intl, is_dom
    )


def _variant_for_donations(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    if _is_sending_donation_table(table_text):
        return "sending"
    unlisted_keywords = ("nicht aufgeführte", "unlisted", "non listée", "non listados", "non listate")
    if _keyword_match(norm_label, unlisted_keywords, word_boundary=True):
        return "campaign_unlisted"
    return _first_variant_match(norm_label, _DONATIONS_VARIANTS) or "standard"


def _variant_for_nonprofit(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    variant = _first_variant_match(norm_label, _NONPROFIT_VARIANTS)
    if variant:
        return variant
    if "interchange" in norm_label:
        if "++" in norm_label or "plus plus" in norm_label or "interchange plus plus" in table_text:
            return "interchange_plus_plus"
        return "interchange_plus"
    return _first_variant_match(norm_label, _ADVANCED_CARD_VARIANTS) or "standard"


def _variant_for_invoice_pay_later(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    return _resolve_variant_lookup(
        "invoice_pay_later", label, norm_label, table_text, combined, methods, is_intl, is_dom
    )


def _variant_for_pay_later_consumer(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    return _resolve_variant_lookup(
        "pay_later_consumer", label, norm_label, table_text, combined, methods, is_intl, is_dom
    )


def _variant_for_withdrawals(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    return _resolve_variant_lookup("withdrawals", label, norm_label, table_text, combined, methods, is_intl, is_dom)


def _variant_for_sepa_direct_debit(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    return _resolve_variant_lookup(
        "sepa_direct_debit", label, norm_label, table_text, combined, methods, is_intl, is_dom
    )


def _variant_for_fraud_protection(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    return _resolve_variant_lookup(
        "fraud_protection", label, norm_label, table_text, combined, methods, is_intl, is_dom
    )


def _variant_for_records_request(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    return _resolve_variant_lookup("records_request", label, norm_label, table_text, combined, methods, is_intl, is_dom)


def _variant_for_card_verification(
    label: str, norm_label: str, table_text: str, combined: str, methods: list[str], is_intl: bool, is_dom: bool
) -> str | None:
    return _resolve_variant_lookup(
        "card_verification", label, norm_label, table_text, combined, methods, is_intl, is_dom
    )


_VARIANT_DISPATCH: dict[str, Callable[..., str | None]] = {
    "alternative_payment_methods": _variant_for_apm,
    "advanced_card_payments": _variant_for_advanced_card,
    "qr_code_payments": _variant_for_qr_code,
    "micropayments": _variant_for_micropayments,
    "paypal_checkout": _variant_for_paypal_checkout,
    "other_commercial": _variant_for_other_commercial,
    "pos_transactions": _variant_for_pos_transactions,
    "donations": _variant_for_donations,
    "nonprofit": _variant_for_nonprofit,
    "invoice_pay_later": _variant_for_invoice_pay_later,
    "pay_later_consumer": _variant_for_pay_later_consumer,
    "withdrawals": _variant_for_withdrawals,
    "sepa_direct_debit": _variant_for_sepa_direct_debit,
    "fraud_protection": _variant_for_fraud_protection,
    "records_request": _variant_for_records_request,
    "card_verification": _variant_for_card_verification,
}


def _variant_id_for_row(
    product_id: str,
    label: str,
    methods: list[str],
    table: Table | None = None,
    fee_text: str | None = None,
) -> str | None:
    """Return a stable variant id for a row, if needed."""
    norm_label = _norm(label)
    table_text = _table_text(table) if table else ""
    combined = norm_label + " " + table_text
    # Direct fixed-fee variants are often encoded in the fee cell text (e.g. two
    # SEPA settlement options in one cell), so include that context when it is
    # available.
    if product_id in _DIRECT_FIXED_FEE_PRODUCTS and fee_text:
        combined = combined + " " + _norm(fee_text)

    # Generic domestic/international variants are detected up-front so that
    # product-specific logic can be layered on top of them.
    is_international = _is_international_label(label)
    is_domestic = _is_domestic_label(label)

    resolver = _VARIANT_DISPATCH.get(product_id)
    if resolver:
        variant = resolver(label, norm_label, table_text, combined, methods, is_international, is_domestic)
        if variant is not None:
            return variant

    if is_international:
        return "international"
    if is_domestic:
        return "domestic"
    return None
