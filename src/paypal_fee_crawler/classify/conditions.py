from __future__ import annotations

import logging
import re
from typing import Any

from ..models import (
    Table,
)
from ..normalize import normalize_decimal_string
from .apm import _extract_apm_methods, _is_domestic_label, _is_international_label
from .text_utils import _keyword_match, _norm, _table_text

logger = logging.getLogger(__name__)


def _is_charity_label(label: str) -> bool:
    """Return True if the label/table text indicates a charity/donation context."""
    text = _norm(label)
    return _keyword_match(
        text,
        (
            "spende",
            "spenden",
            "donation",
            "donations",
            "donativ",
            "don de",
            "donazioni",
            "donativos",
            "caridad",
            "caridade",
            "charity",
            "charitable",
            "liefdadigheid",
            "liefdadigheids",
            "goede doel",
            "goede doelen",
            "välgörenhet",
            "jótékonysági",
            "φιλανθρωπ",
            "dons",
            "dona",
            "donação",
            "donações",
            "humanit",
            "non-profit",
            "nonprofit",
            "nonguvernamental",
        ),
        word_boundary=False,
    )


def _is_generic_other_commercial_label(label: str) -> bool:
    """Return True if the label is a generic 'all other commercial' fallback."""
    text = _norm(label)
    return _keyword_match(
        text,
        (
            "all other commercial",
            "alle anderen",
            "toutes les autres",
            "autres transactions",
            "autre transaction",
            "altre transazioni",
            "otras transacciones",
            "outras transações",
            "outros",
            "andre",
            "andere",
            "pozostałe",
            "transações comerciais",
            "business payments",
            "commercial transactions",
            "other commercial",
            "transacciones comerciales",
            "transactions commerciales",
        ),
        word_boundary=False,
    )


def _is_generic_apm_label(label: str) -> bool:
    """Return True if the label is a generic 'all other APM' fallback."""
    text = _norm(label)
    return _keyword_match(
        text,
        (
            "all other alternative payment",
            "alle anderen alternativen",
            "tous les autres moyens",
            "tous les autres modes",
            "tutti gli altri metodi",
            "todos los otros métodos",
            "todos os outros métodos",
            "alternative payment method",
            "alternative payment methods",
            "alternative zahlungsmethode",
            "alternative zahlungsmethoden",
            "autre moyen de paiement",
            "autres moyens de paiement",
            "autre mode de paiement",
            "metodo di pagamento alternativo",
            "metodi di pagamento alternativi",
            "método de pago alternativo",
            "métodos de pago alternativos",
            "andere betaalmethode",
            "andere betaalmethoden",
            "alternative betalingsmetode",
            "alternative betalingsmetoder",
            "alternative betaalmethode",
            "alternative betaalmethoden",
            "all other apm",
            "alle anderen apm",
            "apm",
            "abm",
        ),
        word_boundary=False,
    )


def _extract_country_group_condition(label: str) -> dict[str, Any] | None:
    """Parse a row label like 'AG, BB, BM & SA' into a list of market codes.

    Returns an applies_to_markets condition if the label contains market codes.
    Generic default phrases are returned as the special 'all_other_markets' code.
    """
    text = label
    default_phrases = (
        "all other markets",
        "all other",
        "todos os outros",
        "tous les autres",
        "tutti gli altri",
        "alle anderen",
        "alla andra",
        "alle andre",
        "todos los demás",
        "overige",
        "pozostałe",
        "pozostale",
        "pozostalých",
        "egyeb",
        "altri",
        "sonstige",
    )
    if _keyword_match(_norm(text), default_phrases, word_boundary=False):
        return {"applies_to_markets": ["all_other_markets"]}
    # Look for 2-character uppercase market codes separated by commas,
    # ampersands, 'and' or whitespace. A 2-char code may be followed by
    # punctuation, not by another letter.
    matches = re.findall(r"(?<![A-Za-z0-9])([A-Z]{2})(?![A-Za-z0-9])", text)
    # Filter out a few common false positives.
    codes = [m for m in matches if m not in {"QR", "ON"}]
    if not codes:
        return None
    return {"applies_to_markets": sorted(set(codes))}


def _pricing_plan_for_label(label: str) -> str | None:
    """Detect blended/standard/interchange pricing plan from a row label."""
    text = _norm(label)
    if "interchange plus plus" in text or "interchange++" in text:
        return "interchange_plus_plus"
    if "interchange plus" in text:
        return "interchange_plus"
    if _keyword_match(
        text,
        (
            "blended",
            "regroupée",
            "regroup",
            "flat rate",
            "forfait",
            "misto",
            "tariffario misto",
            "piano tariffario misto",
            "blandad prissättning",
            "combinada",
            "combinado",
            "tarifa combinada",
            "gecombineerde",
            "gecombineerd tarief",
            "kombinovanými sazbami",
            "kombinovanými sadzbami",
        ),
        word_boundary=False,
    ):
        return "blended"
    if "standard paypal payment" in text:
        return "standard"
    return None


def _card_payment_methods_from_label(label: str) -> list[str] | None:
    """Extract card brand names listed in an advanced card row label."""
    text = _norm(label)
    methods: list[str] = []
    for keyword, method_id in (
        ("visa", "visa"),
        ("mastercard", "mastercard"),
        ("maestro", "maestro"),
        ("china unionpay", "china_union_pay"),
        ("cup", "china_union_pay"),
        ("diners", "diners"),
        ("discover", "discover"),
        ("jcb", "jcb"),
        ("cofidis", "cofidis"),
        ("cetelem", "cetelem"),
        ("cofinoga", "cofinoga"),
        ("carte bancaire", "carte_bancaire"),
        ("debit card", "debit_card"),
        ("credit card", "credit_card"),
        ("other card", "other_card"),
    ):
        if _keyword_match(text, (keyword,), word_boundary=True) and method_id not in methods:
            methods.append(method_id)
    return methods if methods else None


def _service_for_donation_label(label: str) -> str | None:
    """Map a donation row label to its underlying service, if any."""
    text = _norm(label)
    if _keyword_match(
        text,
        (
            "website payments pro",
            "payments pro",
            "solution hébergée",
            "paypal pro",
            "pagamenti con paypal pro",
            "hosted solution",
        ),
        word_boundary=False,
    ):
        return "website_payments_pro"
    if _keyword_match(
        text,
        ("virtual terminal", "eterminal", "e-terminal", "pagamenti telefonici", "telefonici"),
        word_boundary=False,
    ):
        return "virtual_terminal"
    if _keyword_match(
        text,
        (
            "advanced credit",
            "advanced debit",
            "avancerat kredit",
            "carte bancaire avancés",
            "pagamenti avanzati con carta",
            "avancerade betalningar med betalkort",
            "avancerat kredit- och betalkort",
        ),
        word_boundary=False,
    ):
        return "advanced_card"
    return None


def _conditions_for_apm(
    conditions: dict[str, Any],
    label: str,
    methods: list[str] | None,
    variant_id: str | None,
) -> None:
    """Populate conditions for an alternative payment methods row."""
    if methods is None:
        methods, _ = _extract_apm_methods(label)
    if methods:
        conditions["payment_methods"] = sorted(methods)
    if variant_id == "third_party_wallet":
        conditions["payment_methods"] = ["third_party_wallet"]
    if variant_id == "fx_service":
        conditions["service"] = "foreign_exchange"


def _conditions_for_donations(
    conditions: dict[str, Any],
    product_id: str,
    label: str,
) -> None:
    """Populate conditions for a donations-related row."""
    conditions["transaction_purpose"] = "donation"
    if product_id in ("advanced_card_payments", "nonprofit"):
        service = _service_for_donation_label(label)
        if service:
            conditions["service"] = service


def _service_for_advanced_card(label: str, variant_id: str) -> str | None:
    """Return the service condition for advanced-card rows, if any."""
    if variant_id == "fx_service":
        text = _norm(label)
        if "spread" in text:
            return "fx_spread"
        if "as a service" in text:
            return "fx_as_a_service"
    return None


def _conditions_for_advanced_card(
    conditions: dict[str, Any],
    label: str,
    variant_id: str,
) -> None:
    """Populate conditions for an advanced card or nonprofit card row."""
    if variant_id == "eterminal":
        conditions["authorization_channel"] = "terminal"
        conditions["point_of_sale"] = True
    if variant_id.startswith("interchange_plus"):
        conditions["pricing_plan"] = variant_id
    else:
        plan = _pricing_plan_for_label(label)
        if plan:
            conditions["pricing_plan"] = plan
    service = _service_for_advanced_card(label, variant_id)
    if service:
        conditions["service"] = service
    if variant_id == "american_express":
        conditions["payment_methods"] = ["american_express"]
    else:
        card_methods = _card_payment_methods_from_label(label)
        if card_methods:
            conditions["payment_methods"] = card_methods


def _conditions_for_pos(
    conditions: dict[str, Any],
    label: str,
    variant_id: str,
) -> None:
    """Populate conditions for a POS transaction row."""
    if variant_id == "card_present":
        conditions["card_present"] = True
        conditions["point_of_sale"] = True
        conditions["authorization_channel"] = "terminal"
    elif variant_id == "manual_entry":
        conditions["card_present"] = False
        conditions["authorization_channel"] = "manual"
    elif variant_id == "qr_code":
        conditions["payment_methods"] = ["qr_code"]
        conditions["point_of_sale"] = True
    elif variant_id == "payment_links":
        text = _norm(label)
        if _keyword_match(text, ("paypal checkout", "venmo", "pay later", "guest checkout"), word_boundary=False):
            conditions["payment_methods"] = sorted(["paypal_checkout", "venmo", "pay_later", "guest_checkout"])
        elif _keyword_match(
            text,
            ("standard credit", "debit card", "apple pay", "third-party wallets", "third party wallets"),
            word_boundary=False,
        ):
            conditions["payment_methods"] = sorted(["card", "apple_pay", "third_party_wallet"])


def _conditions_for_paypal_checkout(
    conditions: dict[str, Any],
    variant_id: str,
) -> None:
    """Populate conditions for a PayPal Checkout row."""
    if variant_id == "venmo":
        conditions["payment_methods"] = ["venmo"]
    elif variant_id == "crypto":
        conditions["payment_methods"] = ["cryptocurrency"]


def _service_for_other_commercial_ach(table: Table | None) -> str | None:
    """Return the service indicated by an other-commercial ACH table heading."""
    if not table:
        return None
    table_text = _norm(_table_text(table))
    if "invoic" in table_text:
        return "invoicing"
    if "online" in table_text and ("card" in table_text or "payment" in table_text):
        return "online_payments"
    return None


def _conditions_for_other_commercial(
    conditions: dict[str, Any],
    label: str,
    variant_id: str,
    table: Table | None,
) -> None:
    """Populate conditions for an other-commercial row."""
    if variant_id == "pyusd":
        conditions["pricing_plan"] = "pyusd"
    elif variant_id == "ach":
        conditions["payment_methods"] = ["ach"]
        service = _service_for_other_commercial_ach(table)
        if service:
            conditions["service"] = service
    elif variant_id == "card_funded":
        conditions["funding_source"] = "card"


def _transaction_region_for_variant(
    label: str,
    variant_id: str | None,
    table: Table | None,
) -> str | None:
    """Infer a transaction_region from the variant, row label, or table caption."""
    if variant_id in ("domestic", "international"):
        return variant_id
    if variant_id in ("crypto", "digital_goods"):
        if _is_international_label(label):
            return "international"
        if _is_domestic_label(label):
            return "domestic"
        return None
    if _is_international_label(label) and not _is_domestic_label(label):
        return "international"
    if _is_domestic_label(label) and not _is_international_label(label):
        return "domestic"
    if table:
        table_text = _table_text(table)
        if _is_international_label(table_text) and not _is_domestic_label(table_text):
            return "international"
        if _is_domestic_label(table_text) and not _is_international_label(table_text):
            return "domestic"
    return None


def _conditions_for_row(
    product_id: str,
    variant_id: str | None,
    label: str,
    methods: list[str] | None = None,
    table: Table | None = None,
) -> dict[str, Any]:
    """Return calculable conditions for a product rule based on the source row."""
    conditions: dict[str, Any] = {}
    if product_id == "nonprofit":
        conditions["merchant_approval_required"] = True
    if product_id == "alternative_payment_methods":
        _conditions_for_apm(conditions, label, methods, variant_id)
    if product_id == "donations" or variant_id == "donations":
        _conditions_for_donations(conditions, product_id, label)
    if product_id in ("advanced_card_payments", "nonprofit") and variant_id:
        _conditions_for_advanced_card(conditions, label, variant_id)
    if product_id == "pos_transactions" and variant_id:
        _conditions_for_pos(conditions, label, variant_id)
    if product_id == "paypal_checkout" and variant_id:
        _conditions_for_paypal_checkout(conditions, variant_id)
    if product_id == "other_commercial" and variant_id:
        _conditions_for_other_commercial(conditions, label, variant_id, table)
    if product_id == "withdrawals" and variant_id and variant_id != "standard":
        conditions["withdrawal_method"] = variant_id
    region = _transaction_region_for_variant(label, variant_id, table)
    if region:
        conditions["transaction_region"] = region
    market_condition = _extract_country_group_condition(label)
    if market_condition:
        conditions.update(market_condition)
    if product_id == "qr_code_payments" and variant_id:
        amount_condition = _extract_amount_condition(label)
        if amount_condition:
            conditions["amount"] = amount_condition
    return conditions


def _maximum_fee_schedule_for_conditions(conditions: dict[str, Any]) -> str | None:
    """Map withdrawal/payout conditions to the corresponding max-fee schedule."""
    if conditions.get("transaction_region") == "international":
        return "payouts_international"
    if conditions.get("transaction_region") == "domestic":
        return "payouts_domestic"
    if conditions.get("applies_to_markets") == ["US"]:
        return "payouts_us"
    return None


def _extract_amount_condition(label: str) -> dict[str, Any] | None:
    """Parse a threshold expression like 'below 10.00 EUR' into a condition."""
    text = _norm(label)
    operators = {
        "<": "lt",
        "<=": "lte",
        ">": "gt",
        ">=": "gte",
        "under": "lt",
        "below": "lt",
        "less than": "lt",
        "unter": "lt",
        "bis zu": "lt",
        "up to": "lt",
        "jusqu'à": "lt",
        "inférieure": "lt",
        "inférieures": "lt",
        "inferior": "lt",
        "über": "gt",
        "over": "gt",
        "above": "gt",
        "greater than": "gt",
        "mindestens": "gt",
        "at least": "gt",
        "à partir de": "gt",
        "supérieure": "gt",
        "supérieures": "gt",
        "supérieure ou égale": "gte",
        "supérieures ou égales": "gte",
        "superior": "gt",
    }
    for op_token, op in operators.items():
        # Match the operator token followed by a number and optional currency.
        pattern = re.escape(op_token) + r"\s+([0-9]+(?:[.,][0-9]+)?)\s*([A-Za-z]{3})?"
        match = re.search(pattern, text)
        if match:
            value = normalize_decimal_string(match.group(1))
            currency = match.group(2)
            result: dict[str, Any] = {"operator": op, "value": value}
            if currency:
                result["currency"] = currency.upper()
            return result
    return None
