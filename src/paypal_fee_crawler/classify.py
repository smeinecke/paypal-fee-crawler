"""Derive product-specific transaction fee rules from normalized PayPal tables.

The classifier works at the row level: a single PayPal table may contain several
independent payment products, and each relevant fee row becomes a separate
``TransactionFeeRule``.  Fixed-fee and international-surcharge schedules are kept
separate per product or product family so that an HTTP fee calculator can select
the schedule that applies to a given rule.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .models import (
    AmbiguousFeeRow,
    CoverageSummary,
    CurrencyConversion,
    DerivedFeeResult,
    Diagnostic,
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
        "betal med paypal",
        "betaling via paypal",
        "paypal betaling",
        "paypal betal",
        "betal med venmo",
        "betaling med venmo",
        "venmo",
        "pay with paypal",
        "paga con paypal",
        "betalen met paypal",
    ),
    "goods_and_services": (
        "sending and receiving money for goods and services",
        "geld für waren und dienstleistungen senden/empfangen",
        "geld für waren und dienstleistungen",
        "waren und dienstleistungen",
        "goods and services",
        "goods & services",
        "goods or services",
        "varer og tjenesteydelser",
        "varer og tjenester",
        "bienes y servicios",
        "beni e servizi",
        "goederen en diensten",
        "produkter och tjänster",
        "produkter og tjenester",
    ),
    "advanced_card_payments": (
        "advanced credit and debit card payments",
        "erweiterte kredit- und debitkartenzahlungen",
        "zahlungen mit kredit- und debitkarten mit erweiterten funktionen",
        "kredit- und debitkarten mit erweiterten funktionen",
        "advanced card",
        "erweiterte kartenzahlung",
        "kredit- og betalingskort",
        "kredit- og debitkort",
        "kredit- och debitkort",
        "kreditkort",
        "betalingskort",
        "kortbetalinger",
        "avancerede kortbetalinger",
        "credit and debit card",
        "credit/debit card",
        "tarjetas de crédito y débito",
    ),
    "other_commercial": (
        "all other commercial transactions",
        "alle anderen geschäftlichen transaktionen",
        "sonstige gewerbliche transaktionen",
        "other commercial",
        "sonstige geschäftliche",
        "other commercial transactions",
        "commercial transactions",
        "erhvervsbetalinger",
        "erhverv",
        "øvrige erhvervsbetalinger",
        "alle andre erhvervsbetalinger",
        "andre erhvervsbetalinger",
        "forretningsbetalinger",
        "business payments",
        "øvrige forretningsbetalinger",
        "alle anderen",
        "andere commercial",
        "altre transazioni commerciali",
        "otras transacciones comerciales",
        "autres transactions commerciales",
        "andre forretningstransaksjoner",
        "pozostałe transakcje handlowe",
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
        "abm",
        "alternativ betalingsmetode",
        "alternative betalingsmetode",
        "alternative betaalmethode",
        "métodos de pago alternativos",
        "metodi di pagamento alternativi",
        "metodo di pagamento alternativo",
        "online bank transfer",
        "online bankoverførsel",
        "bankoverførsel",
        "bank transfer",
    ),
    "guest_checkout": (
        "zahlung eines nutzers unserer bedingungen für zahlungen ohne paypal-konto",
        "zahlungen ohne paypal-konto",
        "payments without a paypal account",
        "zahlung ohne paypal-konto",
        "guest checkout",
        "betalinger uden en paypal-konto",
        "uden en paypal-konto",
        "uden paypal-konto",
        "betal uden paypal",
        "betalning utan paypal-konto",
        "betalning utan paypal",
        "betaling uten paypal-konto",
        "betalen zonder paypal-account",
        "payer sans compte paypal",
        "pagar sin cuenta de paypal",
        "pagamento senza conto paypal",
        "płatność bez konta paypal",
    ),
    "invoice_pay_later": (
        "rechnungskauf mit ratepay",
        "ratepay",
        "invoice payments",
        "pay later",
        "ratenzahlungsangebote",
        "rechnungskauf",
        "faktura",
        "fakturabetaling",
        "fakturabetalning",
        "invoice payment",
    ),
    "qr_code_payments": (
        "qr-code-transaktionen",
        "qr-code transactions",
        "qr-code-zahlungen",
        "qr code transactions",
        "qr-code",
        "qr code",
        "qr-code-betalinger",
        "qr kode betalinger",
        "qr kode-betalinger",
        "qr-code-betaling",
        "qr-kode-betalinger",
        "qr-kode",
        "qr kode",
    ),
    "donations": (
        "paypal-spendenaktionen",
        "spendenaktionen",
        "spende",
        "donation",
        "donationer",
        "donations",
        "charity donation",
        "don de",
        "donazioni",
    ),
    "nonprofit": (
        "gemeinnützige organisationen",
        "gemeinnützig",
        "gemeinnutzig",
        "nonprofit organisation",
        "nonprofit",
        "non-profit",
        "velgørende",
        "velgørende organisationer",
        "non-profit organisation",
        "organizaciones sin fines de lucro",
        "organizzazioni senza scopo di lucro",
        "organisasjoner",
        "organizacja non-profit",
    ),
    "micropayments": (
        "mikrozahlung",
        "micropayment",
        "kleinbetragszahlung",
        "mikrobetaling",
        "mikrobetalinger",
        "mikromaksu",
        "mikropłatność",
        "micropagos",
        "micropaiement",
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
        "betal senere",
        "afbetaling",
        "paypal pay later",
        "paypal pay later-tilbud",
        "pay later-tilbud",
        "klarna",
        "afterpay",
    ),
    "pos_transactions": (
        "point of sale",
        "paypal point of sale",
        "präsenter karte",
        "card present",
        "kortforevisning",
        "kortterminal",
        "kortterminaler",
        "kortpresent",
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
        "commercial transactions",
        "standardgebyr",
        "standardavgift",
        "standaardtarief",
        "tarifa estándar",
        "tarifa standard",
        "tarifa padrão",
        "tariffa standard",
        "tarification standard",
        "standard taxa",
        "standardtaxa",
        "standardsats",
        "standardowa stawka",
        "standardní sazba",
        "štandardná sadzba",
        "szokásos díja",
        "standard fee",
        "standard rate",
        "modtagelse af indenlandske betalinger",
        "innenlands",
        "indland",
        "innenlandske",
        "inhemska",
        "binnenland",
        "binnenlandse",
        "nationales",
        "transacciones nacionales",
        "nazionali",
        "nacionais",
        "nationella",
        "krajowych",
        "krajinskih",
        "vnitrostátních",
        "domácich",
        "belföldi",
        "εσωτερικές",
        "naționale",
        "национални",
        "nacionalnih",
        "kotimaisten",
        "domestic payments",
        "domestic",
        "commerciële",
        "commerciale",
        "commerciales",
        "kommersielle",
        "kommercielle",
        "kommersiella",
        "kommercielle",
        "kommersielle",
        "komercyjne",
        "komerčních",
        "komerčných",
        "kereskedelmi",
        "εμπορικές",
        "comerciale",
        "comercial",
        "comerciale",
        "tranzakcje",
        "transakcji",
        "transakcí",
        "transakcií",
        "transakció",
        "transakcije",
        "receiving domestic",
        "recepción de transacciones nacionales",
        "recebimento de transações nacionais",
        "ricezione di transazioni nazionali",
        "recepción de transacciones nacionales",
        "ricezione di pagamenti",
    ),
    "online_card_rate_table": (
        "paypal-dienste für online-kartenzahlungen",
        "paypal-dienste für online-zahlungen",
        "online card payments",
        "online-kartenzahlungen",
        "online card",
        "online-kortbetaling",
        "online kort",
        "online kortbetalingstjenester",
        "online-kort",
        "kortbetalingstjenester",
        "kortbetalningstjänster",
        "kortbetalning",
        "kortfinansierad",
        "online kártyás",
        "online kart",
        "online karta",
        "online kortti",
        "online kort",
        "online kartę",
        "online kort",
        "online kart",
        "online kart",
        "online card payment services",
        "online payment services",
        "online kartično",
        "online kartica",
        "servizi di pagamento con carta",
        "services de paiement par carte",
        "serviços de pagamento com cartão",
        "servicios de pago con tarjeta",
        "serviços de pagamento online",
        "tarjetas de crédito y débito",
        "kredit- och debitkort",
        "credit and debit card",
        "kredit- og betalingskort",
        "kredit- och betalkort",
        "avancerat kredit- och betalkort",
        "advanced credit and debit card",
        "ηλεκτρονικές πληρωμές",
        "ηλεκτρονικες πληρωμες",
        "υπηρεσιών paypal για ηλεκτρονικές",
        "υπηρεσιων paypal για ηλεκτρονικες",
        "ηλεκτρονικές υπηρεσίες",
        "ηλεκτρονικες υπηρεσιες",
    ),
    "goods_and_services_rate_table": (
        "geld für waren und dienstleistungen",
        "goods and services",
    ),
    "donation_rate_table": (
        "empfang von inlandsspenden",
        "donation",
        "spenden",
        "donationer",
        "donationer",
        "donations",
        "donaties",
        "donativos",
        "donativas",
        "doações",
        "donazioni",
        "donationer",
        "donationer",
        "lahjoitukset",
        "darowizn",
        "príspevkov",
        "príspevky",
        "adományok",
        "δωρεές",
        "δωρεες",
        "δωρεών",
        "δωρεων",
        "donatii",
        "дарения",
        "donacije",
        "donacijo",
        "donationer",
        "donaties",
        "příspěvky",
        "příspěvků",
        "receiving domestic donations",
        "recepción de donativos nacionales",
        "recebimento de doações domésticas",
        "ricezione di donazioni nazionali",
        "λήψη εγχώριων δωρεών",
        "ληψη εγχωριων δωρεων",
    ),
    "nonprofit_rate_table": (
        "gemeinnützige organisationen",
        "nonprofit",
        "non-profit",
        "velgørende",
        "velgørende organisationer",
        "non-profit organisation",
        "organizaciones sin fines de lucro",
        "organizaciones benéficas",
        "organizzazioni senza scopo di lucro",
        "organisasjoner",
        "organizacja non-profit",
        "organizacje charytatywne",
        "charitatívnych",
        "charitativních",
        "jótékonysági",
        "jótékonysági",
        "välgörenhetsorganisationer",
        "välgörenhet",
        "φιλανθρωπικές",
        "φιλανθρωπικά",
        "φιλανθρωπικού",
        "φιλανθρωπικου",
        "instituições de solidariedade",
        "instituição de caridade",
        "institución de solidaridad",
        "entidades sin ánimo de lucro",
        "associazioni di volontariato",
        "caritative",
        "caritat",
        "caridad",
        "caridade",
        "nonguvernamental",
        "humanitar",
        "charity",
        "charitable",
        "liefdadigheid",
        "liefdadigheids",
        "receiving domestic charity",
        "receiving charity",
        "carity transactions",
    ),
    "apm_rate_table": (
        "alternative zahlungsmethode",
        "alternative payment method",
        "apm-transaktionen",
        "apm",
        "alternativ betalingsmetode",
        "alternative betalingsmetode",
        "alternative betaalmethode",
        "métodos de pago alternativos",
        "metodi di pagamento alternativi",
        "metodo di pagamento alternativo",
        "andere betaalmethode",
        "autre moyen de paiement",
        "autre moyen",
        "autre mode de paiement",
        "alternatív fizetési",
        "alternatívny spôsob platby",
        "alternativní způsob platby",
        "alternatywna forma płatności",
        "alternatywne metody płatności",
        "alternativ betalningsmetod",
        "alternativ betalingsmåte",
        "vaihtoehtoinen maksutapa",
        "vaihtoehtoinen",
        "vmt",
        "apm-transaksjoner",
        "apm-transakcje",
        "apm-transakcje",
        "apm-transakcí",
        "apm-transakcí",
        "apm-transakciók",
        "apm-transakcije",
        "apm-transakcije",
        "alternative betalingsmetoder",
        "alternative payment methods",
        "alternative payment",
    ),
    "pos_rate_table": (
        "point of sale",
        "paypal point of sale",
        "präsenter karte",
        "kortforevisning",
        "card present",
        "point-of-sale",
        "ponto de venda",
        "punto de venta",
        "punto vendita",
        "punkty sprzedaży",
        "płatności kartą",
        "kassapääte",
        "kassapääte",
        "platební terminál",
        "platební terminály",
        "terminál",
        "kortterminal",
        "kortterminaler",
        "kortläsare",
        "pos-terminal",
        "pos terminal",
        "pos-betalinger",
        "pos transakcije",
        "pos transakcie",
        "pos-transakcije",
        "kassasystem",
        "korttransaktioner",
        "korttransaktion",
        "presentkort",
    ),
    "micropayment_rate_table": (
        "mikrozahlung",
        "micropayment",
        "mikrobetaling",
        "mikrobetalinger",
        "mikromaksu",
        "mikromaksujen",
        "mikrobetaling",
        "mikrobetalinger",
        "mikrobetaling",
        "mikrobetalingar",
        "mikropłatność",
        "mikropłatności",
        "mikrotransakcí",
        "mikrotransakcií",
        "mikrotransakciók",
        "mikrotransakcije",
        "mikrotransakcije",
        "micropagos",
        "micropagos",
        "micropaiement",
        "micropaiements",
        "micropagamentos",
        "mikrobetalning",
        "mikrobetalningar",
        "mikromaksu",
        "μικροπληρωμές",
        "μικροπληρωμες",
        "μικροπληρωμών",
        "μικροπληρωμων",
        "mikroplatby",
        "mikroplatby",
    ),
    "fixed_fee_table": (
        "festgebühr",
        "fixed fee",
        "fast gebyr",
        "fast avgift",
        "tarifa fija",
        "comisión fija",
        "comissão fixa",
        "taxa fixa",
        "vast bedrag",
        "vaste kosten",
        "tariffa fissa",
        "paušální poplatek",
        "opłata stała",
        "frais fixe",
        "commission fixe",
        "kiinteä palkkio",
        "kiinteä maksu",
        "fixný poplatok",
        "fiksna naknada",
        "fiksna provizija",
        "comision fixa",
        "σταθερή χρέωση",
        "ค่าธรรมเนียมคงที่",
        "อัตราคงที่",
        "rögzített díja",
        "rögzített díj",
        "固定費用",
    ),
    "international_surcharge_table": (
        "zusätzliche prozentuale gebühr",
        "prozentuale zusatzgebühr",
        "international surcharge",
        "additional percentage fee",
        "additional percentage-based fee",
        "additional percentage based fee",
        "additional percentage",
        "international payments",
        "international",
        "internationales",
        "internationella",
        "internationale",
        "international",
        "internasjonale",
        "international",
        "international",
        "international",
        "kansainvälisten",
        "kansainvälis",
        "international",
        "international",
        "international",
        "medzinárodné",
        "medzinárodné",
        "mezinárodní",
        "zahraniční",
        "nemzetközi",
        "διεθνείς",
        "internaționale",
        "международни",
        "međunarodne",
        "mednarodne",
        "ausland",
        "zusatzgebühr",
        "additional service",
        "service fee",
        "servicegebühr",
        "servicegebyr",
        "serviceavgift",
        "extra percentage",
        "ekstra procentbaseret gebyr",
        "yderligere procentdel af gebyr",
        "extra procentuell avgift",
        "extra procentbaserad avgift",
        "lisäprosenttimaksu",
        "prosenttiperusteinen lisäpalkkio",
        "supplément de commission",
        "sobretasa internacional",
        "tarifa adicional porcentual",
        "tarifa adicional baseada em porcentagem",
        "comissão percentual adicional",
        "tariffa percentuale aggiuntiva",
        "op een percentage gebaseerde",
        "dodatkowa opłata procentowa",
        "dodatečný procentní poplatek",
        "dodatočný percentuálny poplatok",
        "százalékos kiegészítő díja",
        "dodatni postotak",
        "dodatni odstotek",
        "πρόσθετη χρέωση βάσει ποσοστού",
        "προσθετη χρεωση βασει ποσοστου",
        "πρόσθετη χρέωση υπηρεσίας",
        "προσθετη χρεωση υπηρεσιας",
        "διεθνών",
        "διεθνων",
        "διεθνείς",
        "διεθνεις",
        "λήψη διεθνών",
        "ληψη διεθνων",
        "dodatna naknada",
        "dodatna pristojbina",
    ),
    "currency_conversion_table": (
        "währungsumrechnung",
        "umrechnung",
        "currency conversion",
        "guthaben umrechnen",
        "valuutan muuntaminen",
        "valuutanvaihto",
        "valuuttakurssit",
        "valutaomregning",
        "valuuttakurssi",
        "wisselkoers",
        "cambio",
        "cambio de divisa",
        "cambio valuta",
        "conversione",
        "conversão",
        "cambio",
        "devizni",
        "valut",
        "valuuta",
        "valuta",
        "převod",
        "prevod",
        "konverzia",
        "konverzija",
        "μετατροπή",
        "conversie",
        "conversion",
        "valutaveksling",
        "valutakurs",
        "valutakurser",
        "valutakurser",
        "kryptoměna",
        "kryptomena",
        "criptomoeda",
        "criptomoneda",
        "kryptovaluta",
        "converting balance",
        "converting",
    ),
}

# When a caption contains one of these negative signals, it is not treated as a
# rate table even if it also contains product-specific keywords.
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
        "sonstige gebühren",
        "rückbuchung",
        "chargeback",
    ),
    "online_card_rate_table": (
        "sonstige gebühren",
        "rückbuchung",
        "chargeback",
        "betrugsschutz",
        "professionelles tool",
        "professional tool",
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
            if token.kind == "number" and token.value and (
                cell_indicates_pct or _token_text_indicates_percentage(token)
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
    "paypal_checkout": "commercial_rate_table",
    "other_commercial": "commercial_rate_table",
    "guest_checkout": "commercial_rate_table",
    "invoice_pay_later": "commercial_rate_table",
    "pay_later_consumer": "commercial_rate_table",
    "qr_code_payments": "commercial_rate_table",
    "chargebacks": "online_card_rate_table",
    "refunds": "online_card_rate_table",
    "disputes": "online_card_rate_table",
    "card_verification": "apm_rate_table",
    "currency_conversion": "currency_conversion_table",
    "withdrawals": "apm_rate_table",
}


# Default product id for a rate-table category when a row label does not match
# any product alias.  This is used for rows such as "Deutschland" in the
# nonprofit table, where the table context determines the product.
_TABLE_CATEGORY_PRODUCT: dict[str, str] = {
    "nonprofit_rate_table": "nonprofit",
    "donation_rate_table": "donations",
    "micropayment_rate_table": "micropayments",
    "apm_rate_table": "alternative_payment_methods",
    "pos_rate_table": "pos_transactions",
    "online_card_rate_table": "advanced_card_payments",
    "goods_and_services_rate_table": "goods_and_services",
    "commercial_rate_table": "other_commercial",
}


def _classify_table_category(table: Table) -> str | None:
    text = _table_text(table)
    # Explicit schedule-type captions are authoritative and win over product
    # rate-table keywords such as "commercial transactions" or "donations".
    fixed_fee_keywords = (
        "festgebühr",
        "fixed fee",
        "fast gebyr",
        "fast avgift",
        "tarifa fija",
        "comisión fija",
        "comissão fixa",
        "tarifa fixa",
        "taxa fixa",
        "vast bedrag",
        "vaste kosten",
        "tariffa fissa",
        "paušální poplatek",
        "pevný poplatek",
        "opłata stała",
        "frais fixe",
        "commission fixe",
        "kiinteä palkkio",
        "kiinteä maksu",
        "fixný poplatok",
        "pevný poplatok",
        "fiksna naknada",
        "fiksna provizija",
        "comision fixa",
        "σταθερή χρέωση",
        "ค่าธรรมเนียมคงที่",
        "อัตราคงที่",
        "rögzített díja",
        "rögzített díj",
        "fix díj",
        "固定費用",
    )
    if any(kw in text for kw in fixed_fee_keywords):
        return "fixed_fee_table"
    international_surcharge_keywords = (
        "prozentuale zusatzgebühr",
        "zusätzliche prozentuale gebühr",
        "international surcharge",
        "additional percentage fee",
        "additional percentage-based fee",
        "additional percentage based fee",
        "additional percentage",
        "international payments",
        "ekstra procentbaseret gebyr",
        "yderligere procentdel af gebyr",
        "extra procentuell avgift",
        "extra procentbaserad avgift",
        "lisäprosenttimaksu",
        "prosenttiperusteinen lisäpalkkio",
        "supplément de commission",
        "sobretasa internacional",
        "extra percentage",
        "service fee",
        "servicegebühr",
        "servicegebyr",
        "serviceavgift",
        "service charge",
        "servicekosten",
        "servicekostnad",
        "servicekostnader",
        "yderligere servicegebyr",
        "ytterligare serviceavgift",
        "tarifa de servicio adicional",
        "comissão de serviço adicional",
        "tariffa di servizio aggiuntiva",
        "frais de service supplémentaires",
        "dodatkowa opłata za usługę",
        "dodatečný poplatek za služby",
        "dodatočný poplatok za služby",
        "πρόσθετη χρέωση υπηρεσίας",
        "további szolgáltatási díj",
        "tarifa adicional porcentual",
        "comisión porcentual adicional",
        "comisión adicional con base en un porcentaje",
        "tarifa adicional baseada em porcentagem",
        "comissão percentual adicional",
        "tariffa percentuale aggiuntiva",
        "op een percentage gebaseerde",
        "dodatkowa opłata procentowa",
        "dodatečný procentní poplatek",
        "dodatočný percentuálny poplatok",
        "százalékos kiegészítő díja",
        "dodatni postotak",
        "dodatni odstotek",
        "πρόσθετη χρέωση βάσει ποσοστού",
        "additional service",
        "additional service fee",
    )
    if any(kw in text for kw in international_surcharge_keywords):
        return "international_surcharge_table"
    if "währungsumrechnung" in text or "umrechnung des guthabens" in text or "currency conversion" in text:
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
    # Apply negative signals for any candidate category whose text contains a
    # contradictory signal (e.g. "other fees" for an online-card table).
    for category in list(candidates):
        negatives = _TABLE_NEGATIVE_SIGNALS.get(category, ())
        for neg in negatives:
            if _norm(neg) in text:
                candidates.remove(category)
                break
    if not candidates:
        # If the top candidates were removed, fall back to the next-highest-scoring
        # category or to row-label inference.
        removed = set(_TABLE_NEGATIVE_SIGNALS.keys())
        remaining = {cat: sc for cat, sc in scores.items() if cat not in removed}
        if not remaining:
            remaining = scores
        next_score = max(remaining.values())
        candidates = [cat for cat, sc in remaining.items() if sc == next_score]
        # Re-apply negative signals on the fallback candidates.
        for category in list(candidates):
            for neg in _TABLE_NEGATIVE_SIGNALS.get(category, ()):
                if _norm(neg) in text:
                    candidates.remove(category)
                    break
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


# ---------------------------------------------------------------------------
# APM method extraction
# ---------------------------------------------------------------------------

_APM_METHOD_ALIASES: dict[str, tuple[str, ...]] = {
    "thai_online_bank_transfer": (
        "thai online bank transfer",
        "thailändsk online bank transfer",
        "thailandsk online bankoverførsel",
        "thailändische online banküberweisung",
        "thailändische online-banküberweisung",
        "thailändische online bank überweisung",
        "thai online bank",
        "thai online bankoverførsel",
        "thai online banküberweisung",
        "thai online bankoverschrijving",
        "thai online bankovní převod",
        "thai online bankový prevod",
        "thai online banki átutalás",
        "thai virement bancaire en ligne",
        "thai bonifico bancario online",
        "thai transferencia bancaria online",
        "thai transferência bancária online",
        "thai verkkopankkisiirto",
        "thailändsk",
        "thailandsk",
        "thaimaan",
        "thaimaa",
        "thaiföldi",
        "thaïlandaise",
        "tailandesa",
        "tailandese",
        "tailandés",
        "tajlandzkiej",
        "thajsku",
        "thajska",
        "thajském",
        "thajskom",
        "thai",
    ),
    "latvian_online_bank_transfer": (
        "latvian online bank transfer",
        "lettisk online bank transfer",
        "latvijas online bank transfer",
        "latvian online bankoverførsel",
        "lettische online banküberweisung",
        "lettische online-banküberweisung",
        "lettische online bank überweisung",
        "latvian online bank",
        "latvian online banküberweisung",
        "latvian online bankoverschrijving",
        "latvian online bankovní převod",
        "latvian online bankový prevod",
        "latvian online banki átutalás",
        "latvian virement bancaire en ligne",
        "latvian bonifico bancario online",
        "latvian transferencia bancaria online",
        "latvian transferência bancária online",
        "latvian verkkopankkisiirto",
        "lettisk",
        "lettische",
        "latvijas",
        "latvijska",
        "läti",
        "lettországi",
        "letonska",
        "letonské",
        "letonskom",
        "latvian",
    ),
    "lithuanian_online_bank_transfer": (
        "lithuanian online bank transfer",
        "litauisk online bank transfer",
        "lietuvos online bank transfer",
        "lithuanian online bankoverførsel",
        "litauische online banküberweisung",
        "litauische online-banküberweisung",
        "litauische online bank überweisung",
        "lithuanian online bank",
        "lithuanian online banküberweisung",
        "lithuanian online bankoverschrijving",
        "lithuanian online bankovní převod",
        "lithuanian online bankový prevod",
        "lithuanian online banki átutalás",
        "lithuanian virement bancaire en ligne",
        "lithuanian bonifico bancario online",
        "lithuanian transferencia bancaria online",
        "lithuanian transferência bancária online",
        "lithuanian verkkopankkisiirto",
        "litauisk",
        "litauische",
        "lie tuvos",
        "liettualainen",
        "lietuvos",
        "litván",
        "litewska",
        "litewskim",
        "lithuanian",
    ),
    "online_bank_transfer": (
        "online bank transfer",
        "online bankoverførsel",
        "online banküberweisung",
        "online-banküberweisung",
        "online bank überweisung",
        "online bankoverschrijving",
        "online bankovní převod",
        "online bankový prevod",
        "online banki átutalás",
        "virement bancaire en ligne",
        "bonifico bancario online",
        "transferencia bancaria online",
        "transferência bancária online",
        "verkkopankkisiirto",
        "bank transfer",
        "banktransfer",
        "online bank",
        "banküberweisung",
        "bankoverschrijving",
        "banköverföring",
        "bankoverførsel",
    ),
    "skrill": ("skrill",),
    "ovo_premium": (
        "ovo premium",
        "ovo",
    ),
    "gopay": (
        "gopay",
        "go pay",
    ),
    "blik_pay_later": (
        "blik pay later",
        "blik",
    ),
    "kredivo": ("kredivo",),
    "floa_pay": (
        "floa pay",
        "floa",
    ),
    "scalapay": ("scalapay",),
}

_APM_SPECIAL_METHOD_IDS: frozenset[str] = frozenset(
    [
        "thai_online_bank_transfer",
        "latvian_online_bank_transfer",
        "lithuanian_online_bank_transfer",
        "skrill",
        "ovo_premium",
        "gopay",
        "blik_pay_later",
        "kredivo",
        "floa_pay",
        "scalapay",
    ]
)

# Sort aliases by length descending so the longest/most specific phrase wins
# (e.g. "thai online bank transfer" before "online bank transfer").
_APM_SORTED_ALIASES: list[tuple[str, str]] = sorted(
    [
        (canonical, alias)
        for canonical, aliases in _APM_METHOD_ALIASES.items()
        for alias in aliases
    ],
    key=lambda item: (-len(item[1]), item[0], item[1]),
)


_APM_SEPARATOR_RE = re.compile(
    r"[,;/]|"
    r"(?:\s+(?:and|und|i|y|et|og|ja|oraz|och|e)\s+)",
    re.IGNORECASE,
)

# Full phrases that indicate a label part is a generic APM header, not a method.
_APM_HEADER_PHRASES: set[str] = {
    "alle anderen alternativen zahlungsmethoden",
    "alle anderen alternativen zahlungsmethode",
    "alternative zahlungsmethode",
    "alternative zahlungsmethoden",
    "alternative payment method",
    "alternative payment methods",
    "all other alternative payment methods",
    "all other alternative payment method",
    "all other apm",
    "autres moyens de paiement alternatifs",
    "altre modalità di pagamento alternative",
    "otros métodos de pago alternativos",
    "andere alternatieve betaalmethoden",
    "andere alternative betalingsmetoder",
    "andere alternative zahlungsmethoden",
    "andere alternative zahlungsmethode",
    "andre alternative betalingsmetoder",
    "pozostałe metody płatności",
    "övriga alternativa betalningsmetoder",
    "muut vaihtoehtoiset maksutavat",
    "más alternatív fizetési módok",
    "outros métodos de pagamento alternativos",
    "alte metode alternative de plată",
    "ostatné alternatívne spôsoby platby",
    " Ostali alternativni načini plačila",
    "alternatív fizetési módok",
    "alternative betalingsmåter",
    "alternative betalingsmetoder",
    "alternative betalingsmåder",
    "alternative betalningsmetoder",
    "alternative maksutavat",
    "alternativna sredstva plaćanja",
    "alternativni načini plaćanja",
    "alternativne metode plaćanja",
    "alternatívne spôsoby platby",
    "alternativní způsoby platby",
    "alternativní platební metody",
    "alternativne metody płatności",
    "alternatyvūs mokėjimo būdai",
    "alternatīvie maksājumi",
    "alternatīvie maksājumu veidi",
    "alternatívne platobné metódy",
    "alternativne metode plačila",
    "alternativne plačilne metode",
    "alternativni plačilni sistemi",
    "alternativni načini plačevanja",
    "alternativna plačilna sredstva",
    "alternativni plačilni mehanizmi",
    "alternativne metode plačevanja",
    "alternativne finančne storitve",
    "alternativne plačilne rešitve",
    "apm-transaktionsgebühren",
    "apm-transaktion",
    "apm-transaksjoner",
    "apm-transakcje",
    "apm-transakcija",
    "apm transactions",
    "apm transaction",
    "apm-transakciók",
    "apm-maksut",
    "apm-maksutapa",
    "apm-betalinger",
    "apm-betaling",
    "apm-betalning",
    "apm-betalningar",
    "apm-zahlungen",
    "apm-zahlung",
    "apm-maks",
    "apm",
    "abm",
    "vmt",
}

# Tokens that indicate a label part is a generic APM header, not a method list.
_APM_HEADER_TOKENS: set[str] = {
    "alternative",
    "zahlungsmethode",
    "zahlungsmethoden",
    "payment",
    "payments",
    "method",
    "methods",
    "método",
    "métodos",
    "metodo",
    "metodi",
    "betaalmethode",
    "betaalmethoden",
    "moyen",
    "paiement",
    "pagamento",
    "pagamenti",
    "płatności",
    "maksutapa",
    "maksutavat",
    "platba",
    "mód",
    "módy",
    "apm",
    "apms",
    "abm",
    "vmt",
}

# Token sets for the individual payment methods we can extract from APM labels.
_THAI_TOKENS = {
    "thai",
    "thailand",
    "thailändisch",
    "thailändische",
    "thailändsk",
    "thailandsk",
    "thaimaan",
    "thaimaa",
    "thaiföldi",
    "thaïlandaise",
    "tailandesa",
    "tailandese",
    "tailandés",
    "tajlandzkiej",
    "thajsku",
    "thajska",
    "thajském",
    "thajskom",
}
_LATVIAN_TOKENS = {
    "latvian",
    "lettisch",
    "lettische",
    "latvijas",
    "latvijska",
    "läti",
    "lettországi",
    "letonska",
    "letonské",
    "letonskom",
}
_LITHUANIAN_TOKENS = {
    "lithuanian",
    "litauisch",
    "litauische",
    "liettualainen",
    "lietuvos",
    "litván",
    "litewska",
    "litewskim",
    "lie",
    "tuvos",
}
_BANK_TOKENS = {
    "bank",
    "banküberweisung",
    "bankuberweisung",
    "banktransfer",
    "bankoverførsel",
    "bankoverschrijving",
    "bankovní",
    "bankový",
    "banki",
    "átutalás",
    "virement",
    "bonifico",
    "transferencia",
    "transferência",
    "verkkopankki",
    "verkkopankkisiirto",
    "pankki",
    "siirto",
    "überweisung",
    "uberweisung",
    "transfer",
    "transfert",
    "overschrijving",
    "overførsel",
    "prevod",
    "przelew",
    "platba",
    "bankkonto",
    "bankkonten",
    "account",
}
_ONLINE_TOKENS = {
    "online",
    "on-line",
    "on",
    "line",
    "elektronikus",
    "elektronische",
    "elektronisch",
    "eletrônico",
    "electronico",
    "elettronico",
    "internet",
    "verkkopankki",
}

_APM_METHOD_MATCHERS: list[tuple[str, list[set[str]], set[str]]] = [
    ("thai_online_bank_transfer", [_THAI_TOKENS, _ONLINE_TOKENS, _BANK_TOKENS], set()),
    ("latvian_online_bank_transfer", [_LATVIAN_TOKENS, _ONLINE_TOKENS, _BANK_TOKENS], set()),
    ("lithuanian_online_bank_transfer", [_LITHUANIAN_TOKENS, _ONLINE_TOKENS, _BANK_TOKENS], set()),
    ("online_bank_transfer", [_ONLINE_TOKENS, _BANK_TOKENS], _THAI_TOKENS | _LATVIAN_TOKENS | _LITHUANIAN_TOKENS),
    ("skrill", [{"skrill"}], set()),
    ("ovo_premium", [{"ovopremium", "ovo"}], set()),
    ("gopay", [{"gopay", "go"}], set()),
    ("blik_pay_later", [{"blikpaylater", "blik"}], set()),
    ("kredivo", [{"kredivo"}], set()),
    ("floa_pay", [{"floapay", "floa"}], set()),
    ("scalapay", [{"scalapay"}], set()),
]


def _tokenize_apm_label(part_norm: str) -> set[str]:
    """Tokenize an APM label part for robust method matching.

    Collapses multi-word method names (e.g. "go pay", "ovo premium") into a
    single token so they can be matched with word boundaries.
    """
    # Pre-join the small number of multi-word brand names before tokenizing.
    joined = (
        part_norm.replace("go pay", "gopay")
        .replace("ovo premium", "ovopremium")
        .replace("floa pay", "floapay")
        .replace("blik pay later", "blikpaylater")
    )
    # Split on punctuation and whitespace.
    joined = re.sub(r"[^\w\s]", " ", joined)
    return set(joined.split())


def _extract_apm_methods(label: str) -> tuple[list[str], list[str]]:
    """Extract canonical payment-method IDs from an APM row label.

    Returns (canonical_ids, unknown_segments). The canonical IDs are sorted and
    deduplicated. Unknown segments are raw label parts that did not match any
    known method. Token-based matching avoids false positives like "Republik"
    containing "blik" or "Thailändische Baht" containing "thai".
    """
    norm = _norm(label)
    if not norm:
        return [], []

    parts = [p.strip() for p in _APM_SEPARATOR_RE.split(label) if p.strip()]
    if not parts:
        parts = [label]

    methods: set[str] = set()
    unknowns: list[str] = []

    for part in parts:
        part_norm = _norm(part)
        if not part_norm or len(part_norm) < 3:
            continue

        # Drop introductory phrases like "e.g." or "z.b.".
        part_norm = re.sub(r"\b(z\s*\.\s*b\s*\.?|e\s*\.\s*g\s*\.?|np\.|ex\.?)\b", "", part_norm).strip()
        if not part_norm:
            continue

        # Skip generic header phrases ("Alternative payment method", "Alle anderen...").
        if any(phrase in part_norm for phrase in _APM_HEADER_PHRASES):
            continue

        tokens = _tokenize_apm_label(part_norm)

        # Skip parts that are only header tokens.
        if tokens & _APM_HEADER_TOKENS and not (tokens - _APM_HEADER_TOKENS):
            continue

        matched: str | None = None
        for method_id, required_groups, forbidden in _APM_METHOD_MATCHERS:
            if tokens & forbidden:
                continue
            if all(tokens & group for group in required_groups):
                matched = method_id
                break

        if matched:
            methods.add(matched)
        else:
            unknowns.append(part)

    return sorted(methods), sorted(set(unknowns))


def _is_apm_special_label(label: str) -> bool:
    """Return True if a row label describes APM special methods.

    These labels list multiple alternative payment methods (e.g. Thai online
    bank transfer, Skrill, BLIK, Kredivo, etc.) and would otherwise be
    misclassified because they contain substrings like "pay later" (from
    "BLIK Pay Later") that collide with invoice_pay_later / pay_later_consumer.
    """
    methods, _ = _extract_apm_methods(label)
    return any(m in _APM_SPECIAL_METHOD_IDS for m in methods)


def _variant_id_for_row(product_id: str, label: str, methods: list[str]) -> str | None:
    """Return a stable variant id for a row, if needed."""
    if product_id != "alternative_payment_methods":
        return None
    if any(m in _APM_SPECIAL_METHOD_IDS for m in methods):
        return "special"
    if methods or "online" in _norm(label) and "bank" in _norm(label):
        return "default"
    return "default"


def _conditions_for_row(
    product_id: str,
    variant_id: str | None,
    label: str,
    methods: list[str] | None = None,
) -> dict[str, Any]:
    """Return calculable conditions for a product rule based on the source row."""
    conditions: dict[str, Any] = {}
    if product_id == "nonprofit":
        conditions["merchant_approval_required"] = True
    if product_id == "alternative_payment_methods":
        if methods is None:
            methods, _ = _extract_apm_methods(label)
        if methods:
            conditions["payment_methods"] = sorted(methods)
    return conditions


def _schedule_name_from_table(table: Table, default: str | None) -> str:
    text = _table_text(table)
    mapping = {
        "goods_and_services": (
            "geld für waren und dienstleistungen",
            "waren und dienstleistungen",
            "goods and services",
            "varer og tjenesteydelser",
            "varer og tjenester",
            "bienes y servicios",
            "beni e servizi",
            "goederen en diensten",
            "produkter och tjänster",
            "produkter og tjenester",
        ),
        "donations": (
            "spende",
            "donation",
            "donationer",
            "donations",
            "donaties",
            "donativos",
            "donativas",
            "doações",
            "donazioni",
            "lahjoitukset",
            "darowizn",
            "príspevkov",
            "príspevky",
            "adományok",
            "δωρεές",
            "δωρεες",
            "donatii",
            "дарения",
            "donacije",
            "donacijo",
            "příspěvky",
            "příspěvků",
        ),
        "nonprofit": (
            "gemeinnützig",
            "nonprofit",
            "non-profit",
            "velgørende",
            "charity",
            "charitable",
            "charitat",
            "caridad",
            "caridade",
            "caritative",
            "organizaciones sin fines de lucro",
            "organizzazioni senza scopo di lucro",
            "organisasjoner",
            "organizacja non-profit",
            "organizacje charytatywne",
            "charitatívnych",
            "charitativních",
            "charitativní",
            "jótékonysági",
            "välgörenhetsorganisationer",
            "välgörenhet",
            "φιλανθρωπικές",
            "φιλανθρωπικά",
            "φιλανθρωπικού",
            "instituições de solidariedade",
            "instituição de caridade",
            "institución de solidaridad",
            "entidades sin ánimo de lucro",
            "associazioni di volontariato",
            "liefdadigheid",
        ),
        "micropayments": (
            "mikrozahlung",
            "micropayment",
            "mikrobetaling",
            "mikrobetalinger",
            "mikromaksu",
            "mikropłatność",
            "micropagos",
            "micropaiement",
            "micropaiements",
            "micropagamentos",
            "mikrobetalning",
            "mikrobetalningar",
            "mikroplatby",
            "mikroplatby",
            "mikroπληρωμές",
            "μικροπληρωμές",
            "μικροπληρωμες",
            "mikrotransakciók",
            "mikrotransakcije",
        ),
        "alternative_payment_methods": (
            "alternative zahlungsmethode",
            "alternative payment",
            "apm",
            "alternativ betalingsmetode",
            "alternative betalingsmetode",
            "alternative betaalmethode",
            "métodos de pago alternativos",
            "metodi di pagamento alternativi",
            "metodo di pagamento alternativo",
            "andere betaalmethode",
            "autre moyen de paiement",
            "autre mode de paiement",
            "alternatív fizetési",
            "alternatívny spôsob platby",
            "alternativní způsob platby",
            "alternatywna forma płatności",
            "alternativ betalningsmetod",
            "alternativ betalingsmåte",
            "vaihtoehtoinen maksutapa",
            "vmt",
            "abm",
            "εναλλακτικός τρόπος πληρωμής",
            "εναλλακτικος τροπος πληρωμης",
            "συναλλαγές με εμπ",
            "συναλλαγες με εμπ",
        ),
        "online_card_payments": (
            "online-kartenzahlungen",
            "online card",
            "online card payments",
            "online-kortbetaling",
            "online kort",
            "online kortbetalingstjenester",
            "kortbetalingstjenester",
            "kortbetalningstjänster",
            "kortbetalning",
            "tarjetas de crédito y débito",
            "kredit- och debitkort",
            "kredit- og betalingskort",
            "credit and debit card",
            "servizi di pagamento con carta",
            "services de paiement par carte",
            "serviços de pagamento com cartão",
            "servicios de pago con tarjeta",
            "online platby kartou",
            "online platby kartou",
            "online kártyás",
            "online kártyás",
            "verkkokorttimaksupalvelut",
            "verkkokorttimaksu",
            "ηλεκτρονικές πληρωμές με κάρτα",
            "ηλεκτρονικες πληρωμες με καρτα",
            "υπηρεσιών paypal για ηλεκτρονικές",
            "υπηρεσιων paypal για ηλεκτρονικες",
        ),
        "pos_transactions": (
            "point of sale",
            "präsenter karte",
            "kortforevisning",
            "card present",
            "ponto de venda",
            "punto de venta",
            "punto vendita",
            "punkty sprzedaży",
        ),
        "commercial": (
            "geschäftlichen transaktionen",
            "commercial transaction",
            "commercial",
            "erhvervsbetalinger",
            "erhverv",
            "business payments",
            "standardgebyr",
            "standardavgift",
            "standaardtarief",
            "tarifa estándar",
            "tarifa standard",
            "tarifa padrão",
            "tariffa standard",
            "tarification standard",
            "standard fee",
            "standard rate",
            "standardní sazba",
            "štandardná sadzba",
            "szokásos díja",
            "szokásos díj",
            "standardowa stawka",
            "standardtaxa",
            "standardsats",
            "standard taxa",
            "εμπορικές συναλλαγές",
            "εμπορικες συναλλαγες",
            "commerciale",
            "commerciële",
            "commerciale",
            "kommersielle",
            "kommercielle",
            "kommersiella",
            "komercyjne",
            "komerčních",
            "komerčných",
            "kereskedelmi",
            "comerciale",
            "obchodní transakce",
            "obchodné transakcie",
            "emporia",
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


def _extract_fixed_fee_schedule(table: Table, source: Source | None = None) -> FixedFeeSchedule | None:
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
    return FixedFeeSchedule(entries=amounts, sources=sources)


def _extract_international_surcharge_schedule(table: Table, source: Source | None = None) -> InternationalSurchargeSchedule | None:
    entries: list[InternationalSurchargeScheduleEntry] = []
    seen: set[str] = set()
    fallback_rows: list[tuple[str, str]] = []
    for row in table.rows:
        pct = _first_percentage(row)
        label = _row_label(row)
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
            if any(phrase in fee_text for phrase in no_fee_phrases):
                pct = "0"
            else:
                continue
        if region is None:
            # Some region-less tables (e.g. Brazil) list transaction types instead of
            # payer regions. Keep these as a fallback in case no region is recognized.
            if label:
                fallback_rows.append((label, pct))
            continue
        if region in seen:
            continue
        seen.add(region)
        entries.append(InternationalSurchargeScheduleEntry(payer_region=region, percentage_points=pct))
    if not entries and fallback_rows:
        # No recognized region rows: treat the first percentage row as a generic
        # "OTHER" international surcharge. This is typically a region-less rate.
        label, pct = fallback_rows[0]
        entries.append(InternationalSurchargeScheduleEntry(payer_region="OTHER", percentage_points=pct))
    if not entries:
        return None

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
    return InternationalSurchargeSchedule(entries=entries, sources=sources)


def _normalize_region(text: str) -> str | None:
    t = _norm(text)
    if not t:
        return None
    exact = {
        "eu": "EEA",
        "gb": "GB",
        "uk": "GB",
        "us": "US_CA",
        "eøs": "EEA",
        "ees": "EEA",
        "eea": "EEA",
        "eee": "EEA",
        "ehp": "EEA",
        "egt": "EEA",
        "eta": "EEA",
        "see": "EEA",
        "eer": "EEA",
        "εοχ": "EEA",
    }
    if t in exact:
        return exact[t]
    if "europa ii" in t:
        return "EUROPE_II"
    if "europa i" in t:
        return "EUROPE_I"
    if "nordeuropa" in t or "northern europe" in t or "nordic" in t or "pohjois-eurooppa" in t:
        return "NORTHERN_EUROPE"
    # EEA in many languages
    if (
        "europäischer wirtschaftsraum" in t
        or "ewr" in t
        or "eea" in t
        or "e.u" in t
        or "eøs" in t
        or "ees" in t
        or "see" in t
        or "eee" in t
        or "ehp" in t
        or "egt" in t
        or "eta" in t
        or "eer" in t
        or "εοχ" in t
        or "espace économique européen" in t
        or "spazio economico europeo" in t
        or "espacio económico europeo" in t
        or "europæiske økonomiske samarbejdsområde" in t
        or "europeisk økonomisk samarbeidsområde" in t
        or "europeiska ekonomiska samarbetsområdet" in t
        or "europese economische ruimte" in t
        or "euroopan talousalue" in t
        or "europski gospodarski prostor" in t
        or "európai gazdasági térség" in t
    ):
        return "EEA"
    # UK / GB in many languages
    if (
        "vereinigtes königreich" in t
        or "großbritannien" in t
        or "storbritannien" in t
        or "storbritannia" in t
        or "united kingdom" in t
        or "britain" in t
        or "regno unito" in t
        or "royaume-uni" in t
        or "royaume uni" in t
        or "verenigd koninkrijk" in t
        or "iso-britannia" in t
        or "Ηνωμένο Βασίλειο" in t
        or "ηνωμενο βασιλειο" in t
        or "storbritannien" in t
        or "britannien" in t
        or "spojuené kráľovstvo" in t
        or "spojené království" in t
        or "egyesült királyság" in t
        or "britannia" in t
    ):
        return "GB"
    if "usa" in t or "united states" in t or "u.s" in t or "canada" in t or "nordamerika" in t or "états-unis" in t or "etats-unis" in t or "stati uniti" in t or "estados unidos" in t or "verenigde staten" in t or "yhdysvallat" in t or "ΗΠΑ" in t or "ηπα" in t:
        return "US_CA"
    # "All other markets" in many languages
    if (
        ("all" in t and "other" in t)
        or ("all" in t and "andere" in t)
        or ("tutti" in t and "altri" in t)
        or ("tous" in t and "autres" in t)
        or ("kaikki" in t and "muut" in t)
        or ("alle" in t and "andere" in t)
        or ("alle" in t and "ander" in t)
        or ("všechny" in t and "ostatní" in t)
        or ("všetky" in t and "ostatné" in t)
        or ("minden" in t and "egyéb" in t)
        or ("λοιπές" in t)
        or ("λοιπες" in t)
        or "rest" in t
        or "restante" in t
        or "andere" in t
        or "sonstige" in t
        or "welt" in t
        or "andre markeder" in t
        or "andre lande" in t
        or "andere länder" in t
        or "altri paesi" in t
        or "altri mercati" in t
        or "otros países" in t
        or "otros mercados" in t
        or "inni kraje" in t
        or "pozostale" in t
        or "pozostałe" in t
        or "andre marknader" in t
        or "alla andra marknader" in t
        or "alle andere markten" in t
        or "kaikki muut markkinat" in t
        or "všechny ostatní trhy" in t
        or "všetky ostatné trhy" in t
        or "minden egyéb piac" in t
        or "λοιπές αγορές" in t
        or "λοιπες αγορες" in t
    ):
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
        base, suffix = reference.split(".", 1)
        suffix_product = _REFERENCE_SUFFIX_TO_PRODUCT.get(suffix)
        # A qualified reference like "online_card_payments.advanced" points to the
        # advanced card product variant, not to a generic online_card_payments rule.
        target_id = suffix_product or _REFERENCE_SUFFIX_TO_PRODUCT.get(base, base)
    else:
        target_id = reference

    # Find a concrete target rule. Prefer a generic/default variant (variant_id
    # is None or the literal "default" string) because a bare reference like
    # "alternative_payment_methods" should resolve to the default variant, not
    # to a special variant.
    for rule in rules:
        if rule.id == target_id and rule.percentage is not None and rule.variant_id in (None, "default"):
            return ResolvedRate(
                percentage=rule.percentage,
                fixed_fee_schedule=rule.fixed_fee_schedule,
                international_surcharge_schedule=rule.international_surcharge_schedule,
                source=rule.source,
                rule_id=rule.id,
            )
    for rule in rules:
        if rule.id == target_id and rule.percentage is not None:
            return ResolvedRate(
                percentage=rule.percentage,
                fixed_fee_schedule=rule.fixed_fee_schedule,
                international_surcharge_schedule=rule.international_surcharge_schedule,
                source=rule.source,
                rule_id=rule.id,
            )
    # Fallback: find a rule whose label matches the reference product aliases.
    aliases = _PRODUCT_ALIASES.get(target_id, ())
    for rule in rules:
        if rule.label and any(_norm(a) in _norm(rule.label) for a in aliases):
            return ResolvedRate(
                percentage=rule.percentage,
                fixed_fee_schedule=rule.fixed_fee_schedule,
                international_surcharge_schedule=rule.international_surcharge_schedule,
                source=rule.source,
                rule_id=rule.id,
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
        document_id=row.source_document_id or table.document_id,
        component_id=row.source_component_id or table.component_id,
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
    variant_id: str | None
    label: str
    percentage: str | None
    fixed_fee_schedule: str | None
    international_surcharge_schedule: str | None
    conditions: dict[str, Any]
    table: Table
    row: Row
    row_index: int
    reference: str | None = None
    unknown_apm_methods: list[str] = field(default_factory=list)


def _extract_rules_from_rate_table(
    table: Table,
    table_category: str,
    source: Source | None,
) -> tuple[list[_ExtractedRule], list[UnclassifiedFeeRow], list[AmbiguousFeeRow], list[UnclassifiedFeeRow]]:
    rules: list[_ExtractedRule] = []
    unclassified: list[UnclassifiedFeeRow] = []
    ambiguous: list[AmbiguousFeeRow] = []
    ignored: list[UnclassifiedFeeRow] = []

    default_product = _TABLE_CATEGORY_PRODUCT.get(table_category)

    for idx, row in enumerate(table.rows):
        label = _row_label(row)
        if not label:
            ignored.append(
                UnclassifiedFeeRow(
                    normalized_cells=_row_cells_text(row),
                    original_label=label,
                    source=_provenance(table, row, idx, source, original_label=label),
                    reason="empty label",
                )
            )
            continue
        # Pre-check: APM special method labels (Thai/Latvian/Lithuanian bank
        # transfer, Skrill, BLIK, Kredivo, etc.) are unambiguously APM even
        # though they contain "pay later" substrings.
        if _is_apm_special_label(label):
            product_id = "alternative_payment_methods"
            ambiguous_candidates = []
        else:
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
            # Use the table category as a fallback for rows that do not match any
            # product alias (e.g. "Deutschland" in the nonprofit table).
            if default_product and (_row_has_percentage(row) or _detect_reference(row, default_product)):
                product_id = default_product
            elif len(label) > 3 and _row_has_percentage(row):
                unclassified.append(
                    UnclassifiedFeeRow(
                        normalized_cells=_row_cells_text(row),
                        original_label=label,
                        source=_provenance(table, row, idx, source, original_label=label),
                        reason="no product alias matched",
                    )
                )
                continue
            else:
                ignored.append(
                    UnclassifiedFeeRow(
                        normalized_cells=_row_cells_text(row),
                        original_label=label,
                        source=_provenance(table, row, idx, source, original_label=label),
                        reason="no product alias and no rate",
                    )
                )
                continue
        fee_text = _row_fee_cell(row)
        pct, _fixed = _parse_rate_expression(fee_text)
        reference = _detect_reference(row, product_id)
        methods, unknown_methods = _extract_apm_methods(label)
        variant_id = _variant_id_for_row(product_id, label, methods)
        fixed_schedule = _fixed_fee_schedule_for(product_id)
        intl_schedule = _international_surcharge_schedule_for(product_id)
        conditions = _conditions_for_row(product_id, variant_id, label, methods=methods)
        rules.append(
            _ExtractedRule(
                product_id=product_id,
                variant_id=variant_id,
                label=label,
                percentage=pct,
                fixed_fee_schedule=fixed_schedule,
                international_surcharge_schedule=intl_schedule,
                conditions=conditions,
                table=table,
                row=row,
                row_index=idx,
                reference=reference,
                unknown_apm_methods=unknown_methods,
            )
        )
    return rules, unclassified, ambiguous, ignored


# Maps a product to the fixed-fee schedule it uses. The target schedule may be
# the product's own schedule (e.g. goods_and_services) or an inherited schedule
# (e.g. paypal_checkout -> commercial). None means the product has no fixed fee.
_FIXED_FEE_SCHEDULE_FOR: dict[str, str | None] = {
    "paypal_checkout": "commercial",
    "goods_and_services": "goods_and_services",
    "online_card_payments": "online_card_payments",
    "advanced_card_payments": "online_card_payments",
    "other_commercial": "commercial",
    "guest_checkout": "commercial",
    "invoice_pay_later": "commercial",
    "pay_later_consumer": "commercial",
    "qr_code_payments": None,
    "donations": "donations",
    "nonprofit": "nonprofit",
    "micropayments": "micropayments",
    "alternative_payment_methods": "alternative_payment_methods",
    "pos_transactions": None,
    "chargebacks": None,
    "refunds": None,
    "disputes": None,
    "card_verification": None,
    "currency_conversion": None,
    "withdrawals": None,
}

# Subset of _FIXED_FEE_SCHEDULE_FOR that represents explicit inheritance.
_FIXED_FEE_INHERITANCE: dict[str, str] = {
    "paypal_checkout": "commercial",
    "other_commercial": "commercial",
    "guest_checkout": "commercial",
    "invoice_pay_later": "commercial",
    "pay_later_consumer": "commercial",
    "advanced_card_payments": "online_card_payments",
}


def _fixed_fee_schedule_for(product_id: str) -> str | None:
    """Return the fixed-fee schedule name for a product, or None if no fixed fee applies."""
    return _FIXED_FEE_SCHEDULE_FOR.get(product_id)


# Same as above for international surcharge schedules.
_INTERNATIONAL_SURCHARGE_SCHEDULE_FOR: dict[str, str | None] = {
    "paypal_checkout": "commercial",
    "goods_and_services": "goods_and_services",
    "online_card_payments": "online_card_payments",
    "advanced_card_payments": "commercial",
    "other_commercial": "commercial",
    "guest_checkout": "commercial",
    "invoice_pay_later": "commercial",
    "pay_later_consumer": "commercial",
    "qr_code_payments": "commercial",
    "donations": "donations",
    "nonprofit": "nonprofit",
    "micropayments": None,
    "alternative_payment_methods": None,
    "pos_transactions": None,
    "chargebacks": None,
    "refunds": None,
    "disputes": None,
    "card_verification": None,
    "currency_conversion": None,
    "withdrawals": None,
}

_INTERNATIONAL_SURCHARGE_INHERITANCE: dict[str, str] = {
    "paypal_checkout": "commercial",
    "other_commercial": "commercial",
    "advanced_card_payments": "commercial",
    "guest_checkout": "commercial",
    "invoice_pay_later": "commercial",
    "pay_later_consumer": "commercial",
    "qr_code_payments": "commercial",
}


def _international_surcharge_schedule_for(product_id: str) -> str | None:
    """Return the international surcharge schedule name for a product, or None."""
    return _INTERNATIONAL_SURCHARGE_SCHEDULE_FOR.get(product_id)


# ---------------------------------------------------------------------------
# Schedule assembly
# ---------------------------------------------------------------------------


def _collect_schedules(
    tables: list[Table],
    source: Source | None = None,
) -> tuple[dict[str, FixedFeeSchedule], dict[str, InternationalSurchargeSchedule], list[Diagnostic]]:
    """Extract fixed-fee and international-surcharge schedules.

    Schedules are keyed by product name.  If two tables map to the same product
    (e.g. "Fixed fee by received currency" and "Currency fixed fees" both for
    commercial), their entries are merged and sources are combined.  Conflicting
    duplicate keys are reported as diagnostics and the first encountered value
    is kept.
    """
    fixed: dict[str, FixedFeeSchedule] = {}
    international: dict[str, InternationalSurchargeSchedule] = {}
    diagnostics: list[Diagnostic] = []

    for table in tables:
        category = _classify_table_category(table)
        if category == "fixed_fee_table":
            schedule = _extract_fixed_fee_schedule(table, source=source)
            if schedule:
                name = _schedule_name_from_table(table, "commercial")
                existing = fixed.get(name)
                if existing:
                    merged_entries = dict(existing.entries)
                    merged_sources = list(existing.sources)
                    for s in schedule.sources:
                        if s not in merged_sources:
                            merged_sources.append(s)
                    for currency, amount in schedule.entries.items():
                        if currency in merged_entries:
                            if merged_entries[currency] != amount:
                                diagnostics.append(
                                    Diagnostic(
                                        type="conflicting_schedule_entry",
                                        schedule_type="fixed_fee",
                                        schedule_id=name,
                                        normalized_key=currency,
                                        values=[merged_entries[currency], amount],
                                        sources=_merge_provenance_sources(existing.sources, schedule.sources),
                                    )
                                )
                            # Keep first value; do not overwrite.
                        else:
                            merged_entries[currency] = amount
                    fixed[name] = FixedFeeSchedule(entries=merged_entries, sources=merged_sources)
                else:
                    fixed[name] = schedule
        elif category == "international_surcharge_table":
            schedule = _extract_international_surcharge_schedule(table, source=source)
            if schedule:
                name = _schedule_name_from_table(table, "commercial")
                existing = international.get(name)
                if existing:
                    merged_entries = list(existing.entries)
                    seen = {e.payer_region: e for e in merged_entries}
                    merged_sources = list(existing.sources)
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
                    international[name] = InternationalSurchargeSchedule(entries=merged_entries, sources=merged_sources)
                else:
                    international[name] = schedule
    return fixed, international, diagnostics


def _merge_provenance_sources(*source_lists: list[Provenance]) -> list[Provenance]:
    """Combine multiple provenance lists without duplicates."""
    merged: list[Provenance] = []
    for sources in source_lists:
        for s in sources:
            if s not in merged:
                merged.append(s)
    return merged


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def _derive_status(
    rules: list[TransactionFeeRule],
    unclassified: list[UnclassifiedFeeRow],
    ambiguous: list[AmbiguousFeeRow],
    ignored: list[UnclassifiedFeeRow],
    diagnostics: list[Diagnostic],
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
) -> str:
    if not rules and not fixed_schedules and not international_schedules:
        return "unclassified"
    if ambiguous or unclassified or ignored:
        return "partial"
    # Any diagnostic indicates the classification is not fully trustworthy.
    if diagnostics:
        return "partial"
    # A complete result should expose the core commercial rules for a market.
    has_commercial = any(r.id in {"paypal_checkout", "goods_and_services", "other_commercial"} for r in rules)
    if has_commercial and bool(fixed_schedules):
        return "complete"
    return "partial"


def _rule_identity(rule: TransactionFeeRule) -> str:
    """Return a stable identity key for deduplicating equivalent rules.

    The identity covers product family, variant, percentage, applicable
    conditions and the schedule references that determine which fee table is
    used. The source provenance and the concrete rate reference object are
    intentionally excluded so that a source row that references a target and
    the target row itself can be merged into one rule.
    """
    return json.dumps(
        {
            "id": rule.id,
            "variant_id": rule.variant_id,
            "percentage": str(rule.percentage) if rule.percentage is not None else None,
            "conditions": {k: rule.conditions[k] for k in sorted(rule.conditions)} if rule.conditions else {},
            "fixed_fee_schedule": rule.fixed_fee_schedule,
            "international_surcharge_schedule": rule.international_surcharge_schedule,
        },
        sort_keys=True,
        ensure_ascii=False,
    )


def _rule_has_rate(rule: TransactionFeeRule) -> bool:
    """Return True if the rule carries a directly usable percentage."""
    return bool(rule.percentage is not None or (rule.rate_reference is not None and rule.rate_reference.resolved_rate is not None))


def _is_reference_source(rule: TransactionFeeRule) -> bool:
    """Return True if the rule is a reference that resolves to another rule."""
    return bool(rule.rate_reference is not None and rule.rate_reference.resolved_rate is not None)


def _deduplicate_rules(
    rules: list[TransactionFeeRule],
    diagnostics: list[Diagnostic] | None = None,
) -> list[TransactionFeeRule]:
    """Merge equivalent rules, preserving variants and preferring resolved references.

    Rules are equivalent when their product family, variant, conditions and
    schedule references are identical. Within an equivalence group we prefer:
    1. a rule with a usable rate (or a resolved reference), and
    2. a rule that carries a reference (because it ties the source and target
       together), then
    3. the first rule in source order.
    """
    groups: dict[str, list[tuple[int, TransactionFeeRule]]] = {}
    for idx, rule in enumerate(rules):
        groups.setdefault(_rule_identity(rule), []).append((idx, rule))

    selected: list[TransactionFeeRule] = []
    for group in groups.values():
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


def _build_coverage_summary(
    rules: list[TransactionFeeRule],
    unresolved_rules: list[TransactionFeeRule],
    fixed_schedules: dict[str, FixedFeeSchedule],
    international_schedules: dict[str, InternationalSurchargeSchedule],
    ignored_rows: list[UnclassifiedFeeRow],
    unclassified_rows: list[UnclassifiedFeeRow],
    ambiguous_rows: list[AmbiguousFeeRow],
    diagnostics: list[Diagnostic],
    extracted_rules: list[_ExtractedRule],
) -> CoverageSummary:
    """Compute the classification coverage summary."""
    fixed_fee_entries = sum(len(s.entries) for s in fixed_schedules.values())
    intl_entries = sum(len(s.entries) for s in international_schedules.values())
    reference_sources = sum(1 for e in extracted_rules if e.reference)
    # Count targets from all resolved references, including the ones that are
    # deduplicated away in the final transaction rule list.
    reference_target_ids = {
        r.rate_reference.resolved_rate.rule_id
        for r in unresolved_rules
        if r.rate_reference and r.rate_reference.resolved_rate and r.rate_reference.resolved_rate.rule_id
    }
    conflicts = sum(1 for d in diagnostics if d.type == "conflicting_schedule_entry")
    missing_schedules = sum(1 for d in diagnostics if d.type == "missing_required_schedule")
    unresolved_references = sum(1 for d in diagnostics if d.type == "unresolved_reference")
    unknown_apm = sum(1 for d in diagnostics if d.type == "unknown_apm_method")
    extracted_apm = sum(
        len(e.conditions.get("payment_methods", []))
        for e in extracted_rules
        if e.product_id == "alternative_payment_methods"
    )

    inherited = 0
    for rule in rules:
        if rule.fixed_fee_schedule and rule.fixed_fee_schedule in _FIXED_FEE_INHERITANCE:
            inherited += 1
        if (
            rule.international_surcharge_schedule
            and rule.international_surcharge_schedule in _INTERNATIONAL_SURCHARGE_INHERITANCE
        ):
            inherited += 1

    return CoverageSummary(
        transaction_rules=len(rules),
        fixed_fee_entries=fixed_fee_entries,
        international_surcharge_entries=intl_entries,
        reference_sources=reference_sources,
        reference_targets=len(reference_target_ids),
        ignored=len(ignored_rows),
        unclassified=len(unclassified_rows),
        ambiguous=len(ambiguous_rows),
        conflicts=conflicts,
        missing_required_schedules=missing_schedules,
        inherited_schedules=inherited,
        unresolved_references=unresolved_references,
        extracted_apm_methods=extracted_apm,
        unknown_apm_methods=unknown_apm,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def classify_tables(tables: list[Table], source: Source | None = None) -> DerivedFeeResult:
    """Derive product-specific transaction fee rules from normalized tables."""
    fixed_schedules, international_schedules, schedule_diagnostics = _collect_schedules(
        tables, source=source
    )
    diagnostics: list[Diagnostic] = list(schedule_diagnostics)

    extracted_rules: list[_ExtractedRule] = []
    unclassified_rows: list[UnclassifiedFeeRow] = []
    ambiguous_rows: list[AmbiguousFeeRow] = []
    ignored_rows: list[UnclassifiedFeeRow] = []

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
            rules, uncls, ambig, ignored = _extract_rules_from_rate_table(table, category, source)
            extracted_rules.extend(rules)
            unclassified_rows.extend(uncls)
            ambiguous_rows.extend(ambig)
            ignored_rows.extend(ignored)

    # First pass: build TransactionFeeRule objects without resolving references so
    # that all candidate target rules exist for the second pass.
    unresolved_rules: list[TransactionFeeRule] = []
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
                conditions=extracted.conditions,
                rate_reference=None,
                source=prov,
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

    # Second pass: resolve textual references against all collected rules.
    for i, extracted in enumerate(extracted_rules):
        if not extracted.reference:
            continue
        rule = unresolved_rules[i]
        resolved = _resolve_reference(extracted.reference, unresolved_rules)
        if resolved:
            percentage = rule.percentage
            if resolved.percentage and percentage is None:
                percentage = resolved.percentage
            unresolved_rules[i] = rule.model_copy(
                update={
                    "rate_reference": RateReference(
                        reference=extracted.reference,
                        resolved_rate=resolved,
                        source=rule.source,
                    ),
                    "percentage": percentage,
                }
            )
        else:
            diagnostics.append(
                Diagnostic(
                    type="unresolved_reference",
                    rule_id=rule.id,
                    label=extracted.label,
                    sources=[rule.source] if rule.source else [],
                )
            )

    # Validate schedule references. Missing schedules are reported as diagnostics
    # and the dangling reference is cleared; there is no implicit fallback to a
    # different schedule family.
    for idx, rule in enumerate(unresolved_rules):
        if rule.fixed_fee_schedule:
            if rule.fixed_fee_schedule in fixed_schedules:
                if rule.fixed_fee_schedule in _FIXED_FEE_INHERITANCE:
                    diagnostics.append(
                        Diagnostic(
                            type="inherited_schedule",
                            rule_id=rule.id,
                            schedule_type="fixed_fee",
                            expected_schedule=rule.fixed_fee_schedule,
                            inherited_from=_FIXED_FEE_INHERITANCE[rule.fixed_fee_schedule],
                            sources=[rule.source] if rule.source else [],
                        )
                    )
            else:
                diagnostics.append(
                    Diagnostic(
                        type="missing_required_schedule",
                        rule_id=rule.id,
                        schedule_type="fixed_fee",
                        expected_schedule=rule.fixed_fee_schedule,
                        sources=[rule.source] if rule.source else [],
                    )
                )
                rule = rule.model_copy(update={"fixed_fee_schedule": None})
        if rule.international_surcharge_schedule:
            if rule.international_surcharge_schedule in international_schedules:
                if rule.international_surcharge_schedule in _INTERNATIONAL_SURCHARGE_INHERITANCE:
                    diagnostics.append(
                        Diagnostic(
                            type="inherited_schedule",
                            rule_id=rule.id,
                            schedule_type="international_surcharge",
                            expected_schedule=rule.international_surcharge_schedule,
                            inherited_from=_INTERNATIONAL_SURCHARGE_INHERITANCE[rule.international_surcharge_schedule],
                            sources=[rule.source] if rule.source else [],
                        )
                    )
            else:
                diagnostics.append(
                    Diagnostic(
                        type="missing_required_schedule",
                        rule_id=rule.id,
                        schedule_type="international_surcharge",
                        expected_schedule=rule.international_surcharge_schedule,
                        sources=[rule.source] if rule.source else [],
                    )
                )
                rule = rule.model_copy(update={"international_surcharge_schedule": None})
        unresolved_rules[idx] = rule

    # Merge equivalent rules and preserve legitimate variants.
    transaction_rules = _deduplicate_rules(unresolved_rules)
    transaction_rules.sort(key=_rule_sort_key)

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

    coverage = _build_coverage_summary(
        transaction_rules,
        unresolved_rules,
        fixed_schedules,
        international_schedules,
        ignored_rows,
        unclassified_rows,
        ambiguous_rows,
        diagnostics,
        extracted_rules,
    )

    status = _derive_status(
        transaction_rules,
        unclassified_rows,
        ambiguous_rows,
        ignored_rows,
        diagnostics,
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
        ignored_rows=ignored_rows,
        diagnostics=diagnostics,
        coverage_summary=coverage,
    )
