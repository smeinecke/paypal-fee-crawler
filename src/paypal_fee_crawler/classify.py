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
        "skrill",
        "gopay",
        "blik",
        "kredivo",
        "floa",
        "scalapay",
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
        "nacionales",
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
    for field in (token.raw, token.internal_name, token.fee_data_key):
        if _text_indicates_percentage(field):
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


_APM_SPECIAL_METHODS = (
    "thai",
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
    "latvian",
    "lettisk",
    "lettische",
    "latvijas",
    "latvijska",
    "läti",
    "lettországi",
    "letonska",
    "letonské",
    "letonskom",
    "lithuanian",
    "litauisk",
    "litauische",
    "lie tuvos",
    "liettualainen",
    "lietuvos",
    "litván",
    "litewska",
    "litewskim",
    "online bank transfer",
    "online bankoverførsel",
    "online banküberweisung",
    "online bankoverschrijving",
    "online bankovní převod",
    "online bankový prevod",
    "online banki átutalás",
    "virement bancaire en ligne",
    "bonifico bancario online",
    "transferencia bancaria online",
    "transferência bancária online",
    "verkkopankkisiirto",
    "skrill",
    "ovo",
    "gopay",
    "blik",
    "kredivo",
    "floa",
    "scalapay",
)


def _is_apm_special_label(label: str) -> bool:
    """Return True if a row label describes APM special methods.

    These labels list multiple alternative payment methods (e.g. Thai online
    bank transfer, Skrill, BLIK, Kredivo, etc.) and would otherwise be
    misclassified because they contain substrings like "pay later" (from
    "BLIK Pay Later") that collide with invoice_pay_later / pay_later_consumer.
    """
    norm = _norm(label)
    # Must contain at least two APM special method keywords to qualify,
    # or one of the unambiguous single-method keywords (skrill, gopay, etc.).
    matches = sum(1 for m in _APM_SPECIAL_METHODS if _norm(m) in norm)
    if matches >= 2:
        return True
    unambiguous = ("skrill", "gopay", "kredivo", "floa", "scalapay", "blik pay later")
    return any(_norm(m) in norm for m in unambiguous)


def _rule_id_for_row(product_id: str, label: str) -> str:
    """Return a stable rule id for a row, creating variants where needed."""
    norm = _norm(label)
    if product_id == "alternative_payment_methods":
        if any(_norm(m) in norm for m in _APM_SPECIAL_METHODS):
            return "alternative_payment_methods_special"
        return "alternative_payment_methods"
    return product_id


def _conditions_for_row(product_id: str, label: str, table_category: str) -> dict[str, Any]:
    """Return calculable conditions for a product rule based on the source row."""
    conditions: dict[str, Any] = {}
    if product_id == "nonprofit":
        conditions["merchant_approval_required"] = True
    if product_id == "alternative_payment_methods_special":
        conditions["payment_methods"] = [
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
    if product_id == "alternative_payment_methods" and "online bank transfer" in _norm(label):
        conditions["payment_methods"] = ["online_bank_transfer"]
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
                source=rule.source,
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
    rule_id: str
    label: str
    percentage: str | None
    fixed_fee_schedule: str | None
    international_surcharge_schedule: str | None
    conditions: dict[str, Any]
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

    default_product = _TABLE_CATEGORY_PRODUCT.get(table_category)

    for idx, row in enumerate(table.rows):
        label = _row_label(row)
        if not label:
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
                continue
        fee_text = _row_fee_cell(row)
        pct, _fixed = _parse_rate_expression(fee_text)
        reference = _detect_reference(row, product_id)
        rule_id = _rule_id_for_row(product_id, label)
        fixed_schedule = _fixed_fee_schedule_for(product_id, default_schedule)
        intl_schedule = _international_surcharge_schedule_for(product_id, default_schedule)
        conditions = _conditions_for_row(rule_id, label, table_category)
        rules.append(
            _ExtractedRule(
                product_id=product_id,
                rule_id=rule_id,
                label=label,
                percentage=pct,
                fixed_fee_schedule=fixed_schedule,
                international_surcharge_schedule=intl_schedule,
                conditions=conditions,
                table=table,
                row=row,
                row_index=idx,
                reference=reference,
            )
        )
    return rules, unclassified, ambiguous


def _fixed_fee_schedule_for(product_id: str, default: str) -> str | None:
    """Return the fixed-fee schedule name for a product, or None if no fixed fee applies."""
    overrides = {
        "goods_and_services": "goods_and_services",
        "donations": "donations",
        "nonprofit": "nonprofit",
        "micropayments": "micropayments",
        "alternative_payment_methods": "alternative_payment_methods",
        "advanced_card_payments": "online_card_payments",
        "pay_later_consumer": "commercial",
        "pos_transactions": None,
        "qr_code_payments": None,
        "guest_checkout": "commercial",
        "invoice_pay_later": "commercial",
        "other_commercial": "commercial",
        "paypal_checkout": "commercial",
    }
    return overrides.get(product_id, default)


def _international_surcharge_schedule_for(product_id: str, default: str) -> str | None:
    """Return the international surcharge schedule name for a product, or None.

    Many products inherit the general commercial schedule; some have their own;
    a few (e.g. QR-code, APM) do not define a separate international surcharge.
    """
    overrides = {
        "goods_and_services": "goods_and_services",
        "donations": "donations",
        "nonprofit": "nonprofit",
        "advanced_card_payments": "commercial",
        "pay_later_consumer": "commercial",
        "alternative_payment_methods": None,
        "micropayments": None,
        "pos_transactions": None,
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
    source: Source | None = None,
) -> tuple[dict[str, FixedFeeSchedule], dict[str, InternationalSurchargeSchedule]]:
    """Extract fixed-fee and international-surcharge schedules.

    Schedules are keyed by product name.  If two tables map to the same product
    (e.g. "Fixed fee by received currency" and "Currency fixed fees" both for
    commercial), their entries are merged and sources are combined.  Conflicting
    duplicate keys are resolved by keeping the first encountered value.
    """
    fixed: dict[str, FixedFeeSchedule] = {}
    international: dict[str, InternationalSurchargeSchedule] = {}

    for table in tables:
        category = _classify_table_category(table)
        if category == "fixed_fee_table":
            schedule = _extract_fixed_fee_schedule(table, source=source)
            if schedule:
                name = _schedule_name_from_table(table, "commercial")
                existing = fixed.get(name)
                if existing:
                    merged_entries = dict(existing.entries)
                    merged_entries.update(schedule.entries)
                    merged_sources = list(existing.sources)
                    for s in schedule.sources:
                        if s not in merged_sources:
                            merged_sources.append(s)
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
                    seen = {e.payer_region for e in merged_entries}
                    for e in schedule.entries:
                        if e.payer_region not in seen:
                            merged_entries.append(e)
                            seen.add(e.payer_region)
                    merged_sources = list(existing.sources)
                    for s in schedule.sources:
                        if s not in merged_sources:
                            merged_sources.append(s)
                    international[name] = InternationalSurchargeSchedule(entries=merged_entries, sources=merged_sources)
                else:
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
    fixed_schedules, international_schedules = _collect_schedules(tables, source=source)

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
        unresolved_rules.append(
            TransactionFeeRule(
                id=extracted.rule_id,
                label=extracted.label,
                percentage=extracted.percentage,
                fixed_fee_schedule=extracted.fixed_fee_schedule,
                international_surcharge_schedule=extracted.international_surcharge_schedule,
                conditions=extracted.conditions,
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
                            reference=extracted.reference,
                            resolved_rate=resolved,
                            source=_provenance(
                                extracted.table,
                                extracted.row,
                                extracted.row_index,
                                source,
                                original_label=extracted.label,
                            ),
                        ),
                        "percentage": percentage,
                    }
                )
                idx = unresolved_rules.index(rule)
                unresolved_rules[idx] = new_rule
                break

    # Resolve any dangling schedule references by falling back to the general
    # commercial schedule when a product-specific schedule is missing.  This keeps
    # models with incomplete source tables (e.g. test fixtures) valid; real markets
    # define their product-specific schedules explicitly.
    for idx, rule in enumerate(unresolved_rules):
        if rule.fixed_fee_schedule and rule.fixed_fee_schedule not in fixed_schedules and "commercial" in fixed_schedules:
            rule = rule.model_copy(update={"fixed_fee_schedule": "commercial"})
        if (
            rule.international_surcharge_schedule
            and rule.international_surcharge_schedule not in international_schedules
            and "commercial" in international_schedules
        ):
            rule = rule.model_copy(update={"international_surcharge_schedule": "commercial"})
        unresolved_rules[idx] = rule

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
