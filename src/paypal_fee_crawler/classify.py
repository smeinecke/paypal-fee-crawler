"""Derive core merchant fees from normalized tables with fail-closed confidence."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

from . import scoring
from .models import (
    Cell,
    CommercialFee,
    CurrencyConversion,
    DerivedFees,
    FixedFees,
    InternationalSurcharge,
    Row,
    Table,
)
from .normalize import clean_text
from .scoring import FeeCategory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassificationCandidate:
    """Evidence-backed classification candidate for a single table."""

    table: Table
    category: FeeCategory
    confidence: float
    evidence: list[str]


# Strong document-id signals. These are corroborated with table content and are
# not treated as sufficient on their own.
# FEETB16 = standard commercial rate table; FEETB18/306/261 = its commercial fixed-fee tables.
_STANDARD_DOC_IDS = {"FEETB16", "FEETB359"}
_FIXED_DOC_IDS = {"FEETB18", "FEETB306", "FEETB261", "FEETB872", "FEETB871", "FEETB354", "FEETB363", "FEETB440", "FEETB441"}
_INTERNATIONAL_DOC_IDS = {"FEETB91", "FEETB100", "FEETB382", "FEETB153", "FEETB533"}
_CONVERSION_DOC_IDS = {"FEETB539", "FEETB128", "FEETB159", "FEETB160", "FEETB154", "FEETB156", "FEETB157", "FEETB338"}


# ----------------------------- text helpers ---------------------------------


def _norm(text: str | None) -> str:
    return clean_text(text or "").lower()


def _table_text(table: Table) -> str:
    """Combined caption, section path, and headers, but not row data."""
    parts = list(table.section_path or []) + [table.caption or ""]
    for header in table.headers:
        parts.append(header.text)
    return _norm(" ".join(parts))


def _row_text(row: Row) -> str:
    return _norm(" ".join(c.text for c in row.cells))


def _first_percentage_in_row(row: Row) -> str | None:
    for cell in row.cells:
        for token in cell.tokens:
            if token.kind == "percentage" and token.value:
                return token.value
    return None


def _first_percentage_in_table(table: Table) -> str | None:
    for row in table.rows:
        val = _first_percentage_in_row(row)
        if val:
            return val
    for header in table.headers:
        for token in header.tokens:
            if token.kind == "percentage" and token.value:
                return token.value
    return None


def _row_has_token_kind(row: Row, kind: str) -> bool:
    for cell in row.cells:
        for token in cell.tokens:
            if token.kind == kind:
                return True
    return False


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(kw in text for kw in keywords)


def _contains_all(text: str, keywords: tuple[str, ...]) -> bool:
    return all(kw in text for kw in keywords)


def _table_doc_id(table: Table) -> str:
    return (table.document_id or "").upper()


# ---------------------------- category detection -----------------------------


# Negative signals that disqualify a table from a given category.
_NEG_STANDARD = (
    "fixed fee",
    "festgebühr",
    "feste gebühr",
    "currency",
    "währung",
    "international",
    "internacional",
    "internacionales",
    "medzinárodné",
    "medzinárodných",
    "medzinárodná",
    "medzinárodný",
    "medzinarodne",
    "medzinarodnych",
    "zahraničné",
    "zahranicne",
    "transacciones internacionales",
    "medzinárodné transakcie",
    "medzinárodných transakcií",
    "zahraničné transakcie",
    "ausland",
    "cross border",
    "cross-border",
    "conversion",
    "conversiones",
    "conversión",
    "umrechnung",
    "wechselkurs",
    "prepočet",
    "donation",
    "spende",
    "charity",
    "nonprofit",
    "non-profit",
    "dispute",
    "chargeback",
    "rückbuchung",
    "rückabwicklung",
    "micropayment",
    "mikrozahlung",
    "inactive",
    "inaktive",
    "point of sale",
    "region groupings",
    "market/region list",
    "market/region groupings",
    "market/regiongroupings",
    "other fees",
    "sonstige gebühren",
    "additional service fee",
    "additional percentage-based fee",
    "zusatzgebühr",
    "príspevky",
    "príspevkov",
    "príspevok",
    "donación",
    "donaciones",
    "donativo",
    "caridad",
    "beneficencia",
    "charitatívnych",
    "charitatívne",
    "charitatívny",
    "alternatívny spôsob platby",
    "alternativny sposob platby",
    "alternatívny spôsob",
    "alternativny sposob",
    "kryptomien",
    "kryptomeny",
    "criptomonedas",
    "cripto",
    "retiro",
    "retirar",
    "retirada",
    "prevod",
    "prevodu",
    "výber",
    "výberu",
    "withdrawal",
    "contracargo",
    "vrátenie sumy",
    "vratenie sumy",
)

_POS_STANDARD = (
    "standard",
    "štandardná",
    "estándar",
    "estandar",
    "commercial",
    "comercial",
    "comerciales",
    "comercio",
    "komerčné",
    "komercne",
    "komerčná",
    "komerčný",
    "domestic",
    "domácich",
    "domacej",
    "inland",
    "inland",
    "transaktion",
    "transaction",
    "transacciones",
    "transacción",
    "transakcií",
    "transakcie",
    "merchant",
    "händler",
    "händlergebühren",
    "merchant fees",
    "online payment",
    "online card",
    "receiving domestic",
    "zahlungsempfang",
    "payPal-gebühren",
    "comisión",
    "comision",
    "poplatok",
    "poplatkov",
    "poplatky",
    "prijímanie",
    "prijimanie",
    "platieb",
    "platieb",
    "platby",
    "sadzba",
    "tarifa",
    "tasas",
    "nacionales",
    "domestic",
)

_POS_STANDARD_HEADER = (
    "payment type",
    "art der transaktion",
    "rate",
    "gebühr",
    "fee",
    "transaktion",
    "sadzba",
    "comisión",
    "comision",
    "tarifa",
    "poplatok",
    "poplatkov",
)


def _is_standard_commercial(table: Table) -> tuple[bool, float, list[str]]:
    text = _table_text(table)
    if _contains_any(text, _NEG_STANDARD):
        return False, 0.0, []
    evidence: list[str] = []
    confidence = 0.0

    doc_id = _table_doc_id(table)
    if doc_id in _STANDARD_DOC_IDS:
        confidence += 0.6
        evidence.append(f"document_id {doc_id} is a known standard-commercial table")

    if _contains_any(text, _POS_STANDARD):
        confidence += 0.3
        evidence.append("caption/section matches standard commercial keywords")

    header_text = _norm(" ".join(h.text for h in table.headers))
    if _contains_any(header_text, _POS_STANDARD_HEADER):
        confidence += 0.2
        evidence.append("headers match standard commercial patterns")

    has_percentage = any(_row_has_token_kind(row, "percentage") for row in table.rows)
    if has_percentage:
        confidence += 0.1
    else:
        # A standard commercial table must contain a percentage.
        return False, 0.0, []

    return confidence >= 0.4, confidence, evidence


_NEG_FIXED = (
    "charity",
    "donation",
    "spende",
    "nonprofit",
    "non-profit",
    "dispute",
    "chargeback",
    "rückbuchung",
    "rückabwicklung",
    "konfliktgebühr",
    "inactive",
    "inaktive",
    "micropayment",
    "mikrozahlung",
    "mikroplatby",
    "mikro",
    "mikroplatieb",
    "micropago",
    "micropagos",
    "point of sale",
    "international",
    "ausland",
    "conversion",
    "umrechnung",
    "other fees",
    "sonstige gebühren",
    "website payments pro",
    "online card",
    "online payment",
    "online-kartenzahlungen",
    "card payment services",
    "ach",
    "disbursement",
    "region",
    "market/region",
    "apm",
    "alternative payment",
    "qr code",
    "qr-code",
    "qr",
    "kód qr",
    "kod qr",
    "invoicing",
    "rechnungskauf",
    "interchange",
    "interchange plus",
    "max cap",
    "maximum fee",
    "minimum fee",
    "instant transfer",
    "sofort",
    "link and confirmation",
    "cryptocurrency",
    "crypto",
    "mindest",
    "höchst",
    "geld für waren",
    "waren und dienstleistungen",
    "ratepay",
    "payout",
    "payouts",
    "hyperwallet",
    "príspevky",
    "príspevkov",
    "príspevok",
    "príspevkami",
    "príspevk",
    "donación",
    "donaciones",
    "donativo",
    "caridad",
    "beneficencia",
    "charitatívnych",
    "charitatívne",
    "charitatívny",
    "charitatívn",
    "alternatívny spôsob platby",
    "alternativny sposob platby",
    "alternatívny spôsob",
    "alternativny sposob",
    "kryptomien",
    "kryptomeny",
    "criptomonedas",
    "cripto",
    "retiro",
    "retirar",
    "retirada",
    "prevod",
    "prevodu",
    "výber",
    "výberu",
    "withdrawal",
    "vyplatení",
    "vyplatenie",
    "vyplatenia",
    "vyplata",
    "výplata",
    "contracargo",
    "vrátenie sumy",
    "vratenie sumy",
    "spor",
    "spory",
    "sporov",
    "disputa",
    "disputas",
    "overenie",
    "overenia",
    "priradenie",
    "karty",
    "verificación",
    "verificacion",
    "verifizierung",
    "asociación",
    "asociacion",
    "confirmación",
    "confirmacion",
    "neaktívny",
    "neaktivny",
    "inactivo",
    "inactive account",
    "maximálna",
    "maximálne",
    "maximalna",
    "maximalne",
    "minimálna",
    "minimálne",
    "minimalna",
    "minimalne",
    "mínimo",
    "mínima",
    "máximo",
    "máxima",
    "mínim",
    "máxim",
    "minimum",
    "maximum",
    "ostatné",
    "ostatne",
    "otros",
    "otras",
    "ďalšie",
    "dalsie",
    "dodatočné",
    "dodatocne",
    "additional",
    "adicional",
    "elektronické šeky",
    "elektronicke seky",
    "e-check",
    "echeck",
    "služby",
    "sluzby",
    "service fee",
    "servicegebühr",
    "poplatok za služby",
    "comisiones por contracargo",
    "comisión por contracargo",
)

_POS_FIXED = (
    "fixed fee",
    "festgebühr",
    "feste gebühr",
    "fixe gebühr",
    "fixed charge",
    "per transaction",
    "pro transaktion",
    "por transacciones",
    "por transacción",
    "za transakcie",
    "za transakcií",
    "based on currency",
    "auf basis der empfangenen währung",
    "währung",
    "currency",
    "moneda",
    "divisa",
    "meny",
    "mena",
    "mien",
    "commercial",
    "geschäftlich",
    "business transaction",
    "comercial",
    "comerciales",
    "komerčné",
    "komercne",
    "komerčná",
    "komerčný",
    "transacciones",
    "transacción",
    "transakcií",
    "transakcie",
    "fija",
    "fijo",
    "fixný",
    "fixná",
    "fixné",
    "comisión",
    "comision",
    "poplatok",
    "poplatkov",
    "poplatky",
)


def _is_fixed_fee(table: Table) -> tuple[bool, float, list[str]]:
    text = _table_text(table)
    if _contains_any(text, _NEG_FIXED):
        return False, 0.0, []
    evidence: list[str] = []
    confidence = 0.0

    doc_id = _table_doc_id(table)
    if doc_id in _FIXED_DOC_IDS:
        confidence += 0.6
        evidence.append(f"document_id {doc_id} is a known fixed-fee table")

    if _contains_any(text, _POS_FIXED):
        confidence += 0.35
        evidence.append("caption/section matches fixed-fee keywords")
    else:
        return False, 0.0, []

    has_money = any(_row_has_token_kind(row, "money") for row in table.rows)
    if has_money:
        confidence += 0.1
    else:
        return False, 0.0, []

    return confidence >= 0.4, confidence, evidence


_NEG_INTERNATIONAL = (
    "fixed fee",
    "festgebühr",
    "currency conversion",
    "währungsumrechnung",
    "conversion",
    "umrechnung",
    "wechselkurs",
    "donation",
    "spende",
    "charity",
    "nonprofit",
    "non-profit",
    "micropayment",
    "mikrozahlung",
    "dispute",
    "chargeback",
    "inactive",
    "inaktive",
    "other fees",
    "sonstige gebühren",
    "region groupings",
    "market/region groupings",
    "market/regiongroupings",
    "market/region list",
    "market list",
    "region list",
    "standard",
    "inland",
    "domestic",
    "geld für waren",
    "waren und dienstleistungen",
    "senden/empfangen",
    "servicegebühr",
    "service fee",
    "personal",
    "apm",
    "alternative payment",
    "qr code",
    "qr-code",
    "online card",
    "online payment",
    "online-kartenzahlungen",
    "card payment services",
    "invoicing",
    "rechnungskauf",
    "ratepay",
    "payout",
    "disbursement",
    "hyperwallet",
    "interchange",
    "interchange plus",
    "blended",
    "max cap",
    "maximum fee",
    "minimum fee",
    "instant transfer",
    "sofort",
    "cryptocurrency",
    "crypto",
    "point of sale",
    "card present",
    "manual card",
    "príspevky",
    "príspevkov",
    "príspevok",
    "príspevkami",
    "príspevk",
    "donación",
    "donaciones",
    "donativo",
    "caridad",
    "beneficencia",
    "charitatívnych",
    "charitatívne",
    "charitatívny",
    "charitatívn",
    "kryptomien",
    "kryptomeny",
    "criptomonedas",
    "cripto",
    "prepočet",
    "prepocet",
    "conversión",
    "zostatku",
    "zostatok",
    "prijímanie platieb",
    "prijimanie platieb",
    "krajiny kupujúcich",
    "krajiny kupujucich",
    "kupujúcich",
    "kupujucich",
    "služby",
    "sluzby",
    "poplatok za služby",
    "poplatok za sluzby",
)

_POS_INTERNATIONAL = (
    "international",
    "internacional",
    "internacionales",
    "medzinárodné",
    "medzinárodných",
    "medzinárodná",
    "medzinárodný",
    "medzinarodne",
    "medzinarodnych",
    "zahraničné",
    "zahranicne",
    "cross border",
    "cross-border",
    "ausland",
    "auslandszahlung",
    "grenzüberschreitend",
    "zusatzgebühr",
    "additional percentage",
    "adicional",
    "adicionales",
    "dodatočný",
    "dodatočná",
    "dodatočné",
    "dodatocny",
    "dodatocna",
    "dodatocne",
    "percentuálny",
    "percentuálna",
    "percentuálne",
    "percentualny",
    "payer region",
    "markt/region",
    "market/region",
    "markt/das gebiet",
    "region",
    "región",
    "trh",
    "trhy",
    "oblasť",
    "oblast",
    "mercado",
    "mercados",
    "krajina",
    "krajiny",
    "teritórium",
    "teritorium",
    "vendedor",
    "comprador",
    "predávajúci",
    "predavajuci",
    "kupujúci",
    "kupujuci",
    "odosielateľ",
    "odosielatel",
    "príjemca",
    "prijemca",
    "odberateľ",
    "odberatel",
    "recepción",
    "prijímanie",
    "prijimanie",
    "transacciones internacionales",
    "medzinárodné transakcie",
    "medzinárodných transakcií",
    "zahraničné transakcie",
)


def _is_international_surcharge(table: Table) -> tuple[bool, float, list[str]]:
    text = _table_text(table)
    doc_id = _table_doc_id(table)

    # Known international-surcharge tables bypass the broad negative-signal filter.
    # PayPal has added new table IDs (e.g., FEETB100 for GB, FEETB382 for DE) whose
    # captions also contain negative-signal phrases like "service fee".
    if doc_id in _INTERNATIONAL_DOC_IDS:
        evidence: list[str] = [f"document_id {doc_id} is a known international-surcharge table"]
        confidence = 0.6
        if _contains_any(text, _POS_INTERNATIONAL):
            confidence += 0.35
            evidence.append("caption/section matches international-surcharge keywords")
        header_text = _norm(" ".join(h.text for h in table.headers))
        if _contains_any(header_text, _POS_INTERNATIONAL):
            confidence += 0.2
            evidence.append("headers match international-surcharge patterns")
        return confidence >= 0.4, confidence, evidence

    if _contains_any(text, _NEG_INTERNATIONAL):
        return False, 0.0, []
    evidence: list[str] = []
    confidence = 0.0

    if _contains_any(text, _POS_INTERNATIONAL):
        confidence += 0.35
        evidence.append("caption/section matches international-surcharge keywords")
    else:
        return False, 0.0, []

    header_text = _norm(" ".join(h.text for h in table.headers))
    if _contains_any(header_text, _POS_INTERNATIONAL):
        confidence += 0.2
        evidence.append("headers match international-surcharge patterns")

    has_percentage = any(_row_has_token_kind(row, "percentage") for row in table.rows)
    if has_percentage:
        confidence += 0.1

    return confidence >= 0.4, confidence, evidence


_NEG_CONVERSION = (
    "fixed fee",
    "festgebühr",
    "donation",
    "spende",
    "charity",
    "nonprofit",
    "dispute",
    "chargeback",
    "micropayment",
    "other fees",
    "sonstige gebühren",
    "point of sale",
    "mindest",
    "höchst",
    "max cap",
    "minimum fee",
    "maximum fee",
    "príspevky",
    "príspevkov",
    "príspevok",
    "príspevkami",
    "príspevk",
    "donación",
    "donaciones",
    "donativo",
    "caridad",
    "beneficencia",
    "charitatívnych",
    "charitatívne",
    "charitatívny",
    "charitatívn",
    "kryptomien",
    "kryptomeny",
    "criptomonedas",
    "cripto",
    "prevod",
    "prevodu",
    "výber",
    "výberu",
    "withdrawal",
    "retiro",
    "retirar",
    "retirada",
    "contracargo",
    "vrátenie sumy",
    "vratenie sumy",
    "comisión fija",
    "fija",
    "fixný poplatok",
    "fixny poplatok",
    "fixná",
    "fixne",
)

_POS_CONVERSION = (
    "currency conversion",
    "converting balance",
    "währungsumrechnung",
    "umrechnung",
    "wechselkurs",
    "conversion",
    "conversiones",
    "conversión",
    "spread",
    "base exchange rate",
    "basiswechselkurs",
    "moneda",
    "divisa",
    "divisas",
    "prepočet",
    "prepocet",
    "prepočet meny",
    "zmena",
    "zmena meny",
    "zmenárne",
    "meny",
    "mena",
    "mien",
    "devízový",
    "devízový kurz",
    "výmenný",
    "výmenný kurz",
    "vymenny kurz",
    "tipo de cambio",
    "tipos de cambio",
    "cambio",
    "tasas de cambio",
    "foreign exchange",
    "exchange rate",
)


def _is_currency_conversion(table: Table) -> tuple[bool, float, list[str]]:
    text = _table_text(table)
    if _contains_any(text, _NEG_CONVERSION):
        return False, 0.0, []
    evidence: list[str] = []
    confidence = 0.0

    doc_id = _table_doc_id(table)
    if doc_id in _CONVERSION_DOC_IDS:
        confidence += 0.6
        evidence.append(f"document_id {doc_id} is a known currency-conversion table")

    if _contains_any(text, _POS_CONVERSION):
        confidence += 0.35
        evidence.append("caption/section matches currency-conversion keywords")
    else:
        return False, 0.0, []

    header_text = _norm(" ".join(h.text for h in table.headers))
    if _contains_any(header_text, _POS_CONVERSION):
        confidence += 0.2
        evidence.append("headers match currency-conversion patterns")

    has_percentage = any(_row_has_token_kind(row, "percentage") for row in table.rows)
    if has_percentage:
        confidence += 0.1

    return confidence >= 0.4, confidence, evidence


_OTHER_KEYWORDS = {
    "micropayment": ("micropayment", "mikrozahlung", "kleinbetragszahlung"),
    "donation": (
        "donation",
        "spende",
        "charity donation",
        "príspevky",
        "príspevkov",
        "príspevok",
        "donación",
        "donaciones",
        "donativo",
        "caridad",
        "beneficencia",
    ),
    "nonprofit": ("nonprofit", "non-profit", "gemeinnützig", "gemeinnutzig", "charitatívnych", "charitatívne"),
    "chargeback": ("chargeback", "rückbuchung", "rückabwicklung", "rücklastschrift"),
    "dispute": ("dispute", "streitfall", "konfliktlösung"),
    "alternative": (
        "alternative",
        "alternatívny",
        "alternativny",
        "alternatívny spôsob",
        "alternatívny spôsob platby",
        "alternativny sposob platby",
    ),
}


def _other_category(table: Table) -> str | None:
    text = _table_text(table)
    for category, keywords in _OTHER_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return category
    return None


# ------------------------- value extraction ---------------------------------


# Row labels for a standard commercial transaction fee row.
_STD_ROW_INCLUDE = (
    "standard",
    "štandardná",
    "standardna",
    "estándar",
    "estandar",
    "payPal checkout",
    "checkout",
    "commercial",
    "comercial",
    "comerciales",
    "comercio",
    "komerčné",
    "komercne",
    "komerčná",
    "komerčný",
    "transaction",
    "transaktion",
    "transacciones",
    "transacción",
    "transakcií",
    "transakcie",
    "transakciou",
    "payments",
    "payment",
    "zahlung",
    "platby",
    "platieb",
    "platba",
    "nutzer",
    "user",
    "online",
    "card",
    "händler",
    "merchant",
    "other",
    "alle anderen",
    "nutzer",
    "advanced",
    "all",
    "all other",
    "all transactions",
    "all commercial",
    "all commercial transactions",
    "all payment",
    "all payments",
    "todos",
    "todas",
    "demás",
    "demas",
    "otros",
    "otras",
    "mercados",
    "países",
    "paises",
    "všetky",
    "vsetky",
    "ostatné",
    "ostatne",
    "obchodné",
    "obchodne",
    "trhy",
    "trhov",
    "krajín",
    "krajiny",
    "teritórií",
    "teritorii",
    "oblastí",
    "oblasti",
    "todos los demás",
    "todos los demas",
    "todos los demás mercados",
    "todos los demas mercados",
    "todas las demás",
    "todas las demas",
    "todas las demás mercados",
    "todas las demas mercados",
    "všetky ostatné",
    "vsetky ostatne",
    "všetky ostatné trhy",
    "vsetky ostatne trhy",
    "všetky ostatné komerčné",
    "vsetky ostatne komercne",
    "všetky ostatné transakcie",
    "vsetky ostatne transakcie",
    "resto del mundo",
    "resto de los mercados",
    "restantes",
    "restante",
)

_STD_ROW_EXCLUDE = (
    "qr",
    "qr-code",
    "qr code",
    "kód qr",
    "kod qr",
    "charity",
    "nonprofit",
    "non-profit",
    "donation",
    "spende",
    "mikro",
    "micropayment",
    "apm",
    "ach",
    "capped",
    "ratepay",
    "blended",
    "interchange",
    "interchange plus",
    "american express",
    "amex",
    "virtual terminal",
    "sending",
    "geld senden",
    "geld für waren",
    "point of sale",
    "card present",
    "manual card",
    "pay later",
    "venmo",
    "card funded",
    "without a paypal account",
    "send/receive",
    "goods and services",
)

# Localized "all other markets / transactions" labels used when the standard table
# lists rows by market group rather than by payment type.
_STD_ROW_FALLBACK = (
    "all other",
    "alle anderen",
    "todos los demás",
    "todos los demas",
    "todos los demás mercados",
    "todos los demas mercados",
    "todas las demás",
    "todas las demas",
    "todas las demás mercados",
    "todas las demas mercados",
    "všetky ostatné",
    "vsetky ostatne",
    "všetky ostatné trhy",
    "vsetky ostatne trhy",
    "všetky ostatné komerčné",
    "vsetky ostatne komercne",
    "všetky ostatné transakcie",
    "vsetky ostatne transakcie",
    "toutes les autres",
    "tutte le altre",
    "todos los demás casos",
    "todos los demas casos",
    "rest of the world",
    "rest of world",
    "rest of the markets",
    "rest of markets",
)


def _cell_text_starts_with(cell: Cell) -> str:
    return _norm(cell.text.split()[0]) if cell.text.split() else ""


def _extract_standard_percentage(table: Table, market_code: str | None = None) -> tuple[str | None, list[str]]:
    """Return the most confident standard-commercial percentage in a table.

    If the table lists rows by market or region group (e.g. PayPal's localized
    country tables), we fall back to the row containing the target market code or
    a localized "all other markets" row.
    """
    evidence: list[str] = []
    matched_percentages: list[str] = []
    fallback_percentages: list[str] = []
    for row in table.rows:
        pct = _first_percentage_in_row(row)
        if not pct:
            continue
        first_cell_text = _norm(row.cells[0].text) if row.cells else ""
        all_text = _row_text(row)
        if _contains_any(first_cell_text, _STD_ROW_EXCLUDE) or _contains_any(all_text, _STD_ROW_EXCLUDE):
            continue
        is_include = _contains_any(first_cell_text, _STD_ROW_INCLUDE) or _contains_any(all_text, _STD_ROW_INCLUDE)
        # Fallback: the row explicitly references the market being crawled
        # or a localized "all other markets" catch-all.
        is_fallback = (
            bool(market_code and (market_code in first_cell_text or market_code in all_text))
            or _contains_any(first_cell_text, _STD_ROW_FALLBACK)
            or _contains_any(all_text, _STD_ROW_FALLBACK)
        )
        if is_include:
            matched_percentages.append(pct)
        if is_fallback:
            fallback_percentages.append(pct)

    if not matched_percentages and not fallback_percentages:
        evidence.append("no standard-commercial row matched")
        return None, evidence

    # Prefer the country-specific or all-other fallback row; otherwise fall back to
    # the broad keyword-based matches.
    if fallback_percentages:
        evidence.append("standard percentage selected from market/all-other fallback row")
        matched_percentages = fallback_percentages

    counter = Counter(matched_percentages)
    selected = counter.most_common(1)[0][0]
    evidence.append(f"standard percentage {selected} from {len(matched_percentages)} matched row(s)")
    return selected, evidence


def _extract_fixed_fees(table: Table) -> list[FixedFees]:
    """Extract money tokens from a fixed-fee table."""
    fees: list[FixedFees] = []
    for row in table.rows:
        for cell in row.cells:
            for token in cell.tokens:
                if token.kind == "money" and token.amount and token.currency:
                    fees.append(FixedFees(currency=token.currency, amount=token.amount))
    return fees


def _normalize_region(text: str) -> str | None:
    """Map a region cell to one of the canonical surcharge regions."""
    t = _norm(text)
    if not t:
        return None

    if (
        "european economic" in t
        or "european economic area" in t
        or "ewr" in t
        or "eea" in t
        or "ehp" in t
        or "e.u" in t
        or t == "eu"
    ):
        return "EEA"
    if (
        "united kingdom" in t
        or t == "gb"
        or "uk" in t
        or "großbritannien" in t
        or "great britain" in t
        or "britain" in t
        or "england" in t
        or "spojené kráľovstvo" in t
        or "spojene kralovstvo" in t
        or "royaume-uni" in t
    ):
        return "GB"
    if (
        "united states" in t
        or "usa" in t
        or "u.s" in t
        or t == "us"
        or "canada" in t
        or "spojené štáty" in t
        or "spojene staty" in t
    ):
        return "US_CA"
    if "all" in t and "other" in t:
        return "OTHER"
    if "rest" in t and "world" in t:
        return "OTHER"
    if "all commercial" in t or "all payment" in t or "commercial transactions" in t:
        return "OTHER"
    if (
        "other" in t
        or "andere" in t
        or "rest" in t
        or ("todos" in t and ("demás" in t or "demas" in t))
        or ("todas" in t and ("demás" in t or "demas" in t))
        or ("všetky" in t and ("ostatné" in t or "ostatne" in t))
        or ("vsetky" in t and ("ostatne" in t or "ostatne" in t))
        or "todos los mercados" in t
        or "todas las mercados" in t
        or "všetky trhy" in t
        or "vsetky trhy" in t
        or "restantes" in t
        or "otros mercados" in t
        or "otras mercados" in t
    ):
        return "OTHER"
    return None


def _extract_international_surcharges(
    table: Table, market_code: str | None = None
) -> list[InternationalSurcharge]:
    """Extract region->percentage rows from an international-surcharge table.

    When the table lists rows by market group, the row containing the market
    being crawled is preferred; if the market is not present, the row that
    contains a localized "all other" label is used.
    """
    surcharges: list[InternationalSurcharge] = []
    fallback_rows: list[tuple[str, str]] = []
    for row in table.rows:
        percentage: str | None = None
        for cell in row.cells:
            for token in cell.tokens:
                if token.kind == "percentage" and token.value:
                    percentage = token.value
                    break
            if percentage:
                break
        if not percentage:
            continue

        first_cell_text = _norm(row.cells[0].text) if row.cells else ""

        # Prefer the row explicitly referencing the target market.
        if market_code and market_code in first_cell_text:
            region = _normalize_region(first_cell_text) or "OTHER"
            return [InternationalSurcharge(region=region, percentage_points=percentage)]

        # Collect localized "all other" rows as fallback.
        if _contains_any(first_cell_text, _STD_ROW_FALLBACK):
            fallback_rows.append((first_cell_text, percentage))
            continue

        # The region is determined by the first cell only (avoiding columns that
        # are always buyer/receiver market labels like "all markets").
        region = _normalize_region(first_cell_text)
        if region:
            surcharges.append(InternationalSurcharge(region=region, percentage_points=percentage))

    if fallback_rows:
        # Use the first "all other" row found.
        first_fallback_text, first_fallback_pct = fallback_rows[0]
        region = _normalize_region(first_fallback_text) or "OTHER"
        surcharges.append(InternationalSurcharge(region=region, percentage_points=first_fallback_pct))

    return surcharges


def _extract_conversion_spread(table: Table) -> str | None:
    """Extract the commercial currency-conversion spread from a table.

    Tables sometimes list both a personal/family/payout rate and a business/
    all-other rate. This function prefers the commercial rate.
    """
    personal_indicators = (
        "friend",
        "family",
        "personal",
        "goods or services",
        "waren oder dienstleistungen",
        "payout",
        "auszahlung",
        "privat",
        "amigos",
        "familia",
        "osobné",
        "osobne",
        "súkromné",
        "sukromne",
        "osobný",
        "osobny",
    )
    commercial_indicators = (
        "all other",
        "alle anderen",
        "business",
        "geschäftlich",
        "base exchange rate",
        "basiswechselkurs",
        "above the base",
        "über dem basis",
        "todos los demás",
        "todos los demas",
        "všetky ostatné",
        "vsetky ostatne",
        "todos los casos",
        "todos los demas casos",
        "všetky ostatné prípady",
        "vsetky ostatne pripady",
        "firemný",
        "firemny",
        "firemné",
        "firemne",
        "obchodné",
        "obchodne",
        "comercial",
        "comerciales",
        "comercio",
        "komerčné",
        "komercne",
        "base exchange rate",
        "základného výmenného kurzu",
        "zakladneho vymenneho kurzu",
        "nad rámec",
        "nad ramec",
        "nad rámec základného",
        "above base",
        "above the base exchange rate",
    )

    commercial_values: list[str] = []
    all_values: list[str] = []

    for row in table.rows:
        row_text = _row_text(row)
        for cell in row.cells:
            for token in cell.tokens:
                if token.kind == "percentage" and token.value:
                    all_values.append(token.value)
                    if _contains_any(row_text, personal_indicators):
                        continue
                    if _contains_any(row_text, commercial_indicators):
                        commercial_values.append(token.value)

    if commercial_values:
        counter = Counter(commercial_values)
        return counter.most_common(1)[0][0]
    if all_values:
        counter = Counter(all_values)
        return counter.most_common(1)[0][0]
    return None


# ----------------------------- aggregation ---------------------------------


def _aggregate_fixed_fees(
    candidates: list[ClassificationCandidate],
) -> tuple[list[FixedFees], list[str], list[str]]:
    """Aggregate fixed fees from candidates, reporting conflicts and evidence."""
    evidence: list[str] = []
    warnings: list[str] = []
    all_fees: list[FixedFees] = []
    for candidate in candidates:
        fees = _extract_fixed_fees(candidate.table)
        if fees:
            all_fees.extend(fees)
            evidence.append(
                f"fixed-fee table {candidate.table.document_id or candidate.table.caption}: {len(fees)} row(s)"
            )

    amounts: dict[str, set[str]] = {}
    for fee in all_fees:
        amounts.setdefault(fee.currency, set()).add(fee.amount)

    result: list[FixedFees] = []
    for currency, values in sorted(amounts.items()):
        if len(values) == 1:
            result.append(FixedFees(currency=currency, amount=next(iter(values))))
        else:
            warnings.append(f"conflicting fixed-fee values for {currency}: {sorted(values)}")

    return result, evidence, warnings


def _aggregate_international_surcharges(
    candidates: list[ClassificationCandidate],
    market_code: str | None = None,
) -> tuple[list[InternationalSurcharge], list[str], list[str]]:
    evidence: list[str] = []
    warnings: list[str] = []
    all_surcharges: list[InternationalSurcharge] = []
    for candidate in candidates:
        surcharges = _extract_international_surcharges(candidate.table, market_code)
        if surcharges:
            all_surcharges.extend(surcharges)
            evidence.append(
                f"international-surcharge table {candidate.table.document_id or candidate.table.caption}: {surcharges}"
            )

    values: dict[str, set[str]] = {}
    for s in all_surcharges:
        if s.percentage_points is not None:
            values.setdefault(s.region, set()).add(s.percentage_points)

    result: list[InternationalSurcharge] = []
    for region, points in sorted(values.items()):
        if len(points) == 1:
            result.append(InternationalSurcharge(region=region, percentage_points=next(iter(points))))
        else:
            warnings.append(f"conflicting surcharge percentages for {region}: {sorted(points)}")

    return result, evidence, warnings


def _aggregate_conversion(
    candidates: list[ClassificationCandidate],
) -> tuple[CurrencyConversion | None, list[str], list[str]]:
    evidence: list[str] = []
    warnings: list[str] = []
    values: list[str] = []
    for candidate in candidates:
        spread = _extract_conversion_spread(candidate.table)
        if spread:
            values.append(spread)
            evidence.append(
                f"currency-conversion table {candidate.table.document_id or candidate.table.caption}: spread {spread}"
            )
    if not values:
        return None, evidence, warnings
    counter = Counter(values)
    if len(counter) == 1:
        return CurrencyConversion(spread_percentage=values[0]), evidence, warnings
    warnings.append(f"conflicting conversion spreads: {sorted(set(values))}")
    return None, evidence, warnings


# ----------------------------- public API ---------------------------------


def _classify_table(table: Table) -> ClassificationCandidate | None:
    """Return the best category candidate for a table, or None if it is too ambiguous."""
    if not table.rows and not table.headers:
        return None

    other = _other_category(table)
    if other:
        return ClassificationCandidate(
            table=table,
            category=FeeCategory.OTHER,
            confidence=0.5,
            evidence=[f"other known category: {other}"],
        )

    for check, category in (
        (_is_standard_commercial, FeeCategory.STANDARD_COMMERCIAL),
        (_is_fixed_fee, FeeCategory.FIXED_FEE),
        (_is_international_surcharge, FeeCategory.INTERNATIONAL_SURCHARGE),
        (_is_currency_conversion, FeeCategory.CURRENCY_CONVERSION),
    ):
        matched, confidence, evidence = check(table)
        if matched:
            return ClassificationCandidate(table=table, category=category, confidence=confidence, evidence=evidence)

    return None


def classify_tables(tables: list[Table], market_code: str | None = None, locale: str | None = None) -> DerivedFees:
    """Derive core fees from normalized tables using the legacy classifier.

    ``locale`` is accepted for API compatibility but is not used by the legacy
    classifier.
    """
    candidates, other_categories = _classify_all_tables(tables)
    return _derive_from_candidates(candidates, market_code, other_categories)


def _classify_all_tables(tables: list[Table]) -> tuple[list[ClassificationCandidate], list[str]]:
    """Classify each table using the legacy predicates and collect candidates."""
    candidates: list[ClassificationCandidate] = []
    other_categories: list[str] = []

    for table in tables:
        candidate = _classify_table(table)
        if candidate is None:
            continue
        candidates.append(candidate)
        if candidate.category == FeeCategory.OTHER:
            category = _other_category(table)
            if category:
                other_categories.append(category)

    return candidates, other_categories


def _derive_from_candidates(
    candidates: list[ClassificationCandidate],
    market_code: str | None,
    other_categories: list[str],
) -> DerivedFees:
    """Extract fees from a list of already-classified candidates."""
    evidence: list[str] = []
    warnings: list[str] = []

    # Group by category.
    by_category: dict[FeeCategory, list[ClassificationCandidate]] = {
        FeeCategory.STANDARD_COMMERCIAL: [],
        FeeCategory.FIXED_FEE: [],
        FeeCategory.INTERNATIONAL_SURCHARGE: [],
        FeeCategory.CURRENCY_CONVERSION: [],
    }
    for candidate in candidates:
        if candidate.category in by_category:
            by_category[candidate.category].append(candidate)
        if candidate.category != FeeCategory.OTHER:
            evidence.extend(candidate.evidence)

    # Standard commercial.
    standard_percentage: str | None = None
    standard_candidates = by_category[FeeCategory.STANDARD_COMMERCIAL]
    if standard_candidates:
        standard_candidates.sort(key=lambda c: c.confidence, reverse=True)
        selected = standard_candidates[0]
        standard_percentage, pct_evidence = _extract_standard_percentage(selected.table, market_code)
        if standard_percentage:
            evidence.extend(pct_evidence)
            evidence.append(f"standard_commercial table {selected.table.document_id or selected.table.caption}")

    # Fixed fees.
    fixed_fees, fixed_evidence, fixed_warnings = _aggregate_fixed_fees(by_category[FeeCategory.FIXED_FEE])
    evidence.extend(fixed_evidence)
    warnings.extend(fixed_warnings)

    # International surcharges.
    surcharges, intl_evidence, intl_warnings = _aggregate_international_surcharges(
        by_category[FeeCategory.INTERNATIONAL_SURCHARGE], market_code
    )
    evidence.extend(intl_evidence)
    warnings.extend(intl_warnings)

    # Currency conversion.
    conversion, conv_evidence, conv_warnings = _aggregate_conversion(by_category[FeeCategory.CURRENCY_CONVERSION])
    evidence.extend(conv_evidence)
    warnings.extend(conv_warnings)

    # Determine which categories are exposed and whether they produced reliable data.
    exposed_categories: set[FeeCategory] = {
        category for category, category_candidates in by_category.items() if category_candidates
    }

    if by_category[FeeCategory.STANDARD_COMMERCIAL] and not standard_percentage:
        warnings.append("standard-commercial section exposed but no reliable percentage extracted")
    if by_category[FeeCategory.FIXED_FEE] and not fixed_fees:
        warnings.append("fixed-fee section exposed but no reliable values extracted")
    if by_category[FeeCategory.INTERNATIONAL_SURCHARGE] and not surcharges:
        warnings.append("international-surcharge section exposed but no reliable values extracted")
    if by_category[FeeCategory.CURRENCY_CONVERSION] and not conversion:
        warnings.append("currency-conversion section exposed but no reliable spread extracted")

    warnings = sorted(set(warnings))

    has_standard = standard_percentage is not None
    has_fixed = bool(fixed_fees)
    intl_exposed = bool(by_category[FeeCategory.INTERNATIONAL_SURCHARGE])
    conv_exposed = bool(by_category[FeeCategory.CURRENCY_CONVERSION])
    if not exposed_categories:
        status = "unclassified"
    elif warnings:
        status = "partial"
    elif has_standard and has_fixed and (not intl_exposed or surcharges) and (not conv_exposed or conversion):
        status = "complete"
    else:
        status = "partial"

    return DerivedFees(
        status=status,
        standard_commercial=CommercialFee(
            percentage=standard_percentage,
            fixed_fee_reference="commercial_fixed_fees" if fixed_fees else None,
        )
        if standard_percentage
        else None,
        commercial_fixed_fees=fixed_fees,
        international_surcharges=surcharges,
        currency_conversion=conversion,
        international_surcharge_exposed=intl_exposed,
        currency_conversion_exposed=conv_exposed,
        unclassified_sections=sorted(set(other_categories)),
        classification_evidence=evidence + warnings,
    )


class ObservationKind(StrEnum):
    LOW_MARGIN = "low_margin"
    LEXICAL_ONLY_DECISION = "lexical_only_decision"
    EXTRACTION_CONFLICT = "extraction_conflict"
    UNKNOWN_DOCUMENT_ID = "unknown_document_id"
    UNKNOWN_FINGERPRINT = "unknown_fingerprint"


@dataclass(frozen=True)
class ClassificationObservation:
    kind: ObservationKind
    category: FeeCategory | None = None
    table_id: str | None = None
    message: str = ""


@dataclass(frozen=True)
class ClassificationRun:
    derived: DerivedFees
    table_decisions: tuple[scoring.ClassificationDecision, ...]
    observations: tuple[ClassificationObservation, ...]
    classifier_version: str


def classify_legacy(
    tables: list[Table],
    market_code: str | None = None,
    locale: str | None = None,
) -> ClassificationRun:
    """Run the legacy classifier and wrap the result in a common internal format."""
    candidates, other_categories = _classify_all_tables(tables)
    derived = _derive_from_candidates(candidates, market_code, other_categories)
    return ClassificationRun(
        derived=derived,
        table_decisions=(),
        observations=(),
        classifier_version="legacy",
    )


def classify_structural(
    tables: list[Table],
    market_code: str | None = None,
    locale: str | None = None,
) -> ClassificationRun:
    """Run the structural scoring classifier and produce derived fees.

    This function uses the scoring engine in ``scoring.py`` to select a category
    for each table, then extracts values using the existing extraction helpers.
    """
    candidates: list[ClassificationCandidate] = []
    decisions: list[scoring.ClassificationDecision] = []
    observations: list[ClassificationObservation] = []

    for table in tables:
        if not table.rows and not table.headers:
            continue

        scores = scoring.score_all_categories(table, market_code, locale)
        decision = scoring.select_category(scores)
        decisions.append(decision)

        if decision.status == "selected" and decision.selected_category is not None:
            score = decision.ranked_scores[0].score
            candidate = ClassificationCandidate(
                table=table,
                category=decision.selected_category,
                confidence=score / scoring.MAX_CATEGORY_SCORE,
                evidence=[s.detail or s.code for s in decision.ranked_scores[0].signals],
            )
            candidates.append(candidate)

            if score < scoring.MINIMUM_SCORE + scoring.MINIMUM_MARGIN:
                observations.append(
                    ClassificationObservation(
                        kind=ObservationKind.LOW_MARGIN,
                        category=decision.selected_category,
                        table_id=table.document_id or table.table_id,
                        message=f"selected with low margin: {score}",
                    )
                )

            signal_sources = {s.source for s in decision.ranked_scores[0].signals}
            if signal_sources == {scoring.EvidenceSource.LEXICAL}:
                observations.append(
                    ClassificationObservation(
                        kind=ObservationKind.LEXICAL_ONLY_DECISION,
                        category=decision.selected_category,
                        table_id=table.document_id or table.table_id,
                        message="selected based only on lexical evidence",
                    )
                )

            if decision.selected_category is scoring.FeeCategory.CURRENCY_CONVERSION and not {
                scoring.EvidenceCode.KNOWN_DOCUMENT_ID,
                scoring.EvidenceCode.METADATA_KEY_MATCH,
                scoring.EvidenceCode.INTERNAL_NAME_MATCH,
                scoring.EvidenceCode.KNOWN_FINGERPRINT,
            } & {s.code for s in decision.ranked_scores[0].signals}:
                observations.append(
                    ClassificationObservation(
                        kind=ObservationKind.UNKNOWN_FINGERPRINT,
                        category=decision.selected_category,
                        table_id=table.document_id or table.table_id,
                        message="conversion selected without approved registry evidence",
                    )
                )

    other_categories: list[str] = []
    derived = _derive_from_candidates(candidates, market_code, other_categories)
    return ClassificationRun(
        derived=derived,
        table_decisions=tuple(decisions),
        observations=tuple(observations),
        classifier_version="structural-1",
    )
