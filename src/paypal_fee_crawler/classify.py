"""Derive core merchant fees from normalized tables with fail-closed confidence."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field

from . import scoring
from .extraction import (
    ClassificationObservation,
    ObservationKind,
    extract_conversion_spread,
    extract_fixed_fees,
    extract_international_surcharges,
    extract_standard_percentage,
)
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
from .profiles import NormalizedTableRecord, TableContext, TableProfile, build_table_profile
from .registry import FingerprintBuilder, FingerprintRegistry
from .scoring import FeeCategory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClassificationCandidate:
    """Evidence-backed classification candidate for a single table."""

    table: Table
    category: FeeCategory
    confidence: float
    evidence: list[str]
    profile: TableProfile | None = None
    contexts: tuple[TableContext, ...] = ()
    approval_codes: set[str] = field(default_factory=set)


# Strong document-id signals. These are corroborated with table content and are
# not treated as sufficient on their own.
# FEETB16 = standard commercial rate table; FEETB18/306/261 = its commercial fixed-fee tables.
_STANDARD_DOC_IDS = {"FEETB16", "FEETB359"}
_FIXED_DOC_IDS = {
    "FEETB18",
    "FEETB306",
    "FEETB261",
    "FEETB872",
    "FEETB871",
    "FEETB354",
    "FEETB363",
    "FEETB440",
    "FEETB441",
}
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


_REGION_EXACT: dict[str, str] = {
    "eu": "EEA",
    "gb": "GB",
    "us": "US_CA",
}

_REGION_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("european economic",), "EEA"),
    (("ewr",), "EEA"),
    (("eea",), "EEA"),
    (("ehp",), "EEA"),
    (("e.u",), "EEA"),
    (("united kingdom",), "GB"),
    (("uk",), "GB"),
    (("großbritannien",), "GB"),
    (("great britain",), "GB"),
    (("britain",), "GB"),
    (("england",), "GB"),
    (("spojené kráľovstvo",), "GB"),
    (("spojene kralovstvo",), "GB"),
    (("royaume-uni",), "GB"),
    (("united states",), "US_CA"),
    (("usa",), "US_CA"),
    (("u.s",), "US_CA"),
    (("canada",), "US_CA"),
    (("spojené štáty",), "US_CA"),
    (("spojene staty",), "US_CA"),
    (("all", "other"), "OTHER"),
    (("rest", "world"), "OTHER"),
    (("all commercial",), "OTHER"),
    (("all payment",), "OTHER"),
    (("commercial transactions",), "OTHER"),
    (("other",), "OTHER"),
    (("andere",), "OTHER"),
    (("rest",), "OTHER"),
    (("todos", "demás"), "OTHER"),
    (("todos", "demas"), "OTHER"),
    (("todas", "demás"), "OTHER"),
    (("todas", "demas"), "OTHER"),
    (("všetky", "ostatné"), "OTHER"),
    (("všetky", "ostatne"), "OTHER"),
    (("vsetky", "ostatne"), "OTHER"),
    (("todos los mercados",), "OTHER"),
    (("todas las mercados",), "OTHER"),
    (("všetky trhy",), "OTHER"),
    (("vsetky trhy",), "OTHER"),
    (("restantes",), "OTHER"),
    (("otros mercados",), "OTHER"),
    (("otras mercados",), "OTHER"),
)


def _normalize_region(text: str) -> str | None:
    """Map a region cell to one of the canonical surcharge regions."""
    t = _norm(text)
    if not t:
        return None
    if t in _REGION_EXACT:
        return _REGION_EXACT[t]
    for patterns, region in _REGION_RULES:
        if all(pattern in t for pattern in patterns):
            return region
    return None


def _extract_international_surcharges(table: Table, market_code: str | None = None) -> list[InternationalSurcharge]:
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
    return _derive_from_candidates(candidates, market_code, other_categories)[0]


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


def _append_exposure_warnings(
    warnings: list[str],
    by_category: dict[FeeCategory, list[ClassificationCandidate]],
    standard_percentage: str | None,
    fixed_fees: list[FixedFees],
    surcharges: list[InternationalSurcharge],
    conversion: CurrencyConversion | None,
) -> list[str]:
    if by_category[FeeCategory.STANDARD_COMMERCIAL] and not standard_percentage:
        warnings.append("standard-commercial section exposed but no reliable percentage extracted")
    if by_category[FeeCategory.FIXED_FEE] and not fixed_fees:
        warnings.append("fixed-fee section exposed but no reliable values extracted")
    if by_category[FeeCategory.INTERNATIONAL_SURCHARGE] and not surcharges:
        warnings.append("international-surcharge section exposed but no reliable values extracted")
    if by_category[FeeCategory.CURRENCY_CONVERSION] and not conversion:
        warnings.append("currency-conversion section exposed but no reliable spread extracted")
    return warnings


def _derive_status(
    standard_percentage: str | None,
    fixed_fees: list[FixedFees],
    surcharges: list[InternationalSurcharge],
    conversion: CurrencyConversion | None,
    by_category: dict[FeeCategory, list[ClassificationCandidate]],
    warnings: list[str],
) -> str:
    exposed_categories = {category for category, category_candidates in by_category.items() if category_candidates}
    if not exposed_categories:
        return "unclassified"
    if warnings:
        return "partial"
    if (
        standard_percentage
        and fixed_fees
        and (not by_category[FeeCategory.INTERNATIONAL_SURCHARGE] or surcharges)
        and (not by_category[FeeCategory.CURRENCY_CONVERSION] or conversion)
    ):
        return "complete"
    return "partial"


def _build_derived_fees(
    status: str,
    standard_percentage: str | None,
    fixed_fees: list[FixedFees],
    surcharges: list[InternationalSurcharge],
    conversion: CurrencyConversion | None,
    intl_exposed: bool,
    conv_exposed: bool,
) -> DerivedFees:
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
    )


def _extract_standard_commercial_legacy(
    candidates: list[ClassificationCandidate],
    market_code: str | None,
) -> tuple[str | None, list[str]]:
    if not candidates:
        return None, []
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    selected = candidates[0]
    standard_percentage, pct_evidence = _extract_standard_percentage(selected.table, market_code)
    evidence = []
    if standard_percentage:
        evidence.extend(pct_evidence)
        evidence.append(f"standard_commercial table {selected.table.document_id or selected.table.caption}")
    return standard_percentage, evidence


def _derive_from_candidates(
    candidates: list[ClassificationCandidate],
    market_code: str | None,
    other_categories: list[str],
) -> tuple[DerivedFees, list[str], list[str]]:
    """Extract fees from a list of already-classified candidates."""
    evidence: list[str] = []
    warnings: list[str] = []

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

    standard_percentage, std_evidence = _extract_standard_commercial_legacy(
        by_category[FeeCategory.STANDARD_COMMERCIAL], market_code
    )
    evidence.extend(std_evidence)

    fixed_fees, fixed_evidence, fixed_warnings = _aggregate_fixed_fees(by_category[FeeCategory.FIXED_FEE])
    evidence.extend(fixed_evidence)
    warnings.extend(fixed_warnings)

    surcharges, intl_evidence, intl_warnings = _aggregate_international_surcharges(
        by_category[FeeCategory.INTERNATIONAL_SURCHARGE], market_code
    )
    evidence.extend(intl_evidence)
    warnings.extend(intl_warnings)

    conversion, conv_evidence, conv_warnings = _aggregate_conversion(by_category[FeeCategory.CURRENCY_CONVERSION])
    evidence.extend(conv_evidence)
    warnings.extend(conv_warnings)

    warnings = _append_exposure_warnings(
        warnings,
        by_category,
        standard_percentage,
        fixed_fees,
        surcharges,
        conversion,
    )
    warnings = sorted(set(warnings))

    intl_exposed = bool(by_category[FeeCategory.INTERNATIONAL_SURCHARGE])
    conv_exposed = bool(by_category[FeeCategory.CURRENCY_CONVERSION])
    status = _derive_status(
        standard_percentage,
        fixed_fees,
        surcharges,
        conversion,
        by_category,
        warnings,
    )

    return (
        _build_derived_fees(
            status,
            standard_percentage,
            fixed_fees,
            surcharges,
            conversion,
            intl_exposed,
            conv_exposed,
        ),
        evidence,
        warnings,
    )


def _extract_fixed_fees_structural(
    candidates: list[ClassificationCandidate],
) -> tuple[list[FixedFees], list[ClassificationObservation], list[str]]:
    all_fixed_fees: list[FixedFees] = []
    fixed_by_currency: dict[str, str] = {}
    observations: list[ClassificationObservation] = []
    evidence: list[str] = []
    for candidate in candidates:
        profile = candidate.profile or build_table_profile(candidate.table, candidate.contexts)
        decision = extract_fixed_fees(candidate.table, profile)
        observations.extend(decision.observations)
        for sig in decision.evidence:
            evidence.append(sig.detail or sig.code)
        for fee in decision.value or []:
            existing = fixed_by_currency.get(fee.currency)
            if existing == fee.amount:
                continue
            if existing is not None:
                observations.append(
                    ClassificationObservation(
                        kind=ObservationKind.EXTRACTION_CONFLICT,
                        category=FeeCategory.FIXED_FEE,
                        table_id=candidate.table.document_id or candidate.table.table_id,
                        message=f"conflicting fixed fee for {fee.currency}: {existing} vs {fee.amount}",
                    )
                )
                continue
            fixed_by_currency[fee.currency] = fee.amount
            all_fixed_fees.append(fee)
    return all_fixed_fees, observations, evidence


def _extract_standard_commercial_structural(
    candidates: list[ClassificationCandidate],
    market_code: str | None,
    fixed_fees: list[FixedFees],
) -> tuple[str | None, list[ClassificationObservation], list[str]]:
    if not candidates:
        return None, [], []
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    selected = candidates[0]
    profile = selected.profile or build_table_profile(selected.table, selected.contexts)
    decision = extract_standard_percentage(selected.table, profile, market_code, fixed_fees=fixed_fees)
    observations = list(decision.observations)
    evidence = [sig.detail or sig.code for sig in decision.evidence]
    standard_percentage = decision.value
    if standard_percentage:
        evidence.append(f"standard_commercial table {selected.table.document_id or selected.table.caption}")
    return standard_percentage, observations, evidence


def _extract_surcharges_structural(
    candidates: list[ClassificationCandidate],
    market_code: str | None,
) -> tuple[list[InternationalSurcharge], list[ClassificationObservation], list[str]]:
    surcharges: list[InternationalSurcharge] = []
    surcharge_by_region: dict[str, str | None] = {}
    observations: list[ClassificationObservation] = []
    evidence: list[str] = []
    for candidate in candidates:
        profile = candidate.profile or build_table_profile(candidate.table, candidate.contexts)
        decision = extract_international_surcharges(candidate.table, profile, market_code)
        observations.extend(decision.observations)
        for sig in decision.evidence:
            evidence.append(sig.detail or sig.code)
        for surcharge in decision.value or []:
            existing = surcharge_by_region.get(surcharge.region)
            if existing == surcharge.percentage_points:
                continue
            if existing is not None:
                observations.append(
                    ClassificationObservation(
                        kind=ObservationKind.EXTRACTION_CONFLICT,
                        category=FeeCategory.INTERNATIONAL_SURCHARGE,
                        table_id=candidate.table.document_id or candidate.table.table_id,
                        message=f"conflicting surcharge for {surcharge.region}: {existing} vs {surcharge.percentage_points}",
                    )
                )
                continue
            surcharge_by_region[surcharge.region] = surcharge.percentage_points
            surcharges.append(surcharge)
    return surcharges, observations, evidence


def _extract_conversion_structural(
    candidates: list[ClassificationCandidate],
) -> tuple[CurrencyConversion | None, list[ClassificationObservation], list[str]]:
    conversion: CurrencyConversion | None = None
    conversion_values: Counter[str] = Counter()
    last_conversion_candidate: ClassificationCandidate | None = None
    observations: list[ClassificationObservation] = []
    evidence: list[str] = []
    approved_codes = {
        scoring.EvidenceCode.KNOWN_DOCUMENT_ID.value,
        scoring.EvidenceCode.KNOWN_FINGERPRINT.value,
        scoring.EvidenceCode.METADATA_KEY_MATCH.value,
        scoring.EvidenceCode.INTERNAL_NAME_MATCH.value,
    }
    for candidate in candidates:
        last_conversion_candidate = candidate
        profile = candidate.profile or build_table_profile(candidate.table, candidate.contexts)
        has_approved = bool(candidate.approval_codes & approved_codes)
        decision = extract_conversion_spread(candidate.table, profile, has_approved_evidence=has_approved)
        observations.extend(decision.observations)
        for sig in decision.evidence:
            evidence.append(sig.detail or sig.code)
        if decision.value:
            conversion_values[decision.value] += 1
    if conversion_values:
        unique_spreads = set(conversion_values.keys())
        if len(unique_spreads) > 1:
            if last_conversion_candidate is None:
                raise RuntimeError("invariant: no conversion candidate for spread conflict")
            observations.append(
                ClassificationObservation(
                    kind=ObservationKind.EXTRACTION_CONFLICT,
                    category=FeeCategory.CURRENCY_CONVERSION,
                    table_id=last_conversion_candidate.table.document_id or last_conversion_candidate.table.table_id,
                    message=f"conflicting conversion spreads across tables: {sorted(unique_spreads)}",
                )
            )
        else:
            conversion = CurrencyConversion(spread_percentage=next(iter(unique_spreads)))
    return conversion, observations, evidence


def _derive_structural_from_candidates(
    candidates: list[ClassificationCandidate],
    market_code: str | None = None,
    other_categories: list[str] | None = None,
) -> tuple[DerivedFees, tuple[ClassificationObservation, ...], list[str], list[str]]:
    """Build DerivedFees using the new schema-driven extraction helpers.

    Returns the derived fees, extraction-level observations, and top-level
    evidence/warning strings for diagnostics.
    """
    observations: list[ClassificationObservation] = []
    evidence: list[str] = []
    warnings: list[str] = []

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

    all_fixed_fees, fixed_obs, fixed_evidence = _extract_fixed_fees_structural(by_category[FeeCategory.FIXED_FEE])
    observations.extend(fixed_obs)
    evidence.extend(fixed_evidence)

    standard_percentage, std_obs, std_evidence = _extract_standard_commercial_structural(
        by_category[FeeCategory.STANDARD_COMMERCIAL], market_code, all_fixed_fees
    )
    observations.extend(std_obs)
    evidence.extend(std_evidence)

    surcharges, surch_obs, surch_evidence = _extract_surcharges_structural(
        by_category[FeeCategory.INTERNATIONAL_SURCHARGE], market_code
    )
    observations.extend(surch_obs)
    evidence.extend(surch_evidence)

    conversion, conv_obs, conv_evidence = _extract_conversion_structural(by_category[FeeCategory.CURRENCY_CONVERSION])
    observations.extend(conv_obs)
    evidence.extend(conv_evidence)

    warnings = _append_exposure_warnings(
        warnings,
        by_category,
        standard_percentage,
        all_fixed_fees,
        surcharges,
        conversion,
    )
    warnings = sorted(set(warnings))

    intl_exposed = bool(by_category[FeeCategory.INTERNATIONAL_SURCHARGE])
    conv_exposed = bool(by_category[FeeCategory.CURRENCY_CONVERSION])
    status = _derive_status(
        standard_percentage,
        all_fixed_fees,
        surcharges,
        conversion,
        by_category,
        warnings,
    )

    derived = _build_derived_fees(
        status,
        standard_percentage,
        all_fixed_fees,
        surcharges,
        conversion,
        intl_exposed,
        conv_exposed,
    )
    return derived, tuple(observations), evidence, warnings


def _table_decision_from_structural(
    table: Table,
    decision: scoring.ClassificationDecision,
    contexts: tuple[TableContext, ...],
    profile: TableProfile | None = None,
) -> TableDecision:
    """Build a per-table decision record from the structural scorer output."""
    profile = profile or build_table_profile(table, contexts)
    fingerprint = str(FingerprintBuilder.build(profile, table))
    selected_score = decision.selected_score
    # Fall back to the top-ranked score for diagnostics when no category was selected.
    diagnostic_score = (
        selected_score if selected_score else (decision.ranked_scores[0] if decision.ranked_scores else None)
    )
    blockers = diagnostic_score.blockers if diagnostic_score else ()
    evidence_codes = tuple(sorted({s.code.value for s in (diagnostic_score.signals if diagnostic_score else ())}))
    evidence_sources = tuple(sorted({s.source.value for s in (diagnostic_score.signals if diagnostic_score else ())}))
    return TableDecision(
        table_id=table.table_id,
        document_id=table.document_id,
        component_id=table.component_id,
        fingerprint=fingerprint,
        selected_category=decision.selected_category,
        selected_score=selected_score.score if selected_score else None,
        status=decision.status,
        ambiguity_reason=decision.ambiguity_reason,
        winner_margin=decision.winner_margin,
        ranked_scores=decision.ranked_scores,
        blockers=blockers,
        evidence_codes=evidence_codes,
        evidence_sources=evidence_sources,
    )


def _table_decision_from_legacy(candidate: ClassificationCandidate) -> TableDecision:
    """Build a per-table decision record from a legacy candidate."""
    table = candidate.table
    profile = candidate.profile or build_table_profile(table)
    fingerprint = str(FingerprintBuilder.build(profile, table))
    score = int(candidate.confidence * scoring.MAX_CATEGORY_SCORE)
    return TableDecision(
        table_id=table.table_id,
        document_id=table.document_id,
        component_id=table.component_id,
        fingerprint=fingerprint,
        selected_category=candidate.category,
        selected_score=score,
        status="selected",
        ambiguity_reason=None,
        winner_margin=None,
        ranked_scores=(),
        blockers=(),
        evidence_codes=tuple(sorted(set(candidate.evidence))),
        evidence_sources=(),
    )


@dataclass(frozen=True)
class TableDecision:
    """Per-table classification decision with stable identity and evidence."""

    table_id: str | None
    document_id: str | None
    component_id: str | None
    fingerprint: str | None
    selected_category: FeeCategory | None
    selected_score: int | None
    status: str
    ambiguity_reason: str | None
    winner_margin: int | None
    ranked_scores: tuple[scoring.ScoreResult, ...]
    blockers: tuple[scoring.BlockerCode, ...]
    evidence_codes: tuple[str, ...]
    evidence_sources: tuple[str, ...]


@dataclass(frozen=True)
class ClassificationRun:
    derived: DerivedFees
    table_decisions: tuple[TableDecision, ...]
    observations: tuple[ClassificationObservation, ...]
    classifier_version: str
    unclassified_sections: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()


CLASSIFIER_VERSION = "structural-1"


def classify_legacy(
    tables: list[Table],
    market_code: str | None = None,
    locale: str | None = None,
) -> ClassificationRun:
    """Run the legacy classifier and wrap the result in a common internal format."""
    candidates, other_categories = _classify_all_tables(tables)
    derived, evidence, warnings = _derive_from_candidates(candidates, market_code, other_categories)
    table_decisions = tuple(_table_decision_from_legacy(candidate) for candidate in candidates)
    return ClassificationRun(
        derived=derived,
        table_decisions=table_decisions,
        observations=(),
        classifier_version="legacy",
        unclassified_sections=tuple(sorted(set(other_categories))),
        evidence=tuple(sorted(set(evidence + warnings))),
    )


def _structural_candidate_from_decision(
    decision: scoring.ClassificationDecision,
    table: Table,
    contexts: tuple[TableContext, ...],
    profile: TableProfile,
) -> tuple[ClassificationCandidate | None, list[ClassificationObservation]]:
    if not (
        decision.status == "selected" and decision.selected_category is not None and decision.selected_score is not None
    ):
        return None, []

    selected_score = decision.selected_score
    score = selected_score.score
    candidate = ClassificationCandidate(
        table=table,
        category=decision.selected_category,
        confidence=score / scoring.MAX_CATEGORY_SCORE,
        evidence=[s.detail or s.code for s in selected_score.signals],
        profile=profile,
        contexts=contexts,
        approval_codes={s.code.value for s in selected_score.signals},
    )
    observations: list[ClassificationObservation] = []

    if decision.winner_margin is not None and decision.winner_margin <= scoring.MINIMUM_MARGIN:
        observations.append(
            ClassificationObservation(
                kind=ObservationKind.LOW_MARGIN,
                category=decision.selected_category,
                table_id=table.document_id or table.table_id,
                message=f"selected with low margin: {decision.winner_margin}",
            )
        )

    signal_sources = {s.source for s in selected_score.signals}
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
    } & {s.code for s in selected_score.signals}:
        observations.append(
            ClassificationObservation(
                kind=ObservationKind.UNKNOWN_FINGERPRINT,
                category=decision.selected_category,
                table_id=table.document_id or table.table_id,
                message="conversion selected without approved registry evidence",
            )
        )

    return candidate, observations


def classify_structural(
    tables: list[Table] | list[NormalizedTableRecord],
    market_code: str | None = None,
    locale: str | None = None,
    registry: FingerprintRegistry | None = None,
) -> ClassificationRun:
    """Run the structural scoring classifier and produce derived fees.

    This function uses the scoring engine in ``scoring.py`` to select a category
    for each table, then extracts values using the existing extraction helpers.
    It accepts either plain ``Table`` objects or ``NormalizedTableRecord``
    objects that preserve multiple reference contexts.
    """
    candidates: list[ClassificationCandidate] = []
    table_decisions: list[TableDecision] = []
    observations: list[ClassificationObservation] = []

    for item in tables:
        if isinstance(item, NormalizedTableRecord):
            table = item.table
            contexts = item.contexts
        else:
            table = item
            contexts = (TableContext.from_table(table),)

        if not table.rows and not table.headers:
            continue

        profile = build_table_profile(table, contexts)
        scores = scoring.score_all_categories(table, market_code, locale, registry, profile=profile)
        decision = scoring.select_category(scores)
        table_decisions.append(_table_decision_from_structural(table, decision, contexts, profile))

        candidate, candidate_observations = _structural_candidate_from_decision(decision, table, contexts, profile)
        if candidate is not None:
            candidates.append(candidate)
            observations.extend(candidate_observations)

    other_categories: list[str] = []
    derived, extract_observations, evidence, warnings = _derive_structural_from_candidates(
        candidates, market_code, other_categories
    )
    return ClassificationRun(
        derived=derived,
        table_decisions=tuple(table_decisions),
        observations=tuple(observations + list(extract_observations)),
        classifier_version=CLASSIFIER_VERSION,
        unclassified_sections=tuple(sorted(set(other_categories))),
        evidence=tuple(sorted(set(evidence + warnings))),
    )
