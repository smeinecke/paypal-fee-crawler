"""Derive core merchant fees from normalized tables with fail-closed confidence."""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum

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

logger = logging.getLogger(__name__)


class FeeCategory(StrEnum):
    STANDARD_COMMERCIAL = "standard_commercial"
    FIXED_FEE = "fixed_fee"
    INTERNATIONAL_SURCHARGE = "international_surcharge"
    CURRENCY_CONVERSION = "currency_conversion"
    OTHER = "other"


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
_STANDARD_DOC_IDS = {"FEETB16"}
_FIXED_DOC_IDS = {"FEETB18", "FEETB306", "FEETB261"}
_INTERNATIONAL_DOC_IDS = {"FEETB91"}
_CONVERSION_DOC_IDS = {"FEETB539", "FEETB128"}


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
    "ausland",
    "cross border",
    "cross-border",
    "conversion",
    "umrechnung",
    "wechselkurs",
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
)

_POS_STANDARD = (
    "standard",
    "commercial",
    "domestic",
    "inland",
    "transaktion",
    "transaction",
    "merchant",
    "händler",
    "händlergebühren",
    "merchant fees",
    "online payment",
    "online card",
    "receiving domestic",
    "zahlungsempfang",
    "payPal-gebühren",
)

_POS_STANDARD_HEADER = (
    "payment type",
    "art der transaktion",
    "rate",
    "gebühr",
    "fee",
    "transaktion",
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
    "hyperwallet",
)

_POS_FIXED = (
    "fixed fee",
    "festgebühr",
    "feste gebühr",
    "fixe gebühr",
    "fixed charge",
    "per transaction",
    "pro transaktion",
    "based on currency",
    "auf basis der empfangenen währung",
    "währung",
    "currency",
    "commercial",
    "geschäftlich",
    "business transaction",
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
)

_POS_INTERNATIONAL = (
    "international",
    "cross border",
    "cross-border",
    "ausland",
    "auslandszahlung",
    "grenzüberschreitend",
    "zusatzgebühr",
    "additional percentage",
    "payer region",
    "markt/region",
    "market/region",
    "markt/das gebiet",
    "region",
)


def _is_international_surcharge(table: Table) -> tuple[bool, float, list[str]]:
    text = _table_text(table)
    if _contains_any(text, _NEG_INTERNATIONAL):
        return False, 0.0, []
    evidence: list[str] = []
    confidence = 0.0

    doc_id = _table_doc_id(table)
    if doc_id in _INTERNATIONAL_DOC_IDS:
        confidence += 0.6
        evidence.append(f"document_id {doc_id} is a known international-surcharge table")

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
)

_POS_CONVERSION = (
    "currency conversion",
    "converting balance",
    "währungsumrechnung",
    "umrechnung",
    "wechselkurs",
    "conversion",
    "spread",
    "base exchange rate",
    "basiswechselkurs",
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
    "donation": ("donation", "spende", "charity donation"),
    "nonprofit": ("nonprofit", "non-profit", "gemeinnützig", "gemeinnutzig"),
    "chargeback": ("chargeback", "rückbuchung", "rückabwicklung", "rücklastschrift"),
    "dispute": ("dispute", "streitfall", "konfliktlösung"),
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
    "payPal checkout",
    "checkout",
    "commercial",
    "transaction",
    "payments",
    "payment",
    "zahlung",
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
)

_STD_ROW_EXCLUDE = (
    "qr",
    "qr-code",
    "qr code",
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


def _cell_text_starts_with(cell: Cell) -> str:
    return _norm(cell.text.split()[0]) if cell.text.split() else ""


def _extract_standard_percentage(table: Table) -> tuple[str | None, list[str]]:
    """Return the most confident standard-commercial percentage in a table."""
    evidence: list[str] = []
    matched_percentages: list[str] = []
    for row in table.rows:
        pct = _first_percentage_in_row(row)
        if not pct:
            continue
        first_cell_text = _norm(row.cells[0].text) if row.cells else ""
        all_text = _row_text(row)
        if _contains_any(first_cell_text, _STD_ROW_EXCLUDE) or _contains_any(all_text, _STD_ROW_EXCLUDE):
            continue
        if _contains_any(first_cell_text, _STD_ROW_INCLUDE) or _contains_any(all_text, _STD_ROW_INCLUDE):
            matched_percentages.append(pct)

    if not matched_percentages:
        evidence.append("no standard-commercial row matched")
        return None, evidence

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

    if "european economic" in t or "european economic area" in t or "ewr" in t or "eea" in t or "e.u" in t or t == "eu":
        return "EEA"
    if (
        "united kingdom" in t
        or t == "gb"
        or "uk" in t
        or "großbritannien" in t
        or "great britain" in t
        or "britain" in t
        or "england" in t
    ):
        return "GB"
    if "united states" in t or "usa" in t or "u.s" in t or t == "us" or "canada" in t:
        return "US_CA"
    if "all" in t and "other" in t:
        return "OTHER"
    if "rest" in t and "world" in t:
        return "OTHER"
    if "all commercial" in t or "all payment" in t or "commercial transactions" in t:
        return "OTHER"
    if "other" in t or "andere" in t or "rest" in t:
        return "OTHER"
    return None


def _extract_international_surcharges(table: Table) -> list[InternationalSurcharge]:
    """Extract region->percentage rows from an international-surcharge table."""
    surcharges: list[InternationalSurcharge] = []
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

        region: str | None = None
        # The region is usually the first non-empty cell, but we scan all cells.
        for cell in row.cells:
            region = _normalize_region(cell.text)
            if region:
                break

        if region:
            surcharges.append(InternationalSurcharge(region=region, percentage_points=percentage))
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
) -> tuple[list[InternationalSurcharge], list[str], list[str]]:
    evidence: list[str] = []
    warnings: list[str] = []
    all_surcharges: list[InternationalSurcharge] = []
    for candidate in candidates:
        surcharges = _extract_international_surcharges(candidate.table)
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


def classify_tables(tables: list[Table]) -> DerivedFees:
    """Derive core fees from normalized tables with explicit confidence and evidence."""
    evidence: list[str] = []
    warnings: list[str] = []

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
        else:
            evidence.extend(candidate.evidence)

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

    # Standard commercial.
    standard_percentage: str | None = None
    standard_candidates = by_category[FeeCategory.STANDARD_COMMERCIAL]
    if standard_candidates:
        # Prefer the highest-confidence candidate; on ties, preserve source order.
        standard_candidates.sort(key=lambda c: c.confidence, reverse=True)
        selected = standard_candidates[0]
        standard_percentage, pct_evidence = _extract_standard_percentage(selected.table)
        if standard_percentage:
            evidence.extend(pct_evidence)
            evidence.append(f"standard_commercial table {selected.table.document_id or selected.table.caption}")

    # Fixed fees.
    fixed_fees, fixed_evidence, fixed_warnings = _aggregate_fixed_fees(by_category[FeeCategory.FIXED_FEE])
    evidence.extend(fixed_evidence)
    warnings.extend(fixed_warnings)

    # International surcharges.
    surcharges, intl_evidence, intl_warnings = _aggregate_international_surcharges(
        by_category[FeeCategory.INTERNATIONAL_SURCHARGE]
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
        unclassified_sections=sorted(set(other_categories)),
        classification_evidence=evidence + warnings,
    )
