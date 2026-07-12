"""Schema-driven extraction helpers for fee values.

PR 1C hardens extraction so every value is returned as a typed decision with
observations, rather than silently choosing the first plausible token.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .models import FixedFees, InternationalSurcharge, Row, Table
from .normalize import clean_text
from .profiles import TableProfile
from .scoring import (
    EvidenceCode,
    EvidenceSignal,
    EvidenceSource,
    FeeCategory,
    market_code_matches,
    region_from_text,
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
class ExtractionDecision[T]:
    """Typed extraction result with evidence and observations."""

    value: T | None
    selected_rows: tuple[int, ...]
    evidence: tuple[EvidenceSignal, ...]
    observations: tuple[ClassificationObservation, ...]


@dataclass(frozen=True)
class ColumnRoleAssignment:
    """Deterministic column roles for a table with row labels and values."""

    label_column: int | None
    percentage_columns: tuple[int, ...]
    money_columns: tuple[int, ...]
    confidence: int


def _norm(text: str | None) -> str:
    return clean_text(text or "").casefold()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _first_percentage_in_row(row: Row) -> str | None:
    for cell in row.cells:
        for token in cell.tokens:
            if token.kind == "percentage" and token.value:
                return token.value
    return None


def _first_money_in_row(row: Row, column: int | None = None) -> tuple[str, str] | None:
    """Return (amount, currency) for the first money token in a row or column."""
    for idx, cell in enumerate(row.cells):
        if column is not None and idx != column:
            continue
        for token in cell.tokens:
            if token.kind == "money" and token.amount and token.currency:
                return (token.amount, token.currency)
    return None


def _percentages_in_row(row: Row) -> list[tuple[int, str]]:
    """Return (column_index, value) for all percentage tokens in a row."""
    found: list[tuple[int, str]] = []
    for idx, cell in enumerate(row.cells):
        for token in cell.tokens:
            if token.kind == "percentage" and token.value:
                found.append((idx, token.value))
    return found


def _moneys_in_row(row: Row) -> list[tuple[int, str, str]]:
    """Return (column_index, amount, currency) for all money tokens in a row."""
    found: list[tuple[int, str, str]] = []
    for idx, cell in enumerate(row.cells):
        for token in cell.tokens:
            if token.kind == "money" and token.amount and token.currency:
                found.append((idx, token.amount, token.currency))
    return found


def _column_roles(profile: TableProfile) -> ColumnRoleAssignment:
    """Infer label, percentage, and money columns from a profile."""
    percentage_columns = tuple(sorted(profile.percentage_columns))
    money_columns = tuple(sorted(profile.money_columns))

    label_column: int | None = None
    # The label column is the leftmost text-heavy column that is not a value
    # column. If every column is a value column, there is no separate label.
    for col in profile.columns:
        if col.column_index in percentage_columns or col.column_index in money_columns:
            continue
        if col.text_row_count > 0:
            label_column = col.column_index
            break

    # Confidence: full assignment if we have value columns and a label.
    if percentage_columns and label_column is not None:
        confidence = 100
    elif percentage_columns and money_columns:
        confidence = 80
    elif percentage_columns or money_columns:
        confidence = 50
    else:
        confidence = 0

    return ColumnRoleAssignment(
        label_column=label_column,
        percentage_columns=percentage_columns,
        money_columns=money_columns,
        confidence=confidence,
    )


def _add_evidence(
    signals: list[EvidenceSignal],
    code: EvidenceCode,
    source: EvidenceSource,
    weight: int,
    detail: str | None = None,
) -> None:
    signals.append(EvidenceSignal(code=code, source=source, weight=weight, detail=detail))


# ---------------------------------------------------------------------------
# Standard commercial percentage extraction
# ---------------------------------------------------------------------------

# Broad but low-weight labels that signal a standard-commercial row.
_STD_ROW_INCLUDE = (
    "commercial",
    "comercial",
    "comerciales",
    "transaction",
    "transactions",
    "transacción",
    "transacciones",
    "transakcie",
    "transakcií",
    "merchant",
    "händler",
    "merchant fees",
    "online payment",
    "online card",
    "payment",
    "pago",
    "pagos",
    "domestic",
    "inland",
    "nationales",
    "nacionales",
    "commercial payments",
    "commercial payment",
    "online payments",
    "online card payments",
    "online card payment",
    "commercial transaction",
    "commercial transactions",
    "comercial transaction",
    "comercial transactions",
    "transakcie",
    "transakcií",
    "medzinárodné",
    "medzinárodne",
    "medzinárodná",
    "domáce",
    "domace",
    "online",
    "card",
    "card payment",
    "card payments",
    "digital payment",
    "digital payments",
    "send",
    "receive",
    "goods and services",
    "goods & services",
    "goods and service",
    "sale",
    "sales",
    "purchase",
    "purchases",
    "buy",
    "sell",
    "selling",
    "buying",
)

# Labels that should not be allowed to select a standard-commercial percentage.
_STD_ROW_EXCLUDE = (
    "charity",
    "donation",
    "nonprofit",
    "non-profit",
    "micropayment",
    "micro payment",
    "micro-payment",
    "point of sale",
    "pos",
    "retail",
    "store",
    "terminal",
    "website payments pro",
    "virtual terminal",
    "invoice",
    "invoicing",
    "interchange",
    "max cap",
    "maximum fee",
    "minimum fee",
    "instant transfer",
    "cryptocurrency",
    "crypto",
    "additional",
    "additional percentage",
    "additional percentage-based fee",
    "additional service fee",
    "withdrawal",
    "payout",
    "payouts",
    "dispute",
    "chargeback",
    "cross border",
    "cross-border",
    "crossborder",
    "international",
    "international surcharge",
    "currency conversion",
    "conversion",
    "fixed fee",
    "fixed fees",
    "festgebühr",
    "feste gebühr",
)

# Fallback labels for a localized or catch-all standard-commercial row.
_STD_ROW_FALLBACK = (
    "all other",
    "all other markets",
    "all other transactions",
    "rest of world",
    "rest of the world",
    "rest of markets",
    "rest of the markets",
    "other",
    "andere",
    "rest",
    "todos los demás",
    "todos los demas",
    "todas las demás",
    "todas las demas",
    "všetky ostatné",
    "vsetky ostatne",
    "todos los mercados",
    "todas las mercados",
    "všetky trhy",
    "vsetky trhy",
    "restantes",
    "otros mercados",
    "otras mercados",
)


def extract_standard_percentage(
    table: Table,
    profile: TableProfile,
    market_code: str | None = None,
    fixed_fees: list[FixedFees] | None = None,
) -> ExtractionDecision[str]:
    """Extract the standard commercial percentage for a table.

    Selection priority:
    1. a row containing both a percentage and a fixed monetary component;
    2. a row linked to a validated fixed-fee table (same currency present);
    3. a structurally coherent percentage-only row;
    4. lexical row evidence;
    5. otherwise ambiguous.
    """
    signals: list[EvidenceSignal] = []
    observations: list[ClassificationObservation] = []
    selected_rows: list[int] = []
    candidates: list[tuple[int, str, int]] = []  # (row, pct, priority)

    fixed_currencies = {f.currency for f in (fixed_fees or [])}

    for row_idx, row in enumerate(table.rows):
        pct = _first_percentage_in_row(row)
        if not pct:
            continue

        first_cell_text = _norm(row.cells[0].text) if row.cells else ""
        all_text = _norm(" ".join(cell.text for cell in row.cells))

        if _contains_any(first_cell_text, _STD_ROW_EXCLUDE) or _contains_any(all_text, _STD_ROW_EXCLUDE):
            continue

        # Priority 1: mixed percentage and money row.
        has_money = any(_moneys_in_row(row))
        if has_money and row_idx in profile.mixed_percentage_money_rows:
            candidates.append((row_idx, pct, 1))
            continue

        # Priority 2: fixed-fee table linkage (same currency).
        if fixed_currencies:
            row_currencies = {
                token.currency
                for cell in row.cells
                for token in cell.tokens
                if token.kind == "money" and token.currency
            }
            if row_currencies & fixed_currencies:
                candidates.append((row_idx, pct, 2))
                continue

        # Priority 3: structurally coherent percentage-only row.
        row_pct_cols = [c for c, _ in _percentages_in_row(row)]
        if row_pct_cols:
            candidates.append((row_idx, pct, 3))
            continue

        # Priority 4: lexical evidence.
        if _contains_any(first_cell_text, _STD_ROW_INCLUDE) or _contains_any(all_text, _STD_ROW_INCLUDE):
            candidates.append((row_idx, pct, 4))

    # If no keyword candidates, allow a market-specific or catch-all fallback.
    if not candidates:
        for row_idx, row in enumerate(table.rows):
            pct = _first_percentage_in_row(row)
            if not pct:
                continue
            first_cell_text = _norm(row.cells[0].text) if row.cells else ""
            all_text = _norm(" ".join(cell.text for cell in row.cells))
            if _contains_any(first_cell_text, _STD_ROW_EXCLUDE) or _contains_any(all_text, _STD_ROW_EXCLUDE):
                continue
            if market_code and (market_code_matches(first_cell_text, market_code) or market_code_matches(all_text, market_code)):
                candidates.append((row_idx, pct, 5))
                continue
            if _contains_any(first_cell_text, _STD_ROW_FALLBACK) or _contains_any(all_text, _STD_ROW_FALLBACK):
                candidates.append((row_idx, pct, 5))

    if not candidates:
        observations.append(
            ClassificationObservation(
                kind=ObservationKind.EXTRACTION_CONFLICT,
                category=FeeCategory.STANDARD_COMMERCIAL,
                table_id=table.table_id or table.document_id,
                message="no standard-commercial percentage row found",
            )
        )
        return ExtractionDecision(value=None, selected_rows=(), evidence=tuple(signals), observations=tuple(observations))

    candidates.sort(key=lambda x: (x[2], -x[0]))
    selected_row, selected_pct, priority = candidates[0]
    selected_rows.append(selected_row)
    _add_evidence(
        signals,
        EvidenceCode.HAS_PERCENTAGE_COLUMN,
        EvidenceSource.STRUCTURAL,
        10,
        detail=f"priority {priority} row {selected_row}: {selected_pct}%",
    )
    return ExtractionDecision(
        value=selected_pct,
        selected_rows=tuple(selected_rows),
        evidence=tuple(signals),
        observations=tuple(observations),
    )


# ---------------------------------------------------------------------------
# Fixed-fee extraction
# ---------------------------------------------------------------------------


def extract_fixed_fees(table: Table, profile: TableProfile) -> ExtractionDecision[list[FixedFees]]:
    """Extract fixed-fee rows as a list of (currency, amount) values."""
    signals: list[EvidenceSignal] = []
    observations: list[ClassificationObservation] = []
    fees: list[FixedFees] = []
    by_currency: dict[str, str] = {}
    selected_rows: list[int] = []

    roles = _column_roles(profile)
    currency_column: int | None = None
    if len(roles.money_columns) == 1:
        currency_column = roles.money_columns[0]

    for row_idx, row in enumerate(table.rows):
        if profile.rows[row_idx].is_probable_header or profile.rows[row_idx].is_probable_note:
            continue

        moneys = _moneys_in_row(row)
        if not moneys:
            continue

        if len(moneys) != 1:
            observations.append(
                ClassificationObservation(
                    kind=ObservationKind.EXTRACTION_CONFLICT,
                    category=FeeCategory.FIXED_FEE,
                    table_id=table.table_id or table.document_id,
                    message=f"row {row_idx} has {len(moneys)} money tokens",
                )
            )
            continue

        col, amount, currency = moneys[0]

        # If there is a separate currency-label column, use it when the money
        # token did not carry a currency (unlikely for our tokenizer, but
        # defensively handled).
        if not currency and currency_column is not None and currency_column < len(row.cells):
            label = row.cells[currency_column].text.strip()
            if label and len(label) == 3 and label.upper().isalpha():
                currency = label.upper()

        if not currency:
            observations.append(
                ClassificationObservation(
                    kind=ObservationKind.EXTRACTION_CONFLICT,
                    category=FeeCategory.FIXED_FEE,
                    table_id=table.table_id or table.document_id,
                    message=f"row {row_idx} has no determinable currency",
                )
            )
            continue

        existing = by_currency.get(currency)
        if existing is not None and existing != amount:
            observations.append(
                ClassificationObservation(
                    kind=ObservationKind.EXTRACTION_CONFLICT,
                    category=FeeCategory.FIXED_FEE,
                    table_id=table.table_id or table.document_id,
                    message=f"conflicting fixed fee for {currency}: {existing} vs {amount}",
                )
            )
            continue

        by_currency[currency] = amount
        selected_rows.append(row_idx)
        fees.append(FixedFees(currency=currency, amount=amount))

    if not fees:
        observations.append(
            ClassificationObservation(
                kind=ObservationKind.EXTRACTION_CONFLICT,
                category=FeeCategory.FIXED_FEE,
                table_id=table.table_id or table.document_id,
                message="no fixed-fee rows found",
            )
        )

    return ExtractionDecision(
        value=fees,
        selected_rows=tuple(selected_rows),
        evidence=tuple(signals),
        observations=tuple(observations),
    )


# ---------------------------------------------------------------------------
# International surcharge extraction
# ---------------------------------------------------------------------------


def _normalize_region(text: str) -> str | None:
    """Map a region cell to one of the canonical surcharge regions."""
    t = _norm(text)
    if not t:
        return None

    if region_from_text(t) == "EEA":
        return "EEA"
    if region_from_text(t) == "GB":
        return "GB"
    if region_from_text(t) == "US_CA":
        return "US_CA"
    if region_from_text(t) == "OTHER":
        return "OTHER"

    return None


def extract_international_surcharges(
    table: Table,
    profile: TableProfile,
    market_code: str | None = None,
) -> ExtractionDecision[list[InternationalSurcharge]]:
    """Extract international payer-region surcharges."""
    signals: list[EvidenceSignal] = []
    observations: list[ClassificationObservation] = []
    surcharges: list[InternationalSurcharge] = []
    selected_rows: list[int] = []
    seen_regions: set[str] = set()

    roles = _column_roles(profile)
    label_column = roles.label_column
    percentage_columns = roles.percentage_columns

    if not percentage_columns:
        observations.append(
            ClassificationObservation(
                kind=ObservationKind.EXTRACTION_CONFLICT,
                category=FeeCategory.INTERNATIONAL_SURCHARGE,
                table_id=table.table_id or table.document_id,
                message="no percentage column assigned for international surcharge table",
            )
        )
        return ExtractionDecision(
            value=surcharges,
            selected_rows=tuple(selected_rows),
            evidence=tuple(signals),
            observations=tuple(observations),
        )

    pct_column = percentage_columns[0]

    for row_idx, row in enumerate(table.rows):
        if profile.rows[row_idx].is_probable_header or profile.rows[row_idx].is_probable_note:
            continue

        pcts = _percentages_in_row(row)
        if not pcts:
            continue

        # If there are multiple percentages, report ambiguity and continue.
        assigned_pcts = [pct for col, pct in pcts if col == pct_column]
        if not assigned_pcts:
            assigned_pcts = [pct for _, pct in pcts]

        if len(assigned_pcts) > 1:
            observations.append(
                ClassificationObservation(
                    kind=ObservationKind.EXTRACTION_CONFLICT,
                    category=FeeCategory.INTERNATIONAL_SURCHARGE,
                    table_id=table.table_id or table.document_id,
                    message=f"row {row_idx} has multiple candidate percentages",
                )
            )
            continue

        pct = assigned_pcts[0]
        selected_rows.append(row_idx)

        label = ""
        if label_column is not None and label_column < len(row.cells):
            label = row.cells[label_column].text
        else:
            # Fall back to the first non-value cell.
            for idx, cell in enumerate(row.cells):
                if idx in percentage_columns:
                    continue
                if cell.text.strip():
                    label = cell.text
                    break

        region = _normalize_region(label)
        if market_code and region is None and market_code_matches(label, market_code):
            region = market_code.upper()

        if region is None:
            region = _norm(label) or "OTHER"

        if region in seen_regions:
            observations.append(
                ClassificationObservation(
                    kind=ObservationKind.EXTRACTION_CONFLICT,
                    category=FeeCategory.INTERNATIONAL_SURCHARGE,
                    table_id=table.table_id or table.document_id,
                    message=f"duplicate region {region} in row {row_idx}",
                )
            )
            continue

        seen_regions.add(region)
        surcharges.append(InternationalSurcharge(region=region, percentage_points=pct))
        _add_evidence(
            signals,
            EvidenceCode.HAS_PERCENTAGE_COLUMN,
            EvidenceSource.STRUCTURAL,
            5,
            detail=f"{region}: {pct}",
        )

    return ExtractionDecision(
        value=surcharges,
        selected_rows=tuple(selected_rows),
        evidence=tuple(signals),
        observations=tuple(observations),
    )


# ---------------------------------------------------------------------------
# Conversion-spread extraction
# ---------------------------------------------------------------------------


def extract_conversion_spread(
    table: Table,
    profile: TableProfile,
    has_approved_evidence: bool = False,
) -> ExtractionDecision[str]:
    """Extract currency conversion spread percentage.

    For the first structural release, a conversion value is only returned when
    approved evidence is present or the table is unambiguously conversion.
    """
    signals: list[EvidenceSignal] = []
    observations: list[ClassificationObservation] = []
    selected_rows: list[int] = []

    text = _norm(" ".join(
        [table.caption or ""]
        + list(table.section_path or [])
        + list(table.parent_path or [])
        + [header.text for header in table.headers]
    ))

    conversion_keywords = (
        "currency conversion",
        "conversion",
        "conversión",
        "währungsumrechnung",
        "umrechnung",
        "wechselkurs",
        "spread",
        "base exchange rate",
        "tipo de cambio",
        "tipos de cambio",
        "cambio",
        "tasas de cambio",
        "foreign exchange",
        "exchange rate",
        "prepočet",
        "prepocet",
        "zmena",
        "zmena meny",
        "výmenný",
        "vymenny",
        "výmenný kurz",
        "vymenny kurz",
    )
    is_conversion_table = any(kw in text for kw in conversion_keywords)

    if not is_conversion_table and not has_approved_evidence:
        observations.append(
            ClassificationObservation(
                kind=ObservationKind.UNKNOWN_FINGERPRINT,
                category=FeeCategory.CURRENCY_CONVERSION,
                table_id=table.table_id or table.document_id,
                message="conversion table lacks approved evidence or unambiguous label",
            )
        )
        return ExtractionDecision(value=None, selected_rows=(), evidence=tuple(signals), observations=tuple(observations))

    roles = _column_roles(profile)
    pcts: list[tuple[int, str]] = []
    for row_idx, row in enumerate(table.rows):
        if profile.rows[row_idx].is_probable_header or profile.rows[row_idx].is_probable_note:
            continue
        row_pcts = _percentages_in_row(row)
        if len(row_pcts) > 1:
            observations.append(
                ClassificationObservation(
                    kind=ObservationKind.EXTRACTION_CONFLICT,
                    category=FeeCategory.CURRENCY_CONVERSION,
                    table_id=table.table_id or table.document_id,
                    message=f"row {row_idx} has multiple conversion percentages",
                )
            )
        for col, pct in row_pcts:
            if roles.percentage_columns and col not in roles.percentage_columns:
                continue
            pcts.append((row_idx, pct))
            selected_rows.append(row_idx)

    if not pcts:
        observations.append(
            ClassificationObservation(
                kind=ObservationKind.EXTRACTION_CONFLICT,
                category=FeeCategory.CURRENCY_CONVERSION,
                table_id=table.table_id or table.document_id,
                message="no conversion percentage found",
            )
        )
        return ExtractionDecision(value=None, selected_rows=tuple(selected_rows), evidence=tuple(signals), observations=tuple(observations))

    # Prefer the first structurally assigned percentage.
    selected_row, selected_pct = pcts[0]
    _add_evidence(
        signals,
        EvidenceCode.HAS_PERCENTAGE_COLUMN,
        EvidenceSource.STRUCTURAL,
        10,
        detail=f"row {selected_row}: {selected_pct}%",
    )
    return ExtractionDecision(
        value=selected_pct,
        selected_rows=tuple(selected_rows),
        evidence=tuple(signals),
        observations=tuple(observations),
    )
