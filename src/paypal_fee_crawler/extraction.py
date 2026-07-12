"""Schema-driven extraction helpers for fee values.

PR 1C hardens extraction so every value is returned as a typed decision with
observations, rather than silently choosing the first plausible token.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .models import FixedFees, InternationalSurcharge, Row, Table
from .normalize import clean_text
from .pricing_tokens import CURRENCY_CODES, is_numeric_amount, parse_amount
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
    currency_label_column: int | None
    amount_column: int | None
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


def _is_iso_currency_code(text: str) -> bool:
    """Return True if *text* is a three-letter ISO 4217 currency code."""
    return len(text) == 3 and text.isalpha() and text.upper() in CURRENCY_CODES


def _column_roles(profile: TableProfile, table: Table) -> ColumnRoleAssignment:
    """Infer label, percentage, money, currency-label, and amount columns."""
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

    # Infer the currency-label column from non-value cells that contain valid
    # ISO 4217 codes.  The leftmost column matching is preferred.
    currency_label_column: int | None = None
    for col in profile.columns:
        if col.column_index in percentage_columns or col.column_index in money_columns:
            continue
        valid = True
        for row in table.rows:
            if col.column_index >= len(row.cells):
                continue
            cell = row.cells[col.column_index].text.strip()
            if not cell:
                continue
            if not _is_iso_currency_code(cell):
                valid = False
                break
        if valid:
            currency_label_column = col.column_index
            break

    # Infer the amount column.  Prefer columns with money tokens, then fall back
    # to columns whose cells are plain numeric amounts.
    amount_column: int | None = None
    if len(money_columns) == 1:
        amount_column = money_columns[0]
    else:
        for col in profile.columns:
            if col.column_index in percentage_columns:
                continue
            valid = True
            for row in table.rows:
                if col.column_index >= len(row.cells):
                    continue
                cell = row.cells[col.column_index].text.strip()
                if not cell:
                    continue
                if not is_numeric_amount(cell):
                    valid = False
                    break
            if valid:
                amount_column = col.column_index
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
        currency_label_column=currency_label_column,
        amount_column=amount_column,
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
    "personal",
    "friends",
    "friends and family",
    "friends & family",
    "friend",
    "family",
    "personal payments",
    "personal payment",
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


def _score_standard_row(
    row: Row,
    row_idx: int,
    profile: TableProfile,
    fixed_currencies: set[str],
    market_code: str | None,
) -> int | None:
    """Return a non-negative score for a standard-commercial row, or None if excluded."""
    pct = _first_percentage_in_row(row)
    if not pct:
        return None

    first_cell_text = _norm(row.cells[0].text) if row.cells else ""
    all_text = _norm(" ".join(cell.text for cell in row.cells))

    if _contains_any(first_cell_text, _STD_ROW_EXCLUDE) or _contains_any(all_text, _STD_ROW_EXCLUDE):
        return None

    score = 0

    # Mixed percentage-plus-fixed expressions are the strongest signal.
    has_money = any(_moneys_in_row(row))
    if has_money and row_idx in profile.mixed_percentage_money_rows:
        score += 50

    # Rows linked to a validated fixed-fee table (same currency present).
    if fixed_currencies and has_money:
        row_currencies = {
            token.currency for cell in row.cells for token in cell.tokens if token.kind == "money" and token.currency
        }
        if row_currencies & fixed_currencies:
            score += 30

    # Rows with a percentage in an inferred percentage column.
    row_pct_cols = {c for c, _ in _percentages_in_row(row)}
    if row_pct_cols & profile.percentage_columns:
        score += 20

    # Positive lexical evidence.
    if _contains_any(first_cell_text, _STD_ROW_INCLUDE) or _contains_any(all_text, _STD_ROW_INCLUDE):
        score += 10

    # Fallback market or catch-all labels.
    if market_code and (
        market_code_matches(first_cell_text, market_code) or market_code_matches(all_text, market_code)
    ):
        score += 5
    if _contains_any(first_cell_text, _STD_ROW_FALLBACK) or _contains_any(all_text, _STD_ROW_FALLBACK):
        score += 5

    return score if score > 0 else None


def extract_standard_percentage(
    table: Table,
    profile: TableProfile,
    market_code: str | None = None,
    fixed_fees: list[FixedFees] | None = None,
) -> ExtractionDecision[str]:
    """Extract the standard commercial percentage for a table.

    Candidate rows are scored structurally and lexically; the highest-scoring
    value wins.  If two rows tie with the same top score but different values,
    the extraction fails closed and reports a conflict.
    """
    signals: list[EvidenceSignal] = []
    observations: list[ClassificationObservation] = []
    selected_rows: list[int] = []
    candidates: list[tuple[int, str, int]] = []  # (row, pct, score)

    fixed_currencies = {f.currency for f in (fixed_fees or [])}

    for row_idx, row in enumerate(table.rows):
        row_score = _score_standard_row(row, row_idx, profile, fixed_currencies, market_code)
        if row_score is None:
            continue
        pct = _first_percentage_in_row(row)
        if pct:
            candidates.append((row_idx, pct, row_score))

    if not candidates:
        observations.append(
            ClassificationObservation(
                kind=ObservationKind.EXTRACTION_CONFLICT,
                category=FeeCategory.STANDARD_COMMERCIAL,
                table_id=table.table_id or table.document_id,
                message="no standard-commercial percentage row found",
            )
        )
        return ExtractionDecision(
            value=None, selected_rows=(), evidence=tuple(signals), observations=tuple(observations)
        )

    # Sort by score descending, then row index ascending for deterministic order.
    # Row index is used only for deterministic ordering, not as a semantic tie-breaker.
    candidates.sort(key=lambda x: (-x[2], x[0]))
    top_score = candidates[0][2]
    top_value = candidates[0][1]

    # Check for equally supported rows with different values.
    for _row_idx, pct, score in candidates:
        if score == top_score and pct != top_value:
            observations.append(
                ClassificationObservation(
                    kind=ObservationKind.EXTRACTION_CONFLICT,
                    category=FeeCategory.STANDARD_COMMERCIAL,
                    table_id=table.table_id or table.document_id,
                    message=f"equally supported standard rates: {top_value}% and {pct}%",
                )
            )
            return ExtractionDecision(
                value=None, selected_rows=(), evidence=tuple(signals), observations=tuple(observations)
            )

    selected_row, selected_pct, selected_score = candidates[0]
    selected_rows.append(selected_row)
    _add_evidence(
        signals,
        EvidenceCode.HAS_PERCENTAGE_COLUMN,
        EvidenceSource.STRUCTURAL,
        10,
        detail=f"score {selected_score} row {selected_row}: {selected_pct}%",
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
    """Extract fixed-fee rows as a list of (currency, amount) values.

    Supports money tokens (e.g. ``0.39 EUR``), separate currency-label and
    numeric-amount columns (e.g. ``EUR | 0.39``), and ISO 4217 currency-label
    validation.  Duplicate identical (currency, amount) pairs are skipped and
    conflicting values are reported as extraction conflicts.
    """
    signals: list[EvidenceSignal] = []
    observations: list[ClassificationObservation] = []
    fees: list[FixedFees] = []
    by_currency: dict[str, str] = {}
    selected_rows: list[int] = []

    roles = _column_roles(profile, table)
    currency_label_column = roles.currency_label_column
    amount_column = roles.amount_column

    for row_idx, row in enumerate(table.rows):
        if profile.rows[row_idx].is_probable_header or profile.rows[row_idx].is_probable_note:
            continue

        amount: str | None = None
        currency: str | None = None

        moneys = _moneys_in_row(row)
        if len(moneys) > 1:
            observations.append(
                ClassificationObservation(
                    kind=ObservationKind.EXTRACTION_CONFLICT,
                    category=FeeCategory.FIXED_FEE,
                    table_id=table.table_id or table.document_id,
                    message=f"row {row_idx} has {len(moneys)} money tokens",
                )
            )
            continue

        if moneys:
            col, amount, currency = moneys[0]
        elif currency_label_column is not None and amount_column is not None:
            if currency_label_column < len(row.cells) and amount_column < len(row.cells):
                label_text = row.cells[currency_label_column].text.strip()
                amount_text = row.cells[amount_column].text.strip()
                if _is_iso_currency_code(label_text):
                    amount = parse_amount(amount_text)
                    currency = label_text.upper()
        else:
            continue

        if not currency or not amount:
            observations.append(
                ClassificationObservation(
                    kind=ObservationKind.EXTRACTION_CONFLICT,
                    category=FeeCategory.FIXED_FEE,
                    table_id=table.table_id or table.document_id,
                    message=f"row {row_idx} has no determinable currency or amount",
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
        if existing == amount:
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

    roles = _column_roles(profile, table)
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

    text = _norm(
        " ".join(
            [table.caption or ""]
            + list(table.section_path or [])
            + list(table.parent_path or [])
            + [header.text for header in table.headers]
        )
    )

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
        return ExtractionDecision(
            value=None, selected_rows=(), evidence=tuple(signals), observations=tuple(observations)
        )

    roles = _column_roles(profile, table)
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
            continue
        for col, pct in row_pcts:
            if roles.percentage_columns and col not in roles.percentage_columns:
                continue
            pcts.append((row_idx, pct))

    if not pcts:
        observations.append(
            ClassificationObservation(
                kind=ObservationKind.EXTRACTION_CONFLICT,
                category=FeeCategory.CURRENCY_CONVERSION,
                table_id=table.table_id or table.document_id,
                message="no conversion percentage found",
            )
        )
        return ExtractionDecision(
            value=None, selected_rows=tuple(selected_rows), evidence=tuple(signals), observations=tuple(observations)
        )

    unique_values = {pct for _, pct in pcts}
    if len(unique_values) > 1:
        observations.append(
            ClassificationObservation(
                kind=ObservationKind.EXTRACTION_CONFLICT,
                category=FeeCategory.CURRENCY_CONVERSION,
                table_id=table.table_id or table.document_id,
                message=f"conflicting conversion spreads: {sorted(unique_values)}",
            )
        )
        return ExtractionDecision(
            value=None, selected_rows=tuple(selected_rows), evidence=tuple(signals), observations=tuple(observations)
        )

    selected_pct = unique_values.pop()
    for row_idx, pct in pcts:
        if pct == selected_pct:
            selected_rows.append(row_idx)
    _add_evidence(
        signals,
        EvidenceCode.HAS_PERCENTAGE_COLUMN,
        EvidenceSource.STRUCTURAL,
        10,
        detail=f"conversion spread {selected_pct}%",
    )
    return ExtractionDecision(
        value=selected_pct,
        selected_rows=tuple(selected_rows),
        evidence=tuple(signals),
        observations=tuple(observations),
    )
