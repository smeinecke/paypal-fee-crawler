"""Structural scoring engine for PayPal fee table classification.

This module implements the fail-closed, score-based classifier described in
`update.md` (PR 1A).  It operates on normalized `Table` objects and produces
`ScoreResult` / `ClassificationDecision` records that are independent of the
legacy order-dependent predicates in `classify.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from .models import FeeToken, Table
from .normalize import clean_text
from .profiles import TableProfile, build_table_profile
from .registry import ClusterRecord, FingerprintBuilder, FingerprintRegistry


class FeeCategory(StrEnum):
    STANDARD_COMMERCIAL = "standard_commercial"
    FIXED_FEE = "fixed_fee"
    INTERNATIONAL_SURCHARGE = "international_surcharge"
    CURRENCY_CONVERSION = "currency_conversion"
    OTHER = "other"


class EvidenceSource(StrEnum):
    STRUCTURAL = "structural"
    METADATA = "metadata"
    RELATIONSHIP = "relationship"
    REGISTRY = "registry"
    LEXICAL = "lexical"


class EvidenceCode(StrEnum):
    HAS_PERCENTAGE_COLUMN = "has_percentage_column"
    HAS_MONEY_COLUMN = "has_money_column"
    HAS_MIXED_PERCENT_MONEY_ROW = "has_mixed_percent_money_row"
    HAS_MULTIPLE_CURRENCIES = "has_multiple_currencies"
    HAS_ADDITIVE_PERCENTAGES = "has_additive_percentages"
    METADATA_KEY_MATCH = "metadata_key_match"
    INTERNAL_NAME_MATCH = "internal_name_match"
    KNOWN_DOCUMENT_ID = "known_document_id"
    KNOWN_FINGERPRINT = "known_fingerprint"
    REFERENCE_CONTEXT_MATCH = "reference_context_match"
    POSITIVE_LEXICAL_HINT = "positive_lexical_hint"
    NEGATIVE_LEXICAL_HINT = "negative_lexical_hint"


class BlockerCode(StrEnum):
    ONLY_PERCENTAGES_FOR_FIXED_FEE = "only_percentages_for_fixed_fee"
    ONLY_MONEY_FOR_PERCENTAGE_CATEGORY = "only_money_for_percentage_category"
    NO_USABLE_VALUES = "no_usable_values"
    INCOMPATIBLE_COLUMN_SHAPE = "incompatible_column_shape"
    INCOMPATIBLE_FINGERPRINT = "incompatible_fingerprint"


@dataclass(frozen=True)
class EvidenceSignal:
    code: EvidenceCode
    source: EvidenceSource
    weight: int
    detail: str | None = None


@dataclass(frozen=True)
class ScoreResult:
    category: FeeCategory
    score: int
    signals: tuple[EvidenceSignal, ...]
    blockers: tuple[BlockerCode, ...]

    @property
    def eligible(self) -> bool:
        return not self.blockers


@dataclass(frozen=True)
class ClassificationDecision:
    status: Literal["selected", "ambiguous", "unclassified"]
    selected_category: FeeCategory | None
    selected_score: ScoreResult | None
    ranked_scores: tuple[ScoreResult, ...]
    ambiguity_reason: str | None
    winner_margin: int | None


def _required_features_satisfied(cluster: ClusterRecord, profile: TableProfile, table: Table) -> bool:
    """Return True when *table* and *profile* satisfy a cluster's required features."""
    checks: dict[str, bool] = {
        "has_percentage": profile.has_percentage,
        "has_money": profile.has_money,
        "has_multiple_currencies": profile.has_multiple_currencies,
        "has_additive_percentages": profile.has_additive_percentages,
        "mixed_percentage_money_row": bool(profile.mixed_percentage_money_rows),
        "has_reference": bool(table.reference_id or table.source_table_ids),
        "has_headers": bool(table.headers),
        "has_fee_data_keys": bool(profile.fee_data_keys),
        "has_internal_names": bool(profile.internal_names),
        "has_content_types": bool(profile.content_types),
    }
    return all(checks.get(feature, True) for feature in cluster.required_features)


def _registry_signals(
    registry: FingerprintRegistry | None,
    profile: TableProfile,
    table: Table,
    category: FeeCategory,
) -> tuple[list[EvidenceSignal], list[BlockerCode]]:
    """Return registry-derived signals and blockers for a category."""
    signals: list[EvidenceSignal] = []
    blockers: list[BlockerCode] = []
    if registry is None:
        return signals, blockers
    fingerprint = FingerprintBuilder.build(profile, table)
    match = registry.lookup(fingerprint)
    fingerprint_approved = bool(
        match.approved
        and match.cluster is not None
        and match.cluster.category == category.value
        and _required_features_satisfied(match.cluster, profile, table)
    )
    if fingerprint_approved:
        if match.cluster is None:
            return signals, blockers
        signals.append(
            EvidenceSignal(
                code=EvidenceCode.KNOWN_FINGERPRINT,
                source=EvidenceSource.REGISTRY,
                weight=20,
                detail=f"approved cluster {match.cluster.name}",
            )
        )

    document_id = table.document_id or ""
    doc_cluster = registry.cluster_for_document_id(document_id)
    if doc_cluster is not None and doc_cluster.category != category.value and not fingerprint_approved:
        # A fingerprint match for the current category is stronger evidence than
        # a document-id mismatch elsewhere; do not block in that case.
        blockers.append(BlockerCode.INCOMPATIBLE_FINGERPRINT)
    elif doc_cluster is not None and doc_cluster.category == category.value and not fingerprint_approved:
        # Only add registry document-id evidence when the fingerprint did not
        # already approve the table, so the strongest evidence is recorded.
        signals.append(
            EvidenceSignal(
                code=EvidenceCode.KNOWN_DOCUMENT_ID,
                source=EvidenceSource.REGISTRY,
                weight=15,
                detail=f"document_id {document_id} in cluster {doc_cluster.name}",
            )
        )
    return signals, blockers


def _has_registry_document_id(signals: list[EvidenceSignal]) -> bool:
    """Return True if the registry already supplied a known-document-id signal."""
    return any(s.code == EvidenceCode.KNOWN_DOCUMENT_ID and s.source == EvidenceSource.REGISTRY for s in signals)


MAX_CATEGORY_SCORE = 100
MINIMUM_SCORE = 60
MINIMUM_MARGIN = 15

MARKET_ALIASES = {
    "GB": frozenset({"gb", "uk"}),
}

REGION_GROUPS = {
    "US_CA": frozenset({"us", "usa", "ca", "canada"}),
    "EEA": frozenset({"eea", "ewr", "ehp", "e.u", "european economic area", "european economic"}),
    "GB": frozenset(
        {"gb", "uk", "united kingdom", "great britain", "britain", "england", "royaume-uni", "grossbritannien"}
    ),
    "OTHER": frozenset(
        {
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
        }
    ),
}

# ---------------------------------------------------------------------------
# Document ID sets used as relationship evidence.  These are intentionally the
# same sets that the legacy classifier uses; they are aliases for reviewed
# structural clusters rather than stand-alone truth.
# ---------------------------------------------------------------------------

STANDARD_DOC_IDS = frozenset({"FEETB16", "FEETB359"})
FIXED_DOC_IDS = frozenset(
    {"FEETB18", "FEETB306", "FEETB261", "FEETB872", "FEETB871", "FEETB354", "FEETB363", "FEETB440", "FEETB441"}
)
INTERNATIONAL_DOC_IDS = frozenset({"FEETB91", "FEETB100", "FEETB382", "FEETB153", "FEETB533"})
CONVERSION_DOC_IDS = frozenset(
    {"FEETB539", "FEETB128", "FEETB159", "FEETB160", "FEETB154", "FEETB156", "FEETB157", "FEETB338"}
)

# ---------------------------------------------------------------------------
# Lexical keyword lists.  These are intentionally narrower than the legacy
# keyword lists; they are used as low-weight context, not as decisive signals.
# ---------------------------------------------------------------------------

_POS_STANDARD = (
    "standard",
    "commercial",
    "comercial",
    "comerciales",
    "comercio",
    "transaction",
    "transactions",
    "transacción",
    "transacciones",
    "transakcie",
    "transakcií",
    "merchant",
    "händler",
    "domestic",
    "inland",
    "nacionales",
    "online payment",
    "online card",
    "sadzba",
    "tarifa",
    "comisión",
    "comision",
    "poplatok",
    "poplatky",
    "poplatkov",
    "percentage",
    "percent",
    "fee",
    "fees",
    "gebühr",
    "gebühren",
)

_NEG_STANDARD = (
    "fixed fee",
    "festgebühr",
    "feste gebühr",
    "currency",
    "währung",
    "international",
    "internacional",
    "cross border",
    "cross-border",
    "conversion",
    "conversión",
    "donation",
    "charity",
    "nonprofit",
    "non-profit",
    "dispute",
    "chargeback",
    "micropayment",
    "point of sale",
    "other fees",
    "sonstige gebühren",
    "additional service fee",
    "additional percentage-based fee",
    "withdrawal",
    "payout",
    "crypto",
    "service fee",
)

_POS_FIXED = (
    "fixed fee",
    "fixed fees",
    "festgebühr",
    "feste gebühr",
    "per transaction",
    "pro transaktion",
    "por transacciones",
    "por transacción",
    "based on currency",
    "währung",
    "currency",
    "moneda",
    "divisa",
    "mena",
    "mien",
    "commercial",
    "comercial",
    "comerciales",
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

_NEG_FIXED = (
    "charity",
    "donation",
    "dispute",
    "chargeback",
    "micropayment",
    "point of sale",
    "international",
    "conversion",
    "other fees",
    "website payments pro",
    "online card",
    "online payment",
    "disbursement",
    "region",
    "alternative payment",
    "qr code",
    "qr-code",
    "invoicing",
    "interchange",
    "max cap",
    "maximum fee",
    "minimum fee",
    "instant transfer",
    "cryptocurrency",
    "crypto",
    "payout",
    "payouts",
    "mindest",
    "höchst",
    "additional",
    "service fee",
)

_POS_INTERNATIONAL = (
    "international",
    "internacional",
    "internacionales",
    "surcharge",
    "surcharges",
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
    "percentuálny",
    "percentuálna",
    "percentuálne",
    "payer region",
    "markt/region",
    "market/region",
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
    "transacciones internacionales",
    "medzinárodné transakcie",
    "medzinárodných transakcií",
    "zahraničné transakcie",
)

_NEG_INTERNATIONAL = (
    "fixed fee",
    "festgebühr",
    "currency conversion",
    "währungsumrechnung",
    "conversion",
    "umrechnung",
    "wechselkurs",
    "donation",
    "charity",
    "nonprofit",
    "micropayment",
    "dispute",
    "chargeback",
    "other fees",
    "standard",
    "inland",
    "domestic",
    "service fee",
    "personal",
    "alternative payment",
    "qr code",
    "qr-code",
    "online card",
    "online payment",
    "payout",
    "interchange",
    "max cap",
    "maximum fee",
    "minimum fee",
    "instant transfer",
    "cryptocurrency",
    "crypto",
    "point of sale",
    "card present",
    "manual card",
)

_POS_CONVERSION = (
    "currency",
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

_NEG_CONVERSION = (
    "fixed fee",
    "festgebühr",
    "donation",
    "charity",
    "nonprofit",
    "dispute",
    "chargeback",
    "micropayment",
    "other fees",
    "point of sale",
    "mindest",
    "höchst",
    "max cap",
    "minimum fee",
    "maximum fee",
    "crypto",
    "withdrawal",
    "comisión fija",
    "fija",
    "fixný poplatok",
    "fixny poplatok",
    "fixná",
    "fixne",
)


# ---------------------------------------------------------------------------
# Text / token helpers
# ---------------------------------------------------------------------------


def _norm(text: str | None) -> str:
    return clean_text(text or "").casefold()


def _normalize_market_text(text: str) -> str:
    """Return a normalized, space-separated token string for market matching.

    Dots inside single-letter acronyms (e.g. ``U.S.``) are collapsed, then all
    remaining non-word characters are replaced with spaces.  This preserves
    multi-word phrases while preventing short codes from matching inside larger
    words (``us`` inside ``usaus`` or ``business``).
    """
    text = _norm(text)
    # Collapse acronyms like U.S. / e.u. into us / eu.
    text = re.sub(r"(?<=\b[^\W\d_])\.(?=[^\W\d_]\b)", "", text)
    text = re.sub(r"(?<=\w)\.(?!\w)", "", text)
    text = re.sub(r"[\W_]+", " ", text)
    return " ".join(text.split())


def _token_contains_term(text: str, term: str) -> bool:
    """Check whether ``term`` appears as a whole token/phrase in ``text``."""
    norm = _normalize_market_text(text)
    term_norm = _normalize_market_text(term)
    if not norm or not term_norm:
        return False
    return f" {term_norm} " in f" {norm} "


def _table_text(table: Table, include_rows: bool = True) -> str:
    """Combined caption, section path, parent path, headers and optionally rows."""
    parts = list(table.section_path or [])
    parts.extend(table.parent_path or [])
    parts.append(table.caption or "")
    for header in table.headers:
        parts.append(header.text)
    if include_rows:
        for row in table.rows:
            for cell in row.cells:
                parts.append(cell.text)
    return _norm(" ".join(parts))


def _iter_tokens(table: Table) -> list[FeeToken]:
    """Return all tokens from a table's headers and rows."""
    tokens: list[FeeToken] = []
    for header in table.headers:
        tokens.extend(header.tokens)
    for row in table.rows:
        for cell in row.cells:
            tokens.extend(cell.tokens)
    return tokens


# ---------------------------------------------------------------------------
# Market / region matching
# ---------------------------------------------------------------------------


def market_code_matches(text: str, market_code: str) -> bool:
    """Return whether ``text`` contains the canonical market code or an alias."""
    code = market_code.strip().lower()
    if not code:
        return False
    terms = {code} | MARKET_ALIASES.get(code.upper(), frozenset())
    return any(_token_contains_term(text, term) for term in terms)


def region_from_text(text: str) -> str | None:
    """Map a region label to its canonical region group using token matching."""
    for region in ("EEA", "GB", "US_CA", "OTHER"):
        terms = REGION_GROUPS.get(region, frozenset())
        if any(_token_contains_term(text, term) for term in terms):
            return region
    return None


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _add_signal(
    signals: list[EvidenceSignal],
    code: EvidenceCode,
    source: EvidenceSource,
    weight: int,
    detail: str | None = None,
) -> None:
    signals.append(EvidenceSignal(code=code, source=source, weight=weight, detail=detail))


def _clamp_score(raw: int) -> int:
    return max(0, min(MAX_CATEGORY_SCORE, raw))


def _keyword_score(text: str, positive: tuple[str, ...], negative: tuple[str, ...]) -> int:
    """Return a lexical score between -10 and +10 based on keyword matches."""
    norm = _norm(text)
    if not norm:
        return 0
    score = 0
    for kw in positive:
        if _token_contains_term(norm, kw):
            score += 2
    for kw in negative:
        if _token_contains_term(norm, kw):
            score -= 3
    return max(-10, min(10, score))


def _metadata_matches_category(table: Table, category: FeeCategory) -> tuple[bool, str | None]:
    """Return whether token metadata points toward ``category``."""
    keys: set[str] = set()
    for token in _iter_tokens(table):
        if token.fee_data_key:
            keys.add(token.fee_data_key.lower())
        if token.internal_name:
            keys.add(token.internal_name.lower())
        if token.content_type:
            keys.add(token.content_type.lower())

    if not keys:
        return False, None

    hints: dict[FeeCategory, set[str]] = {
        FeeCategory.STANDARD_COMMERCIAL: {
            "standard",
            "commercial",
            "transaction",
            "payment",
            "merchant",
            "percentage",
            "rate",
        },
        FeeCategory.FIXED_FEE: {"fixed", "currency", "per_transaction", "pertransaction"},
        FeeCategory.INTERNATIONAL_SURCHARGE: {"international", "crossborder", "surcharge", "region"},
        FeeCategory.CURRENCY_CONVERSION: {"conversion", "currency_conversion", "spread", "exchange"},
    }

    target = hints.get(category, set())
    matches = keys & target
    if matches:
        return True, ",".join(sorted(matches))
    return False, None


# ---------------------------------------------------------------------------
# Category scorers
# ---------------------------------------------------------------------------


def score_standard_commercial(
    table: Table,
    market_code: str | None = None,
    locale: str | None = None,
    registry: FingerprintRegistry | None = None,
) -> ScoreResult:
    """Score a table for the standard-commercial category."""
    category = FeeCategory.STANDARD_COMMERCIAL
    profile = build_table_profile(table)
    signals: list[EvidenceSignal] = []
    blockers: list[BlockerCode] = []

    registry_signals, registry_blockers = _registry_signals(registry, profile, table, category)
    signals.extend(registry_signals)
    blockers.extend(registry_blockers)

    if not profile.has_percentage:
        blockers.append(BlockerCode.ONLY_MONEY_FOR_PERCENTAGE_CATEGORY)
    else:
        if profile.percentage_columns:
            _add_signal(signals, EvidenceCode.HAS_PERCENTAGE_COLUMN, EvidenceSource.STRUCTURAL, 40)
        if profile.mixed_percentage_money_rows:
            _add_signal(signals, EvidenceCode.HAS_MIXED_PERCENT_MONEY_ROW, EvidenceSource.STRUCTURAL, 25)
        if profile.has_additive_percentages:
            _add_signal(signals, EvidenceCode.HAS_ADDITIVE_PERCENTAGES, EvidenceSource.STRUCTURAL, 10)

    if not profile.has_percentage and not profile.has_money:
        blockers.append(BlockerCode.NO_USABLE_VALUES)

    if profile.money_columns and not profile.percentage_columns:
        # A pure money table is not standard commercial.
        blockers.append(BlockerCode.ONLY_MONEY_FOR_PERCENTAGE_CATEGORY)

    doc_id = (table.document_id or "").upper()
    if doc_id in STANDARD_DOC_IDS and not _has_registry_document_id(signals):
        _add_signal(signals, EvidenceCode.KNOWN_DOCUMENT_ID, EvidenceSource.RELATIONSHIP, 20, detail=doc_id)

    meta_match, meta_detail = _metadata_matches_category(table, category)
    if meta_match:
        _add_signal(signals, EvidenceCode.METADATA_KEY_MATCH, EvidenceSource.METADATA, 20, detail=meta_detail)

    text = _table_text(table, include_rows=False)
    lexical = _keyword_score(text, _POS_STANDARD, _NEG_STANDARD)
    if lexical > 0:
        _add_signal(signals, EvidenceCode.POSITIVE_LEXICAL_HINT, EvidenceSource.LEXICAL, lexical)
    elif lexical < 0:
        _add_signal(signals, EvidenceCode.NEGATIVE_LEXICAL_HINT, EvidenceSource.LEXICAL, lexical)

    if table.reference_id or table.source_table_ids:
        _add_signal(signals, EvidenceCode.REFERENCE_CONTEXT_MATCH, EvidenceSource.RELATIONSHIP, 5)

    score = _clamp_score(sum(s.weight for s in signals))
    return ScoreResult(
        category=category,
        score=score,
        signals=tuple(signals),
        blockers=tuple(blockers),
    )


def score_fixed_fee(
    table: Table,
    market_code: str | None = None,
    locale: str | None = None,
    registry: FingerprintRegistry | None = None,
) -> ScoreResult:
    """Score a table for the fixed-fee category."""
    category = FeeCategory.FIXED_FEE
    profile = build_table_profile(table)
    signals: list[EvidenceSignal] = []
    blockers: list[BlockerCode] = []

    registry_signals, registry_blockers = _registry_signals(registry, profile, table, category)
    signals.extend(registry_signals)
    blockers.extend(registry_blockers)

    if not profile.has_money:
        blockers.append(BlockerCode.ONLY_PERCENTAGES_FOR_FIXED_FEE)
    else:
        if profile.money_columns:
            _add_signal(signals, EvidenceCode.HAS_MONEY_COLUMN, EvidenceSource.STRUCTURAL, 40)
        if profile.has_multiple_currencies:
            _add_signal(signals, EvidenceCode.HAS_MULTIPLE_CURRENCIES, EvidenceSource.STRUCTURAL, 20)
        if profile.mixed_percentage_money_rows:
            _add_signal(signals, EvidenceCode.HAS_MIXED_PERCENT_MONEY_ROW, EvidenceSource.STRUCTURAL, 5)

    if not profile.has_percentage and not profile.has_money:
        blockers.append(BlockerCode.NO_USABLE_VALUES)

    if profile.has_percentage and not profile.has_money:
        blockers.append(BlockerCode.ONLY_PERCENTAGES_FOR_FIXED_FEE)

    doc_id = (table.document_id or "").upper()
    if doc_id in FIXED_DOC_IDS and not _has_registry_document_id(signals):
        _add_signal(signals, EvidenceCode.KNOWN_DOCUMENT_ID, EvidenceSource.RELATIONSHIP, 20, detail=doc_id)

    meta_match, meta_detail = _metadata_matches_category(table, category)
    if meta_match:
        _add_signal(signals, EvidenceCode.METADATA_KEY_MATCH, EvidenceSource.METADATA, 20, detail=meta_detail)

    text = _table_text(table, include_rows=False)
    lexical = _keyword_score(text, _POS_FIXED, _NEG_FIXED)
    if lexical > 0:
        _add_signal(signals, EvidenceCode.POSITIVE_LEXICAL_HINT, EvidenceSource.LEXICAL, lexical)
    elif lexical < 0:
        _add_signal(signals, EvidenceCode.NEGATIVE_LEXICAL_HINT, EvidenceSource.LEXICAL, lexical)

    if table.reference_id or table.source_table_ids:
        _add_signal(signals, EvidenceCode.REFERENCE_CONTEXT_MATCH, EvidenceSource.RELATIONSHIP, 5)

    score = _clamp_score(sum(s.weight for s in signals))
    return ScoreResult(
        category=category,
        score=score,
        signals=tuple(signals),
        blockers=tuple(blockers),
    )


def score_international_surcharge(
    table: Table,
    market_code: str | None = None,
    locale: str | None = None,
    registry: FingerprintRegistry | None = None,
) -> ScoreResult:
    """Score a table for the international-surcharge category."""
    category = FeeCategory.INTERNATIONAL_SURCHARGE
    profile = build_table_profile(table)
    signals: list[EvidenceSignal] = []
    blockers: list[BlockerCode] = []

    registry_signals, registry_blockers = _registry_signals(registry, profile, table, category)
    signals.extend(registry_signals)
    blockers.extend(registry_blockers)

    if not profile.has_percentage:
        blockers.append(BlockerCode.ONLY_MONEY_FOR_PERCENTAGE_CATEGORY)
    else:
        if profile.percentage_columns:
            _add_signal(signals, EvidenceCode.HAS_PERCENTAGE_COLUMN, EvidenceSource.STRUCTURAL, 40)
        if profile.has_multiple_currencies:
            _add_signal(signals, EvidenceCode.HAS_MULTIPLE_CURRENCIES, EvidenceSource.STRUCTURAL, 10)
        if profile.has_additive_percentages:
            _add_signal(signals, EvidenceCode.HAS_ADDITIVE_PERCENTAGES, EvidenceSource.STRUCTURAL, 25)

    if not profile.has_percentage and not profile.has_money:
        blockers.append(BlockerCode.NO_USABLE_VALUES)

    if profile.has_money and not profile.has_percentage:
        blockers.append(BlockerCode.ONLY_MONEY_FOR_PERCENTAGE_CATEGORY)

    doc_id = (table.document_id or "").upper()
    if doc_id in INTERNATIONAL_DOC_IDS and not _has_registry_document_id(signals):
        _add_signal(signals, EvidenceCode.KNOWN_DOCUMENT_ID, EvidenceSource.RELATIONSHIP, 20, detail=doc_id)

    meta_match, meta_detail = _metadata_matches_category(table, category)
    if meta_match:
        _add_signal(signals, EvidenceCode.METADATA_KEY_MATCH, EvidenceSource.METADATA, 20, detail=meta_detail)

    text = _table_text(table, include_rows=False)
    lexical = _keyword_score(text, _POS_INTERNATIONAL, _NEG_INTERNATIONAL)
    if lexical > 0:
        _add_signal(signals, EvidenceCode.POSITIVE_LEXICAL_HINT, EvidenceSource.LEXICAL, lexical)
    elif lexical < 0:
        _add_signal(signals, EvidenceCode.NEGATIVE_LEXICAL_HINT, EvidenceSource.LEXICAL, lexical)

    if table.reference_id or table.source_table_ids:
        _add_signal(signals, EvidenceCode.REFERENCE_CONTEXT_MATCH, EvidenceSource.RELATIONSHIP, 5)

    score = _clamp_score(sum(s.weight for s in signals))
    return ScoreResult(
        category=category,
        score=score,
        signals=tuple(signals),
        blockers=tuple(blockers),
    )


def score_conversion(
    table: Table,
    market_code: str | None = None,
    locale: str | None = None,
    registry: FingerprintRegistry | None = None,
) -> ScoreResult:
    """Score a table for the currency-conversion category."""
    category = FeeCategory.CURRENCY_CONVERSION
    profile = build_table_profile(table)
    signals: list[EvidenceSignal] = []
    blockers: list[BlockerCode] = []

    registry_signals, registry_blockers = _registry_signals(registry, profile, table, category)
    signals.extend(registry_signals)
    blockers.extend(registry_blockers)

    if not profile.has_percentage:
        blockers.append(BlockerCode.ONLY_MONEY_FOR_PERCENTAGE_CATEGORY)
    else:
        if profile.percentage_columns:
            _add_signal(signals, EvidenceCode.HAS_PERCENTAGE_COLUMN, EvidenceSource.STRUCTURAL, 40)
        if profile.has_multiple_currencies:
            _add_signal(signals, EvidenceCode.HAS_MULTIPLE_CURRENCIES, EvidenceSource.STRUCTURAL, 10)
        if profile.has_additive_percentages:
            _add_signal(signals, EvidenceCode.HAS_ADDITIVE_PERCENTAGES, EvidenceSource.STRUCTURAL, 25)

    if not profile.has_percentage and not profile.has_money:
        blockers.append(BlockerCode.NO_USABLE_VALUES)

    if profile.has_money and not profile.has_percentage:
        blockers.append(BlockerCode.ONLY_MONEY_FOR_PERCENTAGE_CATEGORY)

    doc_id = (table.document_id or "").upper()
    if doc_id in CONVERSION_DOC_IDS and not _has_registry_document_id(signals):
        _add_signal(signals, EvidenceCode.KNOWN_DOCUMENT_ID, EvidenceSource.RELATIONSHIP, 20, detail=doc_id)

    meta_match, meta_detail = _metadata_matches_category(table, category)
    if meta_match:
        _add_signal(signals, EvidenceCode.METADATA_KEY_MATCH, EvidenceSource.METADATA, 20, detail=meta_detail)

    text = _table_text(table, include_rows=False)
    lexical = _keyword_score(text, _POS_CONVERSION, _NEG_CONVERSION)
    if lexical > 0:
        _add_signal(signals, EvidenceCode.POSITIVE_LEXICAL_HINT, EvidenceSource.LEXICAL, lexical)
    elif lexical < 0:
        _add_signal(signals, EvidenceCode.NEGATIVE_LEXICAL_HINT, EvidenceSource.LEXICAL, lexical)

    if table.reference_id or table.source_table_ids:
        _add_signal(signals, EvidenceCode.REFERENCE_CONTEXT_MATCH, EvidenceSource.RELATIONSHIP, 5)

    # Conversion is fragile; require at least metadata, relationship, or strong
    # structural evidence beyond a single percentage.
    has_strong_evidence = bool(
        {EvidenceCode.KNOWN_DOCUMENT_ID, EvidenceCode.METADATA_KEY_MATCH, EvidenceCode.INTERNAL_NAME_MATCH}
        & {s.code for s in signals}
        or len(profile.percentage_columns) >= 1
        and lexical > 0
    )
    if not has_strong_evidence and not blockers:
        # Not a blocker, but the score will likely be too low to win.
        _add_signal(signals, EvidenceCode.NEGATIVE_LEXICAL_HINT, EvidenceSource.LEXICAL, -15)

    score = _clamp_score(sum(s.weight for s in signals))
    return ScoreResult(
        category=category,
        score=score,
        signals=tuple(signals),
        blockers=tuple(blockers),
    )


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def score_all_categories(
    table: Table,
    market_code: str | None = None,
    locale: str | None = None,
    registry: FingerprintRegistry | None = None,
) -> tuple[ScoreResult, ...]:
    """Return comparable scores for all four core categories."""
    return (
        score_standard_commercial(table, market_code, locale, registry),
        score_fixed_fee(table, market_code, locale, registry),
        score_international_surcharge(table, market_code, locale, registry),
        score_conversion(table, market_code, locale, registry),
    )


def select_category(
    scores: tuple[ScoreResult, ...],
    minimum_score: int = MINIMUM_SCORE,
    minimum_margin: int = MINIMUM_MARGIN,
) -> ClassificationDecision:
    """Rank scores and return a deterministic classification decision."""
    ranked = tuple(sorted(scores, key=lambda s: (s.score, s.category.value), reverse=True))

    eligible = [s for s in ranked if s.eligible]
    if not eligible:
        return ClassificationDecision(
            status="unclassified",
            selected_category=None,
            selected_score=None,
            ranked_scores=ranked,
            ambiguity_reason="no eligible category",
            winner_margin=None,
        )

    winner = eligible[0]
    runner_up = eligible[1] if len(eligible) > 1 else None

    if winner.score < minimum_score:
        return ClassificationDecision(
            status="unclassified",
            selected_category=None,
            selected_score=None,
            ranked_scores=ranked,
            ambiguity_reason=f"winner score {winner.score} below minimum {minimum_score}",
            winner_margin=None,
        )

    if runner_up is not None and (winner.score - runner_up.score) < minimum_margin:
        return ClassificationDecision(
            status="ambiguous",
            selected_category=None,
            selected_score=None,
            ranked_scores=ranked,
            ambiguity_reason=(
                f"winner {winner.category.value} score {winner.score} and runner-up "
                f"{runner_up.category.value} score {runner_up.score} margin "
                f"{winner.score - runner_up.score} below minimum {minimum_margin}"
            ),
            winner_margin=winner.score - runner_up.score,
        )

    return ClassificationDecision(
        status="selected",
        selected_category=winner.category,
        selected_score=winner,
        ranked_scores=ranked,
        ambiguity_reason=None,
        winner_margin=(winner.score - runner_up.score) if runner_up else None,
    )
