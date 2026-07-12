"""Reviewed fingerprint registry and canonical fingerprint builder.

A fingerprint is a stable, content-derived hash of a normalized table.  It
intentionally excludes translated captions, exact percentages, exact monetary
amounts, and market-specific identifiers so that the same structural family of
fee tables can be recognized across markets and page revisions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from .models import Table
from .profiles import TableProfile


class ClusterStatus(StrEnum):
    """Lifecycle status of a structural cluster in the registry."""

    CANDIDATE = "candidate"
    APPROVED = "approved"
    DEPRECATED = "deprecated"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ClusterRecord:
    """A manually reviewed structural cluster."""

    name: str
    category: str
    fingerprints: frozenset[str]
    document_ids: frozenset[str]
    required_features: frozenset[str]
    reviewed_examples: frozenset[str]
    status: ClusterStatus


class Fingerprint:
    """Type-safe wrapper for a canonical fingerprint string."""

    def __init__(self, value: str) -> None:
        if not value.startswith("sha256:"):
            raise ValueError("Fingerprints must start with 'sha256:'")
        self.value = value

    def __str__(self) -> str:
        return self.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Fingerprint):
            return NotImplemented
        return self.value == other.value


@dataclass(frozen=True)
class FingerprintComponents:
    """Human-readable components that form the canonical fingerprint input."""

    fingerprint_version: int
    row_count_bucket: str
    column_count: int
    column_signatures: tuple[dict, ...]
    has_multiple_currencies: bool
    has_percentage: bool
    has_money: bool
    has_additive_percentages: bool
    mixed_row_patterns: tuple[str, ...]
    percentage_operators: tuple[str, ...]
    fee_data_keys: tuple[str, ...]
    internal_names: tuple[str, ...]
    content_types: tuple[str, ...]
    has_reference: bool


class FingerprintBuilder:
    """Build canonical fingerprints from a table and its profile."""

    FINGERPRINT_VERSION = 1
    ALGORITHM = "sha256"

    @classmethod
    def build(cls, profile: TableProfile, table: Table | None = None) -> Fingerprint:
        """Return a stable fingerprint for *profile* and optional *table*."""
        components = cls.components(profile, table)
        canonical = json.dumps(
            components.__dict__,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        return Fingerprint(f"{cls.ALGORITHM}:{digest}")

    @classmethod
    def components(cls, profile: TableProfile, table: Table | None = None) -> FingerprintComponents:
        """Extract the canonical fingerprint components."""
        row_count_bucket = _row_count_bucket(profile.row_count)
        column_signatures = _column_signatures(profile)
        mixed_row_patterns = _mixed_row_patterns(profile)
        percentage_operators = _percentage_operators(profile, table)
        has_reference = _has_reference(table)

        return FingerprintComponents(
            fingerprint_version=cls.FINGERPRINT_VERSION,
            row_count_bucket=row_count_bucket,
            column_count=profile.column_count,
            column_signatures=column_signatures,
            has_multiple_currencies=profile.has_multiple_currencies,
            has_percentage=profile.has_percentage,
            has_money=profile.has_money,
            has_additive_percentages=profile.has_additive_percentages,
            mixed_row_patterns=mixed_row_patterns,
            percentage_operators=percentage_operators,
            fee_data_keys=tuple(sorted(profile.fee_data_keys)),
            internal_names=tuple(sorted(profile.internal_names)),
            content_types=tuple(sorted(profile.content_types)),
            has_reference=has_reference,
        )


@dataclass(frozen=True)
class FingerprintMatch:
    """Result of looking up a fingerprint in the registry."""

    fingerprint: Fingerprint
    cluster: ClusterRecord | None = None
    approved: bool = False


class FingerprintRegistry:
    """In-memory registry of reviewed structural fingerprints."""

    def __init__(self, clusters: dict[str, ClusterRecord]) -> None:
        self.clusters = clusters
        self._by_fingerprint: dict[Fingerprint, str] = {}
        self._by_document_id: dict[str, str] = {}
        for name, cluster in clusters.items():
            for fp in cluster.fingerprints:
                self._by_fingerprint[Fingerprint(fp)] = name
            for doc_id in cluster.document_ids:
                self._by_document_id[doc_id] = name

    @classmethod
    def load_builtin(cls) -> FingerprintRegistry:
        """Load the bundled ``classifier_clusters.json`` registry."""
        import importlib.resources

        from . import registries

        text = importlib.resources.files(registries).joinpath("classifier_clusters.json").read_text(
            encoding="utf-8"
        )
        return cls.load_json(text)

    @classmethod
    def load_json(cls, text: str) -> FingerprintRegistry:
        """Load a registry from a JSON string."""
        data = json.loads(text)
        clusters: dict[str, ClusterRecord] = {}
        for name, cluster in data.get("clusters", {}).items():
            clusters[name] = ClusterRecord(
                name=name,
                category=cluster.get("category", ""),
                fingerprints=frozenset(cluster.get("fingerprints", [])),
                document_ids=frozenset(cluster.get("document_ids", [])),
                required_features=frozenset(cluster.get("required_features", [])),
                reviewed_examples=frozenset(cluster.get("reviewed_examples", [])),
                status=ClusterStatus(cluster.get("status", "candidate")),
            )
        return cls(clusters)

    @classmethod
    def load_path(cls, path: Path) -> FingerprintRegistry:
        """Load a registry from a JSON file path."""
        return cls.load_json(path.read_text(encoding="utf-8"))

    def lookup(self, fingerprint: str | Fingerprint) -> FingerprintMatch:
        """Return the matching cluster for a fingerprint, if any."""
        fp = fingerprint if isinstance(fingerprint, Fingerprint) else Fingerprint(fingerprint)
        name = self._by_fingerprint.get(fp)
        if name is None:
            return FingerprintMatch(fingerprint=fp, approved=False)
        cluster = self.clusters[name]
        return FingerprintMatch(fingerprint=fp, cluster=cluster, approved=cluster.status == ClusterStatus.APPROVED)

    def cluster_for_document_id(self, document_id: str) -> ClusterRecord | None:
        """Return the cluster associated with a document ID."""
        name = self._by_document_id.get(document_id)
        return self.clusters.get(name) if name else None

    def approved_for_category(self, fingerprint: str | Fingerprint, category: str) -> bool:
        """Return True when the fingerprint is approved for the given category."""
        match = self.lookup(fingerprint)
        if not match.approved or match.cluster is None:
            return False
        return match.cluster.category == category


def _row_count_bucket(row_count: int) -> str:
    """Coarse bucket for row count so the fingerprint is not fragile."""
    if row_count <= 0:
        return "0"
    if row_count <= 2:
        return "1-2"
    if row_count <= 5:
        return "3-5"
    if row_count <= 10:
        return "6-10"
    return "11+"


def _column_signatures(profile: TableProfile) -> tuple[dict, ...]:
    """Return a stable, language-independent signature for each column."""
    signatures: list[dict] = []
    for col in profile.columns:
        kinds: set[str] = set()
        if col.percentage_row_count:
            kinds.add("percentage")
        if col.money_row_count:
            kinds.add("money")
        if col.text_row_count:
            kinds.add("text")
        role = "label"
        if col.column_index in profile.percentage_columns:
            role = "percentage"
        elif col.column_index in profile.money_columns:
            role = "money"
        signatures.append(
            {
                "index": col.column_index,
                "role": role,
                "kinds": sorted(kinds),
            }
        )
    return tuple(signatures)


def _mixed_row_patterns(profile: TableProfile) -> tuple[str, ...]:
    """Return the token-kind patterns of mixed percentage+money rows."""
    patterns: list[str] = []
    for row_profile in profile.rows:
        if row_profile.row_index in profile.mixed_percentage_money_rows:
            patterns.append(",".join(row_profile.token_kind_pattern))
    return tuple(sorted(patterns))


def _percentage_operators(profile: TableProfile, table: Table | None) -> tuple[str, ...]:
    """Return sorted, distinct percentage-token operators from the raw table."""
    if table is None:
        return ()
    operators: set[str] = set()
    for row in table.rows:
        for cell in row.cells:
            for token in cell.tokens:
                if token.kind == "percentage" and token.operator:
                    operators.add(token.operator)
    return tuple(sorted(operators))


def _has_reference(table: Table | None) -> bool:
    """Return whether the table is or references another table."""
    if table is None:
        return bool(table is not None)
    return bool(table.reference_id or table.source_table_ids)
