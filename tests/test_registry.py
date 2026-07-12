"""Tests for the fingerprint registry and builder."""

from __future__ import annotations

import json

from paypal_fee_crawler.models import Cell, Row, Table
from paypal_fee_crawler.pricing_tokens import tokenize_text
from paypal_fee_crawler.profiles import build_table_profile
from paypal_fee_crawler.registry import (
    ClusterRecord,
    ClusterStatus,
    FingerprintBuilder,
    FingerprintRegistry,
)


def _table(rows: list[list[str]]) -> Table:
    return Table(
        rows=[
            Row(
                row_id=str(idx),
                cells=[Cell(text=cell, tokens=tokenize_text(cell)) for cell in row],
            )
            for idx, row in enumerate(rows)
        ],
    )


def test_fingerprint_builder_returns_sha256() -> None:
    table = _table([["2.99%", "0.39 EUR"]])
    profile = build_table_profile(table)
    fingerprint = FingerprintBuilder.build(profile, table)
    assert fingerprint.value.startswith("sha256:")


def test_fingerprint_builder_same_structure_same_fingerprint() -> None:
    table1 = _table([["Commercial", "2.99%", "0.39 EUR"]])
    table2 = _table([["Commercial", "2.49%", "0.49 USD"]])
    profile1 = build_table_profile(table1)
    profile2 = build_table_profile(table2)
    fp1 = FingerprintBuilder.build(profile1, table1)
    fp2 = FingerprintBuilder.build(profile2, table2)
    assert fp1.value == fp2.value


def test_fingerprint_builder_differs_with_column_shape() -> None:
    table1 = _table([["Commercial", "2.99%", "0.39 EUR"]])
    table2 = _table([["2.99%", "0.39 EUR"]])
    fp1 = FingerprintBuilder.build(build_table_profile(table1), table1)
    fp2 = FingerprintBuilder.build(build_table_profile(table2), table2)
    assert fp1.value != fp2.value


def test_fingerprint_registry_lookup_approved() -> None:
    registry = FingerprintRegistry(
        {
            "commercial-fixed-fees-v1": ClusterRecord(
                name="commercial-fixed-fees-v1",
                category="fixed_fee",
                fingerprints=frozenset({"sha256:abc123"}),
                document_ids=frozenset({"FEETB18"}),
                required_features=frozenset({"money_column", "multiple_currencies"}),
                reviewed_examples=frozenset({"DE", "GB"}),
                status=ClusterStatus.APPROVED,
            )
        }
    )
    match = registry.lookup("sha256:abc123")
    assert match.approved
    assert match.cluster is not None
    assert match.cluster.category == "fixed_fee"
    assert registry.cluster_for_document_id("FEETB18") is not None
    assert registry.approved_for_category("sha256:abc123", "fixed_fee")
    assert not registry.approved_for_category("sha256:abc123", "standard_commercial")


def test_fingerprint_registry_builtin_loads_empty() -> None:
    registry = FingerprintRegistry.load_builtin()
    assert registry.clusters == {}


def test_fingerprint_registry_load_json() -> None:
    text = json.dumps(
        {
            "fingerprint_version": 1,
            "clusters": {
                "c": {
                    "category": "fixed_fee",
                    "fingerprints": ["sha256:abc"],
                    "document_ids": ["FEETB1"],
                    "required_features": ["money_column"],
                    "reviewed_examples": ["DE"],
                    "status": "approved",
                }
            },
        }
    )
    registry = FingerprintRegistry.load_json(text)
    assert "c" in registry.clusters
    assert registry.clusters["c"].status == ClusterStatus.APPROVED
