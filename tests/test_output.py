"""Tests for deterministic output generation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from paypal_fee_crawler import output as output_module
from paypal_fee_crawler.exceptions import ValidationError as CrawlerValidationError
from paypal_fee_crawler.models import Cell, CountryOutput, DerivedFees, Market, Row, Source, Table
from paypal_fee_crawler.output import OutputPublisher
from paypal_fee_crawler.validation import validate_output_tree


def _make_output(cc: str) -> CountryOutput:
    return CountryOutput(
        schema_version=1,
        market=Market(paypal_market_code=cc, iso_country_code=cc, country_name=cc),
        source=Source(
            requested_url=f"https://example.com/{cc.lower()}", canonical_url=f"https://example.com/{cc.lower()}"
        ),
        tables=[
            Table(
                rows=[
                    Row(
                        cells=[
                            Cell(
                                text="2.99%",
                                tokens=[{"raw": "2.99%", "kind": "percentage", "value": "2.99"}],
                            )
                        ]
                    )
                ]
            )
        ],
        derived=DerivedFees(status="unclassified"),
    )


def test_publish_generates_expected_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        publisher = OutputPublisher(output_dir, timestamp="2025-01-01T00:00:00+00:00")
        outputs = {"DE": _make_output("DE")}
        _, staging = publisher.publish(outputs, [], [])
        assert (staging / "json" / "de.json").exists()
        assert (staging / "json" / "index.json").exists()
        assert (staging / "json" / "core-fees.json").exists()
        assert (staging / "meta" / "countries.json").exists()
        assert (staging / "meta" / "unsupported-countries.json").exists()
        assert (staging / "schemas" / "paypal-fees-v2.schema.json").exists()
        assert (staging / "schemas" / "core-fees-v2.schema.json").exists()
        assert (staging / "schemas" / "index-v2.schema.json").exists()
        assert (staging / "schemas" / "manifest-v2.schema.json").exists()


def test_commit_detects_no_change_on_second_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        publisher = OutputPublisher(output_dir, timestamp="2025-01-01T00:00:00+00:00")
        outputs = {"DE": _make_output("DE")}
        _, staging = publisher.publish(outputs, [], [])
        changed, _ = publisher.commit(staging)
        assert changed
        # Second run with identical content.
        _, staging2 = publisher.publish(outputs, [], [])
        changed2, _ = publisher.commit(staging2)
        assert not changed2


def test_content_sha256_present() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        publisher = OutputPublisher(output_dir, timestamp="2025-01-01T00:00:00+00:00")
        outputs = {"DE": _make_output("DE")}
        _, staging = publisher.publish(outputs, [], [])
        publisher.commit(staging)
        data = json.loads((output_dir / "json" / "de.json").read_text())
        assert data["source"]["artifact_sha256"]
        assert len(data["source"]["artifact_sha256"]) == 64


def test_generated_at_present() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        timestamp = "2025-01-01T00:00:00+00:00"
        publisher = OutputPublisher(output_dir, timestamp=timestamp)
        outputs = {"DE": _make_output("DE")}
        _, staging = publisher.publish(outputs, [], [])
        publisher.commit(staging)
        country = json.loads((output_dir / "json" / "de.json").read_text())
        index = json.loads((output_dir / "json" / "index.json").read_text())
        core = json.loads((output_dir / "json" / "core-fees.json").read_text())
        manifest = json.loads((output_dir / "meta" / "countries.json").read_text())
        assert country["generated_at"] == timestamp
        assert index["generated_at"] == timestamp
        assert core["generated_at"] == timestamp
        assert manifest["generated_at"] == timestamp


def test_commit_rolls_back_on_validation_failure() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        publisher = OutputPublisher(output_dir, timestamp="2025-01-01T00:00:00+00:00")
        outputs = {"DE": _make_output("DE")}
        _, staging = publisher.publish(outputs, [], [])
        changed, _ = publisher.commit(staging)
        assert changed
        first_snapshot = json.loads((output_dir / "json" / "de.json").read_text())

        # Second publish adds a new country so the commit path runs validation.
        outputs2 = {"DE": _make_output("DE"), "US": _make_output("US")}
        _, staging2 = publisher.publish(outputs2, [], [])
        call_count = 0

        def _fake_validate(path: Path) -> list[str]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return validate_output_tree(path)
            return ["Simulated live validation failure"]

        with patch.object(output_module, "validate_output_tree", _fake_validate), pytest.raises(CrawlerValidationError):
            publisher.commit(staging2)

        # Live output must be identical to the first successful publication.
        restored = json.loads((output_dir / "json" / "de.json").read_text())
        assert restored == first_snapshot
        assert not (output_dir / "json" / "us.json").exists()
        # No stale backup should be left behind.
        assert not any(str(p).endswith(".old") for p in output_dir.rglob("*"))


def test_public_output_excludes_internal_fields() -> None:
    output = CountryOutput(
        schema_version=1,
        market=Market(paypal_market_code="DE", iso_country_code="DE", country_name="Germany"),
        source=Source(
            requested_url="https://example.com/de",
            canonical_url="https://example.com/de",
            etag='"abc"',
            last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
            content_sha256="raw-content-hash",
        ),
        tables=[
            Table(
                table_id="t1",
                component_id="c1",
                source_table_ids=["s1"],
                reference_id="r1",
                parent_path=["p1"],
                rows=[
                    Row(
                        row_id="row-1",
                        cells=[
                            Cell(
                                text="2.99%",
                                tokens=[
                                    {
                                        "raw": "2.99%",
                                        "kind": "percentage",
                                        "value": "2.99",
                                        "token_id": "t1",
                                        "internal_name": "pct",
                                        "fee_data_key": "std",
                                        "content_type": "fee",
                                    }
                                ],
                            )
                        ],
                    )
                ],
            )
        ],
        derived=DerivedFees(status="unclassified"),
    )

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        publisher = OutputPublisher(output_dir, timestamp="2025-01-01T00:00:00+00:00")
        _, staging = publisher.publish({"DE": output}, [], [])
        publisher.commit(staging)

        data = json.loads((output_dir / "json" / "de.json").read_text())
        core = json.loads((output_dir / "json" / "core-fees.json").read_text())
        serialized = json.dumps({"country": data, "core": core})

        forbidden = {
            "classification_evidence",
            "unclassified_sections",
            "token_id",
            "internal_name",
            "fee_data_key",
            "content_type",
            "row_id",
            "component_id",
            "parent_path",
            "source_table_ids",
            "reference_id",
            "etag",
            "last_modified",
            "content_sha256",
        }
        assert forbidden.isdisjoint(serialized)


def test_core_fees_json_uses_public_models() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        publisher = OutputPublisher(output_dir, timestamp="2025-01-01T00:00:00+00:00")
        outputs = {"DE": _make_output("DE")}
        _, staging = publisher.publish(outputs, [], [])
        publisher.commit(staging)

        core = json.loads((output_dir / "json" / "core-fees.json").read_text())
        entry = core["countries"][0]
        assert entry["derived_status"] == entry["derived"]["status"]
        assert "country_code" in entry
        assert "classification_evidence" not in json.dumps(entry)
        assert "unclassified_sections" not in json.dumps(entry)


def test_cache_retained_after_reused_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        publisher = OutputPublisher(output_dir, timestamp="2025-01-01T00:00:00+00:00")

        first = CountryOutput(
            schema_version=1,
            market=Market(paypal_market_code="DE", iso_country_code="DE", country_name="Germany"),
            source=Source(
                requested_url="https://example.com/de",
                canonical_url="https://example.com/de",
                etag='"abc"',
                last_modified="Mon, 01 Jan 2024 00:00:00 GMT",
                content_sha256="raw-hash",
            ),
            tables=[Table(rows=[Row(cells=[Cell(text="2.99%", tokens=[{"raw": "2.99%", "kind": "percentage", "value": "2.99"}])])])],
            derived=DerivedFees(status="unclassified"),
        )
        _, staging1 = publisher.publish({"DE": first}, [], [])
        publisher.commit(staging1)

        first_cache = json.loads((output_dir / "meta" / "crawl-cache.json").read_text())
        assert first_cache["markets"]["DE"]["etag"] == '"abc"'

        # Reused output: no ETag/Last-Modified on the internal source.
        reused = CountryOutput(
            schema_version=1,
            market=Market(paypal_market_code="DE", iso_country_code="DE", country_name="Germany"),
            source=Source(
                requested_url="https://example.com/de",
                canonical_url="https://example.com/de",
                content_sha256="different-raw-hash",
            ),
            tables=[Table(rows=[Row(cells=[Cell(text="2.99%", tokens=[{"raw": "2.99%", "kind": "percentage", "value": "2.99"}])])])],
            derived=DerivedFees(status="unclassified"),
        )
        _, staging2 = publisher.publish({"DE": reused}, [], [])
        second_cache = json.loads((staging2 / "meta" / "crawl-cache.json").read_text())
        assert second_cache["markets"]["DE"]["etag"] == '"abc"'


def test_diagnostics_only_with_keep_diagnostics() -> None:
    output = _make_output("DE")
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        publisher = OutputPublisher(output_dir, timestamp="2025-01-01T00:00:00+00:00", keep_diagnostics=False)
        _, staging = publisher.publish({"DE": output}, [], [], diagnostics={"DE": {"run": "x"}})
        publisher.commit(staging)
        assert not (output_dir / "meta" / "diagnostics").exists()

        publisher2 = OutputPublisher(output_dir, timestamp="2025-01-01T00:00:00+00:00", keep_diagnostics=True)
        _, staging2 = publisher2.publish({"DE": output}, [], [], diagnostics={"DE": {"run": "x"}})
        publisher2.commit(staging2)
        assert (output_dir / "meta" / "diagnostics" / "de.json").exists()
