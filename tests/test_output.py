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
        assert data["source"]["content_sha256"]
        assert len(data["source"]["content_sha256"]) == 64


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
