"""Tests for deterministic output generation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from paypal_fee_crawler.models import CountryOutput, DerivedFees, Market, Source
from paypal_fee_crawler.output import OutputPublisher


def _make_output(cc: str) -> CountryOutput:
    return CountryOutput(
        schema_version=1,
        market=Market(country_code=cc, country_name=cc),
        source=Source(
            requested_url=f"https://example.com/{cc.lower()}", canonical_url=f"https://example.com/{cc.lower()}"
        ),
        derived=DerivedFees(status="unclassified"),
    )


def test_publish_generates_expected_files() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        publisher = OutputPublisher(output_dir)
        outputs = {"DE": _make_output("DE")}
        _, staging = publisher.publish(outputs, [], [])
        assert (staging / "json" / "de.json").exists()
        assert (staging / "json" / "index.json").exists()
        assert (staging / "json" / "core-fees.json").exists()
        assert (staging / "meta" / "countries.json").exists()
        assert (staging / "meta" / "unsupported-countries.json").exists()
        assert (staging / "schemas" / "paypal-fees-v1.schema.json").exists()


def test_commit_detects_no_change_on_second_run() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        publisher = OutputPublisher(output_dir)
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
        publisher = OutputPublisher(output_dir)
        outputs = {"DE": _make_output("DE")}
        _, staging = publisher.publish(outputs, [], [])
        publisher.commit(staging)
        data = json.loads((output_dir / "json" / "de.json").read_text())
        assert data["source"]["content_sha256"]
        assert len(data["source"]["content_sha256"]) == 64
