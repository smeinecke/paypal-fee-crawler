"""Tests for output validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from paypal_fee_crawler.exceptions import ValidationError as CrawlerValidationError
from paypal_fee_crawler.models import Cell, CountryOutput, DerivedFeeResult, Market, Row, Source, Table
from paypal_fee_crawler.output import OutputPublisher
from paypal_fee_crawler.validation import (
    generate_core_fees_schema,
    generate_country_schema,
    generate_index_schema,
    generate_manifest_schema,
    validate_all_output,
    validate_country_output,
    validate_file,
    validate_output_tree,
    validate_public_country_output,
)


def _make_output(cc: str) -> CountryOutput:
    return CountryOutput(
        schema_version=1,
        market=Market(paypal_market_code=cc, iso_country_code=cc, country_name=cc),
        source=Source(
            requested_url=f"https://example.com/{cc.lower()}", canonical_url=f"https://example.com/{cc.lower()}"
        ),
        tables=[
            Table(
                rows=[Row(cells=[Cell(text="2.99%", tokens=[{"raw": "2.99%", "kind": "percentage", "value": "2.99"}])])]
            )
        ],
        derived=DerivedFeeResult(status="unclassified"),
    )


def test_validate_country_output_valid() -> None:
    data = _make_output("DE").model_dump(mode="json")
    errors = validate_country_output(data)
    assert not errors


def test_validate_country_output_invalid_currency() -> None:
    data = _make_output("DE").model_dump(mode="json")
    data["tables"] = [
        {
            "component_type": "FeeTable",
            "rows": [
                {
                    "cells": [
                        {
                            "text": "Fee",
                            "tokens": [{"raw": "0.99 XYZ", "kind": "money", "amount": "0.99", "currency": "XYZ"}],
                        }
                    ]
                }
            ],
        }
    ]
    errors = validate_country_output(data)
    assert any("Invalid currency" in e for e in errors)


def test_validate_country_output_no_tables() -> None:
    data = _make_output("DE").model_dump(mode="json")
    data["tables"] = []
    errors = validate_country_output(data)
    assert any("No fee tables" in e for e in errors)


def test_validate_all_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        publisher = OutputPublisher(output_dir)
        outputs = {"DE": _make_output("DE")}
        _, staging = publisher.publish(outputs, [], [])
        publisher.commit(staging)
        publisher.rollback(staging)
        errors = validate_all_output(output_dir)
        assert not errors


def test_validate_file_bad_schema_type(tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(CrawlerValidationError, match="Unknown schema type"):
        validate_file(path, "unknown")


def test_generate_country_schema_has_id() -> None:
    schema = generate_country_schema()
    assert "$id" in schema
    assert "paypal-fees-v4.schema.json" in schema["$id"]


def test_validate_output_tree_valid() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        publisher = OutputPublisher(output_dir)
        outputs = {"DE": _make_output("DE")}
        _, staging = publisher.publish(outputs, [], [])
        errors = validate_output_tree(staging)
        assert not errors


def test_validate_output_tree_missing_file() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        publisher = OutputPublisher(output_dir)
        outputs = {"DE": _make_output("DE")}
        _, staging = publisher.publish(outputs, [], [])
        (staging / "meta" / "schema-version.json").unlink()
        errors = validate_output_tree(staging)
        assert any("Missing required file" in e for e in errors)


def test_validate_output_tree_hash_mismatch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        publisher = OutputPublisher(output_dir)
        outputs = {"DE": _make_output("DE")}
        _, staging = publisher.publish(outputs, [], [])
        data = json.loads((staging / "json" / "de.json").read_text())
        data["market"]["country_name"] = "Different"
        (staging / "json" / "de.json").write_text(json.dumps(data), encoding="utf-8")
        errors = validate_output_tree(staging)
        assert any("hash" in e.lower() for e in errors)


def test_public_validation_rejects_internal_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        publisher = OutputPublisher(output_dir)
        outputs = {"DE": _make_output("DE")}
        _, staging = publisher.publish(outputs, [], [])
        data = json.loads((staging / "json" / "de.json").read_text())
        data["source"] = {"requested_url": "https://example.com/de"}

        errors = validate_public_country_output(data)
        assert any("source" in e for e in errors)


def test_public_country_schema_includes_computed_fields() -> None:
    manifest = generate_manifest_schema()
    index = generate_index_schema()
    core = generate_core_fees_schema()

    market = manifest["$defs"]["Market"]
    assert "country_code" in market["properties"]
    assert "url_slug" in market["properties"]

    entry = index["$defs"]["CountryIndexEntry"]
    assert "country_code" in entry["properties"]

    core_entry = core["$defs"]["PublicCoreFeeEntry"]
    assert "country_code" in core_entry["properties"]


def test_validate_output_tree_overlap() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        publisher = OutputPublisher(output_dir)
        outputs = {"DE": _make_output("DE")}
        _, staging = publisher.publish(outputs, [], [])
        manifest = json.loads((staging / "meta" / "countries.json").read_text())
        from paypal_fee_crawler.models import UnsupportedCountry

        manifest["unsupported"].append(UnsupportedCountry(paypal_market_code="DE").model_dump(mode="json"))
        (staging / "meta" / "countries.json").write_text(json.dumps(manifest), encoding="utf-8")
        errors = validate_output_tree(staging)
        assert any("overlap" in e.lower() for e in errors)
