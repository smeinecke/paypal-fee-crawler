"""Tests for output validation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from paypal_fee_crawler.exceptions import ValidationError as CrawlerValidationError
from paypal_fee_crawler.models import Cell, CountryOutput, DerivedFees, Market, Row, Source, Table
from paypal_fee_crawler.output import OutputPublisher
from paypal_fee_crawler.validation import (
    generate_country_schema,
    validate_all_output,
    validate_country_output,
    validate_file,
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
        derived=DerivedFees(status="unclassified"),
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
    assert "paypal-fees-v1.schema.json" in schema["$id"]
