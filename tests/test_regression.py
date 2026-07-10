"""Tests for regression guards."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from paypal_fee_crawler.exceptions import RegressionError
from paypal_fee_crawler.models import Cell, CountryOutput, DerivedFees, Market, Row, Source, Table
from paypal_fee_crawler.output import OutputPublisher
from paypal_fee_crawler.regression import PreviousState, RegressionLimits, check_regression, enforce_regression


def _make_output(cc: str, tables: int = 0, rows: int = 0, status: str = "complete") -> CountryOutput:
    table_rows = [Row(cells=[Cell(text="2.99%")]) for _ in range(rows)]
    return CountryOutput(
        schema_version=1,
        market=Market(country_code=cc, country_name=cc),
        source=Source(
            requested_url=f"https://example.com/{cc.lower()}", canonical_url=f"https://example.com/{cc.lower()}"
        ),
        tables=[Table(rows=table_rows) for _ in range(tables)],
        derived=DerivedFees(status=status),
    )


def test_no_regression_on_identical_run() -> None:
    previous = PreviousState(countries={"DE"}, country_tables={"DE": 0}, country_rows={"DE": 0})
    current = {"DE": _make_output("DE")}
    report = check_regression(previous, current, RegressionLimits())
    assert not report.has_regression


def test_removed_country_regression() -> None:
    previous = PreviousState(countries={"DE", "US"})
    current = {"DE": _make_output("DE")}
    report = check_regression(previous, current, RegressionLimits())
    assert report.has_regression
    assert any(c.kind == "removed_country" for c in report.changes)


def test_added_country_not_regression() -> None:
    previous = PreviousState(countries={"DE"})
    current = {"DE": _make_output("DE"), "US": _make_output("US")}
    report = check_regression(previous, current, RegressionLimits())
    assert not report.has_regression


def test_enforce_regression_raises() -> None:
    previous = PreviousState(countries={"DE"})
    current = {}
    report = check_regression(previous, current, RegressionLimits())
    with pytest.raises(RegressionError):
        enforce_regression(report, fail_on_regression=True)


def test_allow_country_drop_flag() -> None:
    previous = PreviousState(countries={"DE", "US"})
    current = {"DE": _make_output("DE")}
    limits = RegressionLimits(allow_country_drop=True)
    report = check_regression(previous, current, limits)
    # Removed country is still recorded but not a regression.
    assert not report.has_regression


def test_classified_to_unclassified_regression() -> None:
    previous = PreviousState(derived_status={"DE": "complete"})
    current = {"DE": _make_output("DE", status="unclassified")}
    report = check_regression(previous, current, RegressionLimits())
    assert report.has_regression
    assert any(c.kind == "classified_to_unclassified" for c in report.changes)


def test_lost_core_category_regression() -> None:
    previous = PreviousState(core_categories={"DE": {"standard_commercial", "commercial_fixed_fees"}})
    current = {"DE": _make_output("DE", status="partial")}
    report = check_regression(previous, current, RegressionLimits())
    assert report.has_regression
    assert any(c.kind == "lost_core_category" for c in report.changes)


def test_sharp_table_drop_regression() -> None:
    previous = PreviousState(countries={"DE"}, country_tables={"DE": 10})
    current = {"DE": _make_output("DE", tables=1, rows=1)}
    report = check_regression(previous, current, RegressionLimits())
    assert report.has_regression
    assert any(c.kind == "sharp_table_drop" for c in report.changes)


def test_previous_state_load_from_output() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        publisher = OutputPublisher(output_dir)
        outputs = {"DE": _make_output("DE", tables=1, rows=2)}
        _, staging = publisher.publish(outputs, [outputs["DE"].market], [])
        publisher.commit(staging)
        publisher.rollback(staging)
        previous = PreviousState.load(output_dir)
        assert "DE" in previous.countries
        assert previous.country_tables.get("DE") == 1
        assert previous.country_rows.get("DE") == 2
