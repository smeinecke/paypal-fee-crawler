"""Tests for regression guards."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from paypal_fee_crawler.exceptions import RegressionError
from paypal_fee_crawler.models import Cell, CountryOutput, DerivedFees, Market, Row, Source, Table
from paypal_fee_crawler.output import OutputPublisher
from paypal_fee_crawler.regression import PreviousState, RegressionLimits, check_regression, enforce_regression


def _make_output(
    cc: str, tables: int = 0, rows: int = 0, status: str = "unclassified", intl: bool = False, conversion: bool = False
) -> CountryOutput:
    table_rows = [
        Row(
            cells=[
                Cell(
                    text="2.99%",
                    tokens=[{"raw": "2.99%", "kind": "percentage", "value": "2.99"}],
                )
            ]
        )
        for _ in range(rows)
    ]
    extra_tables: list[Table] = []
    if intl:
        extra_tables.append(
            Table(
                caption="International surcharge",
                rows=[Row(cells=[Cell(text="1.29%")])],
            )
        )
    if conversion:
        extra_tables.append(
            Table(
                caption="Currency conversion",
                rows=[Row(cells=[Cell(text="3.0%")])],
            )
        )
    return CountryOutput(
        schema_version=1,
        market=Market(paypal_market_code=cc, iso_country_code=cc, country_name=cc),
        source=Source(
            requested_url=f"https://example.com/{cc.lower()}", canonical_url=f"https://example.com/{cc.lower()}"
        ),
        tables=[Table(rows=table_rows) for _ in range(tables)] + extra_tables,
        derived=DerivedFees(status=status),
    )


def _check(
    previous: PreviousState,
    current_outputs: dict[str, CountryOutput],
    discovered: set[str] | None = None,
    unsupported: set[str] | None = None,
    transient: set[str] | None = None,
    limits: RegressionLimits | None = None,
) -> Any:
    discovered = discovered or set(current_outputs.keys())
    supported = set(current_outputs.keys())
    unsupported = unsupported or set()
    transient = transient or set()
    return check_regression(
        previous, discovered, supported, unsupported, transient, current_outputs, limits or RegressionLimits()
    )


def test_no_regression_on_identical_run() -> None:
    previous = PreviousState(supported_countries={"DE"}, country_tables={"DE": 1}, country_rows={"DE": 1})
    current = {"DE": _make_output("DE", tables=1, rows=1)}
    report = _check(previous, current)
    assert not report.has_regression


def test_no_regression_when_discovered_remains_known() -> None:
    """Discovered markets are compared to discovered markets, not to supported markets."""
    previous = PreviousState(
        discovered_countries={"DE", "US", "XY"},
        supported_countries={"DE"},
    )
    current = {"DE": _make_output("DE", tables=1, rows=1)}
    report = _check(previous, current, discovered={"DE", "US", "XY"})
    assert not report.has_regression


def test_removed_country_regression() -> None:
    previous = PreviousState(supported_countries={"DE", "US"})
    current = {"DE": _make_output("DE", tables=1, rows=1)}
    report = _check(previous, current)
    assert report.has_regression
    assert any(c.kind == "removed_country" for c in report.changes)


def test_discovered_to_missing_regression() -> None:
    previous = PreviousState(discovered_countries={"DE", "US"})
    current = {"DE": _make_output("DE", tables=1, rows=1)}
    report = _check(previous, current, discovered={"DE"})
    assert report.has_regression
    assert any(c.kind == "discovered_to_missing" for c in report.changes)


def test_added_country_not_regression() -> None:
    previous = PreviousState(supported_countries={"DE"})
    current = {"DE": _make_output("DE", tables=1, rows=1), "US": _make_output("US", tables=1, rows=1)}
    report = _check(previous, current)
    assert not report.has_regression


def test_enforce_regression_raises() -> None:
    previous = PreviousState(supported_countries={"DE"})
    current: dict[str, CountryOutput] = {}
    report = _check(previous, current)
    with pytest.raises(RegressionError):
        enforce_regression(report, fail_on_regression=True)


def test_allow_country_drop_flag() -> None:
    previous = PreviousState(supported_countries={"DE", "US"})
    current = {"DE": _make_output("DE", tables=1, rows=1)}
    limits = RegressionLimits(allow_country_drop=True)
    report = _check(previous, current, limits=limits)
    # Removed country is still recorded but not a regression.
    assert not report.has_regression


def test_supported_to_transient_regression() -> None:
    previous = PreviousState(supported_countries={"DE"})
    current: dict[str, CountryOutput] = {}
    report = _check(previous, current, transient={"DE"})
    assert report.has_regression
    assert any(c.kind == "supported_to_transient" for c in report.changes)


def test_supported_to_unsupported_regression() -> None:
    previous = PreviousState(supported_countries={"DE"})
    current: dict[str, CountryOutput] = {}
    report = _check(previous, current, unsupported={"DE"})
    assert report.has_regression
    assert any(c.kind == "supported_to_unsupported" for c in report.changes)


def test_unsupported_to_supported_not_regression() -> None:
    previous = PreviousState(supported_countries=set(), unsupported_countries={"US"})
    current = {"US": _make_output("US", tables=1, rows=1)}
    report = _check(previous, current)
    assert not report.has_regression
    assert any(c.kind == "unsupported_to_supported" for c in report.changes)


def test_structural_overlap_regression() -> None:
    previous = PreviousState()
    current = {"DE": _make_output("DE", tables=1, rows=1)}
    report = _check(previous, current, unsupported={"DE"})
    assert report.has_regression
    assert any(c.kind == "structural_regression" for c in report.changes)


def test_classified_to_unclassified_regression() -> None:
    previous = PreviousState(derived_status={"DE": "complete"})
    current = {"DE": _make_output("DE", tables=1, rows=1, status="unclassified")}
    report = _check(previous, current)
    assert report.has_regression
    assert any(c.kind == "classified_to_unclassified" for c in report.changes)


def test_lost_core_category_regression() -> None:
    previous = PreviousState(core_categories={"DE": {"standard_commercial", "commercial_fixed_fees"}})
    current = {"DE": _make_output("DE", tables=1, rows=1, status="partial")}
    report = _check(previous, current)
    assert report.has_regression
    assert any(c.kind == "lost_core_category" for c in report.changes)


def test_sharp_table_drop_regression() -> None:
    previous = PreviousState(supported_countries={"DE"}, country_tables={"DE": 10})
    current = {"DE": _make_output("DE", tables=1, rows=1)}
    report = _check(previous, current)
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
        assert "DE" in previous.supported_countries
        assert previous.country_tables.get("DE") == 1
        assert previous.country_rows.get("DE") == 2


def test_previous_state_loads_legacy_manifest_with_country_code() -> None:
    """Old manifests stored country_code instead of paypal_market_code."""
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "out"
        meta_dir = output_dir / "meta"
        meta_dir.mkdir(parents=True)
        legacy_manifest = {
            "schema_version": 1,
            "generated_at": "2026-04-30",
            "markets": [
                {
                    "country_code": "DE",
                    "country_name": "Germany",
                    "url_prefix": "https://www.paypal.com/de",
                },
                {
                    "country_code": "US",
                    "country_name": "United States",
                    "url_prefix": "https://www.paypal.com/us",
                },
            ],
            "unsupported": [
                {
                    "country_code": "XY",
                    "country_name": "Unknown",
                    "temporary": True,
                }
            ],
            "fee_page_urls": {},
        }
        (meta_dir / "countries.json").write_text(json.dumps(legacy_manifest), encoding="utf-8")
        previous = PreviousState.load(output_dir)
        assert previous.discovered_countries == {"DE", "US"}
        assert previous.unsupported_countries == {"XY"}
        assert previous.transient_countries == {"XY"}
