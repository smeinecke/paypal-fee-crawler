"""Offline corpus comparison between legacy and structural classifiers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .classify import ClassificationRun, classify_legacy, classify_structural
from .models import CountryOutput, CurrencyConversion, DerivedFees, FixedFees, InternationalSurcharge, Market
from .scoring import FeeCategory


def _fixed_fees_set(fees: list[FixedFees]) -> set[tuple[str, str]]:
    return {(f.currency, f.amount) for f in fees}


def _surcharges_set(surcharges: list[InternationalSurcharge]) -> dict[str, str | None]:
    return {s.region: s.percentage_points for s in surcharges}


def _conversion_spread(conversion: CurrencyConversion | None) -> str | None:
    return conversion.spread_percentage if conversion else None


def _selected_categories(derived: DerivedFees) -> set[str]:
    categories: set[str] = set()
    if derived.standard_commercial is not None or derived.commercial_fixed_fees:
        categories.add(FeeCategory.STANDARD_COMMERCIAL.value)
    if derived.commercial_fixed_fees:
        categories.add(FeeCategory.FIXED_FEE.value)
    if derived.international_surcharges or derived.international_surcharge_exposed:
        categories.add(FeeCategory.INTERNATIONAL_SURCHARGE.value)
    if derived.currency_conversion is not None or derived.currency_conversion_exposed:
        categories.add(FeeCategory.CURRENCY_CONVERSION.value)
    return categories


@dataclass(frozen=True)
class ValueChange:
    field: str
    legacy: str | None
    structural: str | None
    kind: str


@dataclass(frozen=True)
class CountryComparison:
    country_code: str
    market_code: str
    locale: str | None
    status_match: bool
    legacy_status: str
    structural_status: str
    selected_categories_match: bool
    legacy_selected_categories: tuple[str, ...]
    structural_selected_categories: tuple[str, ...]
    value_changes: tuple[ValueChange, ...]
    observation_count: int
    structural_observations: tuple[dict[str, str | None], ...]
    legacy_classifier_version: str
    structural_classifier_version: str


def compare_runs(
    legacy_run: ClassificationRun,
    structural_run: ClassificationRun,
    market: Market,
) -> CountryComparison:
    """Compare a legacy and a structural classification run for a single market."""
    market_code = market.paypal_market_code
    locale = market.locale

    legacy = legacy_run.derived
    structural = structural_run.derived

    value_changes: list[ValueChange] = []

    legacy_pct = legacy.standard_commercial.percentage if legacy.standard_commercial else None
    structural_pct = structural.standard_commercial.percentage if structural.standard_commercial else None
    if legacy_pct != structural_pct:
        if legacy_pct is None and structural_pct is not None:
            kind = "new"
        elif legacy_pct is not None and structural_pct is None:
            kind = "missing"
        else:
            kind = "changed"
        value_changes.append(
            ValueChange("standard_commercial.percentage", legacy_pct, structural_pct, kind)
        )

    legacy_fixed = _fixed_fees_set(legacy.commercial_fixed_fees)
    structural_fixed = _fixed_fees_set(structural.commercial_fixed_fees)
    all_currencies = {c for c, _ in legacy_fixed | structural_fixed}
    for currency in sorted(all_currencies):
        legacy_amount = next((a for c, a in legacy_fixed if c == currency), None)
        structural_amount = next((a for c, a in structural_fixed if c == currency), None)
        if legacy_amount != structural_amount:
            if legacy_amount is None:
                kind = "new"
            elif structural_amount is None:
                kind = "missing"
            else:
                kind = "conflict"
            value_changes.append(
                ValueChange(f"fixed_fee.{currency}", legacy_amount, structural_amount, kind)
            )

    legacy_surcharges = _surcharges_set(legacy.international_surcharges)
    structural_surcharges = _surcharges_set(structural.international_surcharges)
    all_regions = set(legacy_surcharges) | set(structural_surcharges)
    for region in sorted(all_regions):
        legacy_pct = legacy_surcharges.get(region)
        structural_pct = structural_surcharges.get(region)
        if legacy_pct != structural_pct:
            if legacy_pct is None:
                kind = "new"
            elif structural_pct is None:
                kind = "missing"
            else:
                kind = "conflict"
            value_changes.append(
                ValueChange(f"international_surcharge.{region}", legacy_pct, structural_pct, kind)
            )

    legacy_conv = _conversion_spread(legacy.currency_conversion)
    structural_conv = _conversion_spread(structural.currency_conversion)
    if legacy_conv != structural_conv:
        if legacy_conv is None and structural_conv is not None:
            kind = "new"
        elif legacy_conv is not None and structural_conv is None:
            kind = "missing"
        else:
            kind = "changed"
        value_changes.append(
            ValueChange("currency_conversion.spread_percentage", legacy_conv, structural_conv, kind)
        )

    legacy_cats = _selected_categories(legacy)
    structural_cats = _selected_categories(structural)

    observations = tuple(
        {
            "kind": str(o.kind),
            "category": o.category.value if o.category else None,
            "table_id": o.table_id,
            "message": o.message,
        }
        for o in structural_run.observations
    )

    return CountryComparison(
        country_code=market.iso_country_code or market_code,
        market_code=market_code,
        locale=locale,
        status_match=legacy.status == structural.status,
        legacy_status=legacy.status,
        structural_status=structural.status,
        selected_categories_match=legacy_cats == structural_cats,
        legacy_selected_categories=tuple(sorted(legacy_cats)),
        structural_selected_categories=tuple(sorted(structural_cats)),
        value_changes=tuple(value_changes),
        observation_count=len(structural_run.observations),
        structural_observations=observations,
        legacy_classifier_version=legacy_run.classifier_version,
        structural_classifier_version=structural_run.classifier_version,
    )


def compare_country(country: CountryOutput) -> CountryComparison:
    """Run both classifiers on a stored country output and compare results."""
    market = country.market
    legacy_run = classify_legacy(country.tables, market_code=market.paypal_market_code, locale=market.locale)
    structural_run = classify_structural(country.tables, market_code=market.paypal_market_code, locale=market.locale)
    return compare_runs(legacy_run, structural_run, market)


@dataclass(frozen=True)
class ComparisonSummary:
    total_countries: int
    status_changed: int
    categories_changed: int
    value_changes: int
    total_observations: int
    countries_with_observations: int
    countries_with_value_changes: int


def _compare_summary(comparisons: list[CountryComparison]) -> ComparisonSummary:
    status_changed = sum(1 for c in comparisons if not c.status_match)
    categories_changed = sum(1 for c in comparisons if not c.selected_categories_match)
    value_changes = sum(len(c.value_changes) for c in comparisons)
    total_observations = sum(c.observation_count for c in comparisons)
    countries_with_observations = sum(1 for c in comparisons if c.observation_count)
    countries_with_value_changes = sum(1 for c in comparisons if c.value_changes)
    return ComparisonSummary(
        total_countries=len(comparisons),
        status_changed=status_changed,
        categories_changed=categories_changed,
        value_changes=value_changes,
        total_observations=total_observations,
        countries_with_observations=countries_with_observations,
        countries_with_value_changes=countries_with_value_changes,
    )


def compare_classifiers(json_dir: Path, output_dir: Path, countries: set[str] | None = None) -> Path:
    """Compare legacy and structural classifiers across a corpus and write reports."""
    comparisons: list[CountryComparison] = []
    paths = sorted(json_dir.glob("*.json"))
    for path in paths:
        if countries and path.stem.upper() not in {c.upper() for c in countries}:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            country = CountryOutput(**data)
        except Exception as exc:
            # Skip malformed files and report them in the output.
            comparisons.append(
                CountryComparison(
                    country_code=path.stem.upper(),
                    market_code=path.stem.upper(),
                    locale=None,
                    status_match=False,
                    legacy_status="failed",
                    structural_status="failed",
                    selected_categories_match=False,
                    legacy_selected_categories=(),
                    structural_selected_categories=(),
                    value_changes=(ValueChange("load", None, str(exc), "error"),),
                    observation_count=0,
                    structural_observations=(),
                    legacy_classifier_version="legacy",
                    structural_classifier_version="structural-1",
                )
            )
            continue
        comparisons.append(compare_country(country))

    summary = _compare_summary(comparisons)
    report = {
        "summary": asdict(summary),
        "countries": [asdict(c) for c in comparisons],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "classification-comparison.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    md_path = output_dir / "classification-comparison.md"
    md_path.write_text(_render_markdown(summary, comparisons), encoding="utf-8")

    return json_path


def _render_markdown(summary: ComparisonSummary, comparisons: list[CountryComparison]) -> str:
    lines = [
        "# Classifier comparison report",
        "",
        "## Summary",
        "",
        f"- Total countries: {summary.total_countries}",
        f"- Status changed: {summary.status_changed}",
        f"- Selected categories changed: {summary.categories_changed}",
        f"- Value differences: {summary.value_changes}",
        f"- Total structural observations: {summary.total_observations}",
        f"- Countries with observations: {summary.countries_with_observations}",
        f"- Countries with value changes: {summary.countries_with_value_changes}",
        "",
        "## Per-country results",
        "",
        "| Country | Legacy status | Structural status | Categories match | Value changes | Observations |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for c in comparisons:
        changes = "load error" if c.value_changes and c.value_changes[0].field == "load" else str(len(c.value_changes))
        lines.append(
            f"| {c.country_code} | {c.legacy_status} | {c.structural_status} | "
            f"{c.selected_categories_match} | {changes} | {c.observation_count} |"
        )

    lines.append("")
    lines.append("## Value changes")
    lines.append("")
    for c in comparisons:
        if not c.value_changes:
            continue
        lines.append(f"### {c.country_code}")
        if c.value_changes and c.value_changes[0].field == "load":
            lines.append(f"- Load error: {c.value_changes[0].structural}")
        else:
            for change in c.value_changes:
                lines.append(f"- `{change.field}`: legacy={change.legacy!r} structural={change.structural!r} ({change.kind})")
        lines.append("")

    return "\n".join(lines)
