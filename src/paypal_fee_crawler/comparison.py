"""Offline corpus comparison between legacy and structural classifiers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .classify import CLASSIFIER_VERSION, ClassificationRun, TableDecision, classify_legacy, classify_structural
from .models import CountryOutput, CurrencyConversion, DerivedFees, FixedFees, InternationalSurcharge, Market, Table
from .profiles import build_table_profile
from .registry import FingerprintBuilder, FingerprintRegistry
from .scoring import (
    CONVERSION_DOC_IDS,
    FIXED_DOC_IDS,
    INTERNATIONAL_DOC_IDS,
    STANDARD_DOC_IDS,
    FeeCategory,
)


def _fixed_fees_set(fees: list[FixedFees]) -> set[tuple[str, str]]:
    return {(f.currency, f.amount) for f in fees}


def _surcharges_set(surcharges: list[InternationalSurcharge]) -> dict[str, str | None]:
    return {s.region: s.percentage_points for s in surcharges}


def _conversion_spread(conversion: CurrencyConversion | None) -> str | None:
    return conversion.spread_percentage if conversion else None


def _selected_categories_from_derived(derived: DerivedFees) -> set[str]:
    """Return selected output categories from a derived fees object."""
    categories: set[str] = set()
    if derived.standard_commercial is not None:
        categories.add(FeeCategory.STANDARD_COMMERCIAL.value)
    if derived.commercial_fixed_fees:
        categories.add(FeeCategory.FIXED_FEE.value)
    if derived.international_surcharges:
        categories.add(FeeCategory.INTERNATIONAL_SURCHARGE.value)
    if derived.currency_conversion is not None:
        categories.add(FeeCategory.CURRENCY_CONVERSION.value)
    return categories


def _selected_categories_from_decisions(run: ClassificationRun) -> set[str]:
    """Return selected core categories from a classification run's table decisions."""
    return {
        d.selected_category.value
        for d in run.table_decisions
        if d.selected_category is not None and d.selected_category.value != FeeCategory.OTHER.value
    }


def _gold_category_for_table(table: Table, derived: DerivedFees, table_count: int) -> FeeCategory | None:
    """Infer the reviewed category for a table in a gold fixture.

    Document IDs are authoritative when they match a known category.  For single-
    table fixtures with no known document ID, the derived output is used.
    """
    doc_id = (table.document_id or "").upper()
    if doc_id in STANDARD_DOC_IDS:
        return FeeCategory.STANDARD_COMMERCIAL
    if doc_id in FIXED_DOC_IDS:
        return FeeCategory.FIXED_FEE
    if doc_id in INTERNATIONAL_DOC_IDS:
        return FeeCategory.INTERNATIONAL_SURCHARGE
    if doc_id in CONVERSION_DOC_IDS:
        return FeeCategory.CURRENCY_CONVERSION

    if table_count == 1:
        if derived.standard_commercial is not None:
            return FeeCategory.STANDARD_COMMERCIAL
        if derived.commercial_fixed_fees:
            return FeeCategory.FIXED_FEE
        if derived.international_surcharges:
            return FeeCategory.INTERNATIONAL_SURCHARGE
        if derived.currency_conversion is not None:
            return FeeCategory.CURRENCY_CONVERSION

    return None


def _gold_table_decisions(country: CountryOutput) -> tuple[TableDecision, ...]:
    """Build reviewed table decisions from a gold CountryOutput fixture."""
    derived = country.derived
    table_count = len(country.tables)
    decisions: list[TableDecision] = []
    for table in country.tables:
        profile = build_table_profile(table)
        fingerprint = str(FingerprintBuilder.build(profile, table))
        selected_category = _gold_category_for_table(table, derived, table_count)
        decisions.append(
            TableDecision(
                table_id=table.table_id,
                document_id=table.document_id,
                component_id=table.component_id,
                fingerprint=fingerprint,
                selected_category=selected_category,
                selected_score=None,
                status="selected" if selected_category is not None else "unclassified",
                ambiguity_reason=None,
                winner_margin=None,
                ranked_scores=(),
                blockers=(),
                evidence_codes=(),
                evidence_sources=(),
            )
        )
    return tuple(decisions)


@dataclass(frozen=True)
class ValueChange:
    field: str
    legacy: str | None
    structural: str | None
    kind: str


@dataclass(frozen=True)
class TableDecisionChange:
    table_id: str | None
    document_id: str | None
    component_id: str | None
    fingerprint: str | None
    legacy_category: str | None
    structural_category: str | None
    legacy_score: int | None
    structural_score: int | None
    legacy_blockers: tuple[str, ...]
    structural_blockers: tuple[str, ...]
    legacy_evidence: tuple[str, ...]
    structural_evidence: tuple[str, ...]
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
    table_decision_changes: tuple[TableDecisionChange, ...]
    observation_count: int
    structural_observations: tuple[dict[str, str | None], ...]
    legacy_classifier_version: str
    structural_classifier_version: str


def _table_decision_key(d: TableDecision) -> str:
    """Stable priority-based identity for a table decision.

    Prefer explicit table/component identifiers over the content-derived
    fingerprint, so small structural changes are reported as a changed decision
    rather than a missing/new pair.
    """
    if d.table_id:
        return f"table:{d.table_id}"
    if d.component_id:
        return f"component:{d.component_id}"
    if d.document_id:
        return f"document:{d.document_id}"
    if d.fingerprint:
        return f"fingerprint:{d.fingerprint}"
    return "unknown"


def _compare_table_decisions(
    legacy: tuple[TableDecision, ...],
    structural: tuple[TableDecision, ...],
) -> tuple[TableDecisionChange, ...]:
    """Compare per-table classification decisions between legacy and structural."""
    legacy_by_key: dict[str, TableDecision] = {}
    structural_by_key: dict[str, TableDecision] = {}
    for d in legacy:
        legacy_by_key[_table_decision_key(d)] = d
    for d in structural:
        structural_by_key[_table_decision_key(d)] = d

    changes: list[TableDecisionChange] = []
    all_keys = set(legacy_by_key) | set(structural_by_key)
    for key in sorted(all_keys):
        legacy_decision = legacy_by_key.get(key)
        structural_decision = structural_by_key.get(key)
        if legacy_decision is None or structural_decision is None:
            present = structural_decision if legacy_decision is None else legacy_decision
            if present is None or present.selected_category is None:
                continue
            kind = "new" if legacy_decision is None else "missing"
            changes.append(
                TableDecisionChange(
                    table_id=present.table_id,
                    document_id=present.document_id,
                    component_id=present.component_id,
                    fingerprint=present.fingerprint,
                    legacy_category=legacy_decision.selected_category.value
                    if legacy_decision and legacy_decision.selected_category
                    else None,
                    structural_category=structural_decision.selected_category.value
                    if structural_decision and structural_decision.selected_category
                    else None,
                    legacy_score=legacy_decision.selected_score if legacy_decision else None,
                    structural_score=structural_decision.selected_score if structural_decision else None,
                    legacy_blockers=tuple(sorted({b.value for b in legacy_decision.blockers}))
                    if legacy_decision
                    else (),
                    structural_blockers=tuple(sorted({b.value for b in structural_decision.blockers}))
                    if structural_decision
                    else (),
                    legacy_evidence=tuple(sorted(legacy_decision.evidence_codes)) if legacy_decision else (),
                    structural_evidence=tuple(sorted(structural_decision.evidence_codes))
                    if structural_decision
                    else (),
                    kind=kind,
                )
            )
            continue

        legacy_cat = legacy_decision.selected_category.value if legacy_decision.selected_category else None
        structural_cat = structural_decision.selected_category.value if structural_decision.selected_category else None
        if legacy_cat == structural_cat:
            continue
        kind = "new" if legacy_cat is None else "missing" if structural_cat is None else "changed"
        changes.append(
            TableDecisionChange(
                table_id=legacy_decision.table_id,
                document_id=legacy_decision.document_id,
                component_id=legacy_decision.component_id,
                fingerprint=legacy_decision.fingerprint,
                legacy_category=legacy_cat,
                structural_category=structural_cat,
                legacy_score=legacy_decision.selected_score,
                structural_score=structural_decision.selected_score,
                legacy_blockers=tuple(sorted({b.value for b in legacy_decision.blockers})),
                structural_blockers=tuple(sorted({b.value for b in structural_decision.blockers})),
                legacy_evidence=tuple(sorted(legacy_decision.evidence_codes)),
                structural_evidence=tuple(sorted(structural_decision.evidence_codes)),
                kind=kind,
            )
        )
    return tuple(changes)


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
    table_decision_changes = _compare_table_decisions(legacy_run.table_decisions, structural_run.table_decisions)

    legacy_pct = legacy.standard_commercial.percentage if legacy.standard_commercial else None
    structural_pct = structural.standard_commercial.percentage if structural.standard_commercial else None
    if legacy_pct != structural_pct:
        if legacy_pct is None and structural_pct is not None:
            kind = "new"
        elif legacy_pct is not None and structural_pct is None:
            kind = "missing"
        else:
            kind = "changed"
        value_changes.append(ValueChange("standard_commercial.percentage", legacy_pct, structural_pct, kind))

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
            value_changes.append(ValueChange(f"fixed_fee.{currency}", legacy_amount, structural_amount, kind))

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
            value_changes.append(ValueChange(f"international_surcharge.{region}", legacy_pct, structural_pct, kind))

    legacy_conv = _conversion_spread(legacy.currency_conversion)
    structural_conv = _conversion_spread(structural.currency_conversion)
    if legacy_conv != structural_conv:
        if legacy_conv is None and structural_conv is not None:
            kind = "new"
        elif legacy_conv is not None and structural_conv is None:
            kind = "missing"
        else:
            kind = "changed"
        value_changes.append(ValueChange("currency_conversion.spread_percentage", legacy_conv, structural_conv, kind))

    legacy_cats = _selected_categories_from_decisions(legacy_run)
    structural_cats = _selected_categories_from_decisions(structural_run)

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
        table_decision_changes=table_decision_changes,
        observation_count=len(structural_run.observations),
        structural_observations=observations,
        legacy_classifier_version=legacy_run.classifier_version,
        structural_classifier_version=structural_run.classifier_version,
    )


def compare_country(country: CountryOutput) -> CountryComparison:
    """Run both classifiers on a stored country output and compare results."""
    market = country.market
    legacy_run = classify_legacy(country.tables, market_code=market.paypal_market_code, locale=market.locale)
    structural_run = classify_structural(
        country.tables,
        market_code=market.paypal_market_code,
        locale=market.locale,
        registry=FingerprintRegistry.load_builtin(),
    )
    return compare_runs(legacy_run, structural_run, market)


@dataclass(frozen=True)
class ComparisonSummary:
    total_countries: int
    status_changed: int
    categories_changed: int
    value_changes: int
    table_decision_changes: int
    total_observations: int
    countries_with_observations: int
    countries_with_value_changes: int


def _compare_summary(comparisons: list[CountryComparison]) -> ComparisonSummary:
    status_changed = sum(1 for c in comparisons if not c.status_match)
    categories_changed = sum(1 for c in comparisons if not c.selected_categories_match)
    value_changes = sum(len(c.value_changes) for c in comparisons)
    table_decision_changes = sum(len(c.table_decision_changes) for c in comparisons)
    total_observations = sum(c.observation_count for c in comparisons)
    countries_with_observations = sum(1 for c in comparisons if c.observation_count)
    countries_with_value_changes = sum(1 for c in comparisons if c.value_changes)
    return ComparisonSummary(
        total_countries=len(comparisons),
        status_changed=status_changed,
        categories_changed=categories_changed,
        value_changes=value_changes,
        table_decision_changes=table_decision_changes,
        total_observations=total_observations,
        countries_with_observations=countries_with_observations,
        countries_with_value_changes=countries_with_value_changes,
    )


@dataclass(frozen=True)
class ComparisonReport:
    """Aggregate comparison report for a corpus of country outputs."""

    json_path: Path
    summary: ComparisonSummary
    comparisons: tuple[CountryComparison, ...]


def compare_against_gold(gold_dir: Path, output_dir: Path, countries: set[str] | None = None) -> ComparisonReport:
    """Compare structural classifier output against reviewed gold expectations.

    Loads each CountryOutput in GOLD_DIR, treats the stored ``derived`` field as
    the authoritative reviewed expectation, and runs the structural classifier
    on the stored tables. The returned report surfaces status, category, and
    value differences against the gold.
    """
    comparisons: list[CountryComparison] = []
    paths = sorted(gold_dir.glob("*.json"))
    for path in paths:
        if countries and path.stem.upper() not in {c.upper() for c in countries}:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            country = CountryOutput(**data)
        except Exception as exc:
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
                    table_decision_changes=(),
                    observation_count=0,
                    structural_observations=(),
                    legacy_classifier_version="gold",
                    structural_classifier_version=CLASSIFIER_VERSION,
                )
            )
            continue

        market = country.market
        gold_run = ClassificationRun(
            derived=country.derived,
            table_decisions=_gold_table_decisions(country),
            observations=(),
            classifier_version="gold",
        )
        structural_run = classify_structural(
            country.tables,
            market_code=market.paypal_market_code,
            locale=market.locale,
            registry=FingerprintRegistry.load_builtin(),
        )
        comparisons.append(compare_runs(gold_run, structural_run, market))

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

    return ComparisonReport(
        json_path=json_path,
        summary=summary,
        comparisons=tuple(comparisons),
    )


def compare_classifiers(diagnostics_dir: Path, output_dir: Path, countries: set[str] | None = None) -> ComparisonReport:
    """Compare legacy and structural classifiers across a corpus and write reports.

    Loads each diagnostic sidecar from ``diagnostics_dir``, extracts the
    internal ``normalized_output`` (which contains the full table structure),
    and re-classifies it with both engines.
    """
    comparisons: list[CountryComparison] = []
    paths = sorted(diagnostics_dir.glob("*.json"))
    for path in paths:
        if countries and path.stem.upper() not in {c.upper() for c in countries}:
            continue
        try:
            wrapper = json.loads(path.read_text(encoding="utf-8"))
            data = wrapper.get("normalized_output") or wrapper
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
                    table_decision_changes=(),
                    observation_count=0,
                    structural_observations=(),
                    legacy_classifier_version="legacy",
                    structural_classifier_version=CLASSIFIER_VERSION,
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

    return ComparisonReport(
        json_path=json_path,
        summary=summary,
        comparisons=tuple(comparisons),
    )


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
                lines.append(
                    f"- `{change.field}`: legacy={change.legacy!r} structural={change.structural!r} ({change.kind})"
                )
        lines.append("")

    return "\n".join(lines)
