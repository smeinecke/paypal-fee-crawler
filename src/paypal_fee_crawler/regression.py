"""Regression guards and change reporting."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .derived_categories import _selected_categories_from_derived
from .exceptions import RegressionError
from .hashing import _country_output_hash  # noqa: F401  re-exported for backwards compatibility
from .models import (
    _CHANGE_SEVERITY_BY_KIND,
    AcceptedRegressions,
    ChangeReport,
    ChangeSeverity,
    ChangeType,
    ClassifierMetadata,
    CountryIndex,
    CountryManifest,
    CountryOutput,
    CrawlState,
    UnsupportedCountry,
)

logger = logging.getLogger(__name__)


@dataclass
class RegressionLimits:
    max_table_count_delta_ratio: float = 0.5
    max_row_count_delta_ratio: float = 0.5
    max_country_count_delta_ratio: float = 0.1
    allow_country_drop: bool = False


def _change_type(kind: str, **kwargs: Any) -> ChangeType:
    """Build a ChangeType whose severity is derived from its kind."""
    return ChangeType(kind=kind, **kwargs)


@dataclass
class PreviousState:
    """Baseline state loaded from a previous published output directory."""

    discovered_countries: set[str] = field(default_factory=set)
    supported_countries: set[str] = field(default_factory=set)
    unsupported_countries: set[str] = field(default_factory=set)
    transient_countries: set[str] = field(default_factory=set)
    country_tables: dict[str, int] = field(default_factory=dict)
    country_rows: dict[str, int] = field(default_factory=dict)
    core_categories: dict[str, set[str]] = field(default_factory=dict)
    derived_status: dict[str, str] = field(default_factory=dict)
    unsupported_records: dict[str, UnsupportedCountry] = field(default_factory=dict)
    accepted_regressions: AcceptedRegressions = field(default_factory=AcceptedRegressions)
    classifier_metadata: ClassifierMetadata | None = None
    # Backward-compatible alias for code that expects a single set.
    countries: set[str] = field(default_factory=set)

    @classmethod
    def load(cls, output_dir: Path | str) -> PreviousState:
        output_dir = Path(output_dir)
        state = cls()
        _load_manifest(state, output_dir / "meta" / "countries.json")
        _load_unsupported(state, output_dir / "meta" / "unsupported-countries.json")
        _load_accepted_regressions(state, output_dir / "meta" / "accepted-regressions.json")
        _load_classifier(state, output_dir / "meta" / "classifier-version.json")
        _load_index(state, output_dir / "json" / "index.json")
        _load_crawl_state(state, output_dir / "meta" / "crawl-state.json")
        # Backward-compatible alias includes all discovered markets.
        state.countries = state.discovered_countries
        return state


def _safe_load[T](
    path: Path,
    loader: Callable[[str], T],
    description: str,
) -> T | None:
    """Load and parse a file, returning None if it is missing or fails."""
    if not path.exists():
        return None
    try:
        return loader(path.read_text(encoding="utf-8"))
    except Exception as exc:  # nosec B112 # noqa: S112
        logger.warning("Could not load previous %s: %s", description, exc)
        return None


def _load_manifest(state: PreviousState, manifest_path: Path) -> None:
    manifest = _safe_load(manifest_path, CountryManifest.model_validate_json, "country manifest")
    if manifest is None:
        return
    state.discovered_countries = {m.paypal_market_code for m in manifest.markets}
    state.unsupported_countries = {u.paypal_market_code for u in manifest.unsupported}
    state.transient_countries = {u.paypal_market_code for u in manifest.unsupported if u.temporary}
    state.unsupported_records = {u.paypal_market_code: u for u in manifest.unsupported}


def _load_unsupported(state: PreviousState, unsupported_path: Path) -> None:
    data = _safe_load(
        unsupported_path,
        lambda text: json.loads(text),
        "unsupported metadata",
    )
    if data is None:
        return
    for item in data.get("unsupported", []):
        u = UnsupportedCountry.model_validate(item)
        state.unsupported_records[u.paypal_market_code] = u
        state.unsupported_countries.add(u.paypal_market_code)
        if u.temporary:
            state.transient_countries.add(u.paypal_market_code)


def _load_accepted_regressions(state: PreviousState, accepted_path: Path) -> None:
    accepted = _safe_load(accepted_path, AcceptedRegressions.model_validate_json, "accepted regressions")
    if accepted is not None:
        state.accepted_regressions = accepted


def _load_classifier(state: PreviousState, classifier_path: Path) -> None:
    metadata = _safe_load(classifier_path, ClassifierMetadata.model_validate_json, "classifier metadata")
    if metadata is not None:
        state.classifier_metadata = metadata


def _load_index(state: PreviousState, index_path: Path) -> None:
    index = _safe_load(index_path, CountryIndex.model_validate_json, "country index")
    if index is None:
        return
    state.supported_countries = {entry.paypal_market_code for entry in index.countries}


def _load_crawl_state(state: PreviousState, state_path: Path) -> None:
    crawl_state = _safe_load(state_path, CrawlState.model_validate_json, "crawl state")
    if crawl_state is None:
        return
    for cc, entry in crawl_state.markets.items():
        state.country_tables[cc] = entry.table_count
        state.country_rows[cc] = entry.row_count
        state.derived_status[cc] = entry.derived_status or "unclassified"
        state.core_categories[cc] = set(entry.selected_categories)


def _is_accepted(
    previous: PreviousState,
    kind: str,
    country_code: str,
    identifier: str | None = None,
) -> bool:
    """Return whether a specific change has been reviewed and accepted."""
    for accepted in previous.accepted_regressions.accepted:
        if accepted.country_code != country_code.upper():
            continue
        if accepted.kind != kind:
            continue
        if accepted.identifier is not None and accepted.identifier != identifier:
            continue
        return True
    return False


def _check_structural_regressions(
    current_supported: set[str],
    current_unsupported: set[str],
    current_transient: set[str],
) -> list[ChangeType]:
    changes: list[ChangeType] = []
    pairs = [
        (current_supported, current_unsupported, "supported", "unsupported"),
        (current_supported, current_transient, "supported", "transient"),
        (current_unsupported, current_transient, "unsupported", "transient"),
    ]
    for set_a, set_b, name_a, name_b in pairs:
        if set_a & set_b:
            changes.append(
                _change_type(
                    kind="structural_regression",
                    message=f"A market is both {name_a} and {name_b} in the current state",
                )
            )
    return changes


def _check_removed_supported(
    previous: PreviousState,
    current_supported: set[str],
    current_transient: set[str],
    current_unsupported: set[str],
    limits: RegressionLimits,
) -> list[ChangeType]:
    changes: list[ChangeType] = []
    for cc in sorted(previous.supported_countries - current_supported):
        if cc in current_transient:
            changes.append(
                _change_type(
                    kind="supported_to_transient",
                    country_code=cc,
                    message=f"Supported country {cc} became transient",
                )
            )
        elif cc in current_unsupported:
            changes.append(
                _change_type(
                    kind="supported_to_unsupported",
                    country_code=cc,
                    message=f"Supported country {cc} became unsupported",
                )
            )
        elif not limits.allow_country_drop:
            changes.append(
                _change_type(
                    kind="removed_country",
                    country_code=cc,
                    message=f"Supported country {cc} disappeared",
                )
            )
    return changes


def _check_sharp_country_drop(
    previous: PreviousState,
    current_supported: set[str],
    limits: RegressionLimits,
) -> list[ChangeType]:
    changes: list[ChangeType] = []
    if (
        not limits.allow_country_drop
        and previous.supported_countries
        and len(previous.supported_countries - current_supported) / len(previous.supported_countries)
        > limits.max_country_count_delta_ratio
    ):
        changes.append(
            _change_type(
                kind="sharp_country_drop",
                message=f"Supported country count dropped by more than {limits.max_country_count_delta_ratio:.0%}",
            )
        )
    return changes


def _check_removed_discovered(
    previous: PreviousState,
    current_discovered: set[str],
    limits: RegressionLimits,
) -> list[ChangeType]:
    changes: list[ChangeType] = []
    removed_discovered = previous.discovered_countries - current_discovered
    if removed_discovered and not limits.allow_country_drop:
        for cc in sorted(removed_discovered):
            changes.append(
                _change_type(
                    kind="discovered_to_missing",
                    country_code=cc,
                    message=f"Discovered country {cc} is no longer known",
                )
            )
    return changes


def _check_added_supported(
    previous: PreviousState,
    current_supported: set[str],
) -> list[ChangeType]:
    changes: list[ChangeType] = []
    for cc in sorted(current_supported - previous.supported_countries):
        if cc in previous.unsupported_countries:
            changes.append(
                _change_type(
                    kind="unsupported_to_supported",
                    country_code=cc,
                    message=f"Unsupported country {cc} is now supported",
                )
            )
        elif cc in previous.transient_countries:
            changes.append(
                _change_type(
                    kind="transient_to_supported",
                    country_code=cc,
                    message=f"Transient country {cc} is now supported",
                )
            )
        else:
            changes.append(
                _change_type(
                    kind="added_country",
                    country_code=cc,
                    message=f"Country {cc} newly supported",
                )
            )
    return changes


def _check_newly_discovered(
    previous: PreviousState,
    current_discovered: set[str],
    current_supported: set[str],
    current_unsupported: set[str],
    current_transient: set[str],
) -> list[ChangeType]:
    changes: list[ChangeType] = []
    for cc in sorted(
        current_discovered - previous.discovered_countries - current_supported - current_unsupported - current_transient
    ):
        changes.append(
            _change_type(
                kind="newly_discovered",
                country_code=cc,
                message=f"Country {cc} discovered but not yet resolved",
            )
        )
    return changes


def _check_table_change(
    prev_tables: int,
    table_count: int,
    cc: str,
    limits: RegressionLimits,
) -> ChangeType | None:
    if prev_tables > 0 and table_count == 0:
        return _change_type(
            kind="removed_all_tables",
            country_code=cc,
            before=prev_tables,
            after=table_count,
            message=f"All tables removed for {cc}",
        )
    if prev_tables > 0 and table_count < prev_tables:
        ratio = (prev_tables - table_count) / prev_tables
        kind = "sharp_table_drop" if ratio > limits.max_table_count_delta_ratio else "table_count_decreased"
        return _change_type(
            kind=kind,
            country_code=cc,
            before=prev_tables,
            after=table_count,
            message=f"Table count for {cc} dropped by {ratio:.0%}"
            if kind == "sharp_table_drop"
            else f"Table count for {cc} decreased",
        )
    if table_count > prev_tables:
        return _change_type(
            kind="new_table",
            country_code=cc,
            before=prev_tables,
            after=table_count,
            message=f"Table count for {cc} increased",
        )
    return None


def _check_row_change(
    prev_rows: int,
    row_count: int,
    cc: str,
    limits: RegressionLimits,
) -> ChangeType | None:
    if prev_rows > 0 and row_count < prev_rows:
        ratio = (prev_rows - row_count) / prev_rows if prev_rows else 0
        if ratio > limits.max_row_count_delta_ratio:
            return _change_type(
                kind="sharp_row_drop",
                country_code=cc,
                before=prev_rows,
                after=row_count,
                message=f"Row count for {cc} dropped by {ratio:.0%}",
            )
    return None


def _check_lost_categories(
    previous: PreviousState,
    output: CountryOutput,
    cc: str,
) -> list[ChangeType]:
    prev_categories = previous.core_categories.get(cc, set())
    current_categories = _selected_categories_from_derived(output.derived)
    changes: list[ChangeType] = []
    for category in sorted(prev_categories - current_categories):
        accepted = _is_accepted(previous, "lost_core_category", cc, category)
        changes.append(
            _change_type(
                kind="lost_core_category",
                country_code=cc,
                identifier=category,
                before=True,
                after=False,
                message=f"Core category {category} disappeared for {cc}"
                + (" (accepted regression)" if accepted else ""),
                accepted=accepted,
            )
        )
    return changes


def _check_status_change(
    prev_status: str | None,
    output: CountryOutput,
    cc: str,
) -> ChangeType | None:
    if prev_status in {"complete", "partial"} and output.derived.status == "unclassified":
        return _change_type(
            kind="classified_to_unclassified",
            country_code=cc,
            before=prev_status,
            after="unclassified",
            message=f"Derived data for {cc} became unclassified",
        )
    return None


def _check_per_country_changes(
    previous: PreviousState,
    current_supported: set[str],
    current_outputs: dict[str, CountryOutput],
    limits: RegressionLimits,
) -> list[ChangeType]:
    changes: list[ChangeType] = []
    for cc in sorted(current_supported):
        output = current_outputs[cc]
        prev_tables = previous.country_tables.get(cc, 0)
        prev_rows = previous.country_rows.get(cc, 0)
        prev_status = previous.derived_status.get(cc)

        table_count = len(output.tables)
        row_count = sum(len(table.rows) for table in output.tables)

        table_change = _check_table_change(prev_tables, table_count, cc, limits)
        if table_change:
            changes.append(table_change)

        row_change = _check_row_change(prev_rows, row_count, cc, limits)
        if row_change:
            changes.append(row_change)

        changes.extend(_check_lost_categories(previous, output, cc))

        status_change = _check_status_change(prev_status, output, cc)
        if status_change:
            changes.append(status_change)
    return changes


def _check_classifier_version(
    previous: PreviousState,
    current_classifier_metadata: ClassifierMetadata | None,
) -> list[ChangeType]:
    changes: list[ChangeType] = []
    if (
        current_classifier_metadata is not None
        and previous.classifier_metadata is not None
        and current_classifier_metadata != previous.classifier_metadata
    ):
        changes.append(
            _change_type(
                kind="classifier_version_changed",
                before=previous.classifier_metadata.model_dump(),
                after=current_classifier_metadata.model_dump(),
                message=f"Classifier metadata changed from {previous.classifier_metadata.classifier_version} to {current_classifier_metadata.classifier_version}",
            )
        )
    return changes


def check_regression(
    previous: PreviousState,
    current_discovered: set[str],
    current_supported: set[str],
    current_unsupported: set[str],
    current_transient: set[str],
    current_outputs: dict[str, CountryOutput],
    limits: RegressionLimits,
    current_classifier_metadata: ClassifierMetadata | None = None,
) -> ChangeReport:
    """Compare the new market states with the previous state and return a change report.

    Every market state is compared against its equivalent previous state:
    discovered vs discovered, supported vs supported, unsupported vs unsupported,
    and transient vs transient. No state is compared against a different state.
    """
    changes: list[ChangeType] = []
    changes.extend(_check_structural_regressions(current_supported, current_unsupported, current_transient))
    changes.extend(
        _check_removed_supported(previous, current_supported, current_transient, current_unsupported, limits)
    )
    changes.extend(_check_sharp_country_drop(previous, current_supported, limits))
    changes.extend(_check_removed_discovered(previous, current_discovered, limits))
    changes.extend(_check_added_supported(previous, current_supported))
    changes.extend(
        _check_newly_discovered(
            previous,
            current_discovered,
            current_supported,
            current_unsupported,
            current_transient,
        )
    )
    changes.extend(_check_per_country_changes(previous, current_supported, current_outputs, limits))
    changes.extend(_check_classifier_version(previous, current_classifier_metadata))
    return ChangeReport(schema_version=1, changes=changes)


def _regression_kind_allowlist() -> set[str]:
    """Return the set of change kinds treated as regressions.

    Derived from ``models._CHANGE_SEVERITY_BY_KIND`` so the regression guard
    and the severity model stay in sync.
    """
    return {kind for kind, severity in _CHANGE_SEVERITY_BY_KIND.items() if severity == ChangeSeverity.REGRESSION}


def enforce_regression(
    report: ChangeReport,
    fail_on_regression: bool = False,
) -> None:
    """Raise RegressionError if the report contains regressions and enforcement is enabled."""
    if not report.has_regression:
        return
    if fail_on_regression:
        regression_kinds = _regression_kind_allowlist()
        raise RegressionError(
            "Regression detected:\n"
            + "\n".join(f"- {c.kind}: {c.message}" for c in report.changes if c.kind in regression_kinds)
        )
    logger.warning("Regression detected but not enforced: %d changes", len(report.changes))
