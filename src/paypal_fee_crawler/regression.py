"""Regression guards and change reporting."""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .comparison import _selected_categories_from_derived
from .exceptions import RegressionError
from .models import (
    AcceptedRegressions,
    ChangeReport,
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
        manifest_path = output_dir / "meta" / "countries.json"
        if manifest_path.exists():
            try:
                manifest = CountryManifest.model_validate_json(manifest_path.read_text())
                state.discovered_countries = {m.paypal_market_code for m in manifest.markets}
                state.unsupported_countries = {u.paypal_market_code for u in manifest.unsupported}
                state.transient_countries = {u.paypal_market_code for u in manifest.unsupported if u.temporary}
                state.unsupported_records = {u.paypal_market_code: u for u in manifest.unsupported}
            except Exception as exc:  # nosec B112 # noqa: S112
                logger.warning("Could not load previous country manifest: %s", exc)

        unsupported_path = output_dir / "meta" / "unsupported-countries.json"
        if unsupported_path.exists():
            try:
                data = json.loads(unsupported_path.read_text(encoding="utf-8"))
                for item in data.get("unsupported", []):
                    u = UnsupportedCountry.model_validate(item)
                    state.unsupported_records[u.paypal_market_code] = u
                    state.unsupported_countries.add(u.paypal_market_code)
                    if u.temporary:
                        state.transient_countries.add(u.paypal_market_code)
            except Exception as exc:  # nosec B112 # noqa: S112
                logger.warning("Could not load previous unsupported metadata: %s", exc)

        accepted_path = output_dir / "meta" / "accepted-regressions.json"
        if accepted_path.exists():
            try:
                state.accepted_regressions = AcceptedRegressions.model_validate_json(accepted_path.read_text())
            except Exception as exc:  # nosec B112 # noqa: S112
                logger.warning("Could not load accepted regressions: %s", exc)

        classifier_path = output_dir / "meta" / "classifier-version.json"
        if classifier_path.exists():
            try:
                state.classifier_metadata = ClassifierMetadata.model_validate_json(classifier_path.read_text())
            except Exception as exc:  # nosec B112 # noqa: S112
                logger.warning("Could not load previous classifier metadata: %s", exc)

        index_path = output_dir / "json" / "index.json"
        if index_path.exists():
            try:
                index = CountryIndex.model_validate_json(index_path.read_text())
                state.supported_countries = {entry.paypal_market_code for entry in index.countries}
            except Exception as exc:  # nosec B112 # noqa: S112
                logger.warning("Could not load previous country index: %s", exc)

        state_path = output_dir / "meta" / "crawl-state.json"
        if state_path.exists():
            try:
                crawl_state = CrawlState.model_validate_json(state_path.read_text())
                for cc, entry in crawl_state.markets.items():
                    state.country_tables[cc] = entry.table_count
                    state.country_rows[cc] = entry.row_count
                    state.derived_status[cc] = entry.derived_status or "unclassified"
                    state.core_categories[cc] = set(entry.selected_categories)
            except Exception as exc:  # nosec B112 # noqa: S112
                logger.warning("Could not load previous crawl state: %s", exc)

        # Backward-compatible alias includes all discovered markets.
        state.countries = state.discovered_countries
        return state


def _country_output_hash(data: dict[str, Any]) -> str:
    """Return a deterministic hash of the public business content of a country output."""
    canonical = {
        "market": data.get("market"),
        "source": {
            k: v
            for k, v in (data.get("source") or {}).items()
            if k not in {"etag", "last_modified", "content_sha256", "artifact_sha256"}
        },
        "sections": data.get("sections"),
        "tables": data.get("tables"),
        "derived": data.get("derived"),
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


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

    if current_supported & current_unsupported:
        changes.append(
            _change_type(
                kind="structural_regression",
                message="A market is both supported and unsupported in the current state",
            )
        )
    if current_supported & current_transient:
        changes.append(
            _change_type(
                kind="structural_regression",
                message="A market is both supported and transient in the current state",
            )
        )
    if current_unsupported & current_transient:
        changes.append(
            _change_type(
                kind="structural_regression",
                message="A market is both unsupported and transient in the current state",
            )
        )

    # Supported -> anything.
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
        else:
            if not limits.allow_country_drop:
                changes.append(
                    _change_type(
                        kind="removed_country",
                        country_code=cc,
                        message=f"Supported country {cc} disappeared",
                    )
                )

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

    # Discovered -> missing.
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

    # Added / state transitions.
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
            changes.append(_change_type(kind="added_country", country_code=cc, message=f"Country {cc} newly supported"))

    # Newly discovered but not yet supported.
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

    for cc in sorted(current_supported):
        output = current_outputs[cc]
        prev_tables = previous.country_tables.get(cc, 0)
        prev_rows = previous.country_rows.get(cc, 0)
        prev_categories = previous.core_categories.get(cc, set())
        prev_status = previous.derived_status.get(cc)

        table_count = len(output.tables)
        row_count = sum(len(table.rows) for table in output.tables)

        if prev_tables > 0 and table_count == 0:
            changes.append(
                _change_type(
                    kind="removed_all_tables",
                    country_code=cc,
                    before=prev_tables,
                    after=table_count,
                    message=f"All tables removed for {cc}",
                )
            )
        elif prev_tables > 0 and table_count < prev_tables:
            ratio = (prev_tables - table_count) / prev_tables
            kind = "sharp_table_drop" if ratio > limits.max_table_count_delta_ratio else "table_count_decreased"
            changes.append(
                _change_type(
                    kind=kind,
                    country_code=cc,
                    before=prev_tables,
                    after=table_count,
                    message=f"Table count for {cc} dropped by {ratio:.0%}"
                    if kind == "sharp_table_drop"
                    else f"Table count for {cc} decreased",
                )
            )
        elif table_count > prev_tables:
            changes.append(
                _change_type(
                    kind="new_table",
                    country_code=cc,
                    before=prev_tables,
                    after=table_count,
                    message=f"Table count for {cc} increased",
                )
            )

        if prev_rows > 0 and row_count < prev_rows:
            ratio = (prev_rows - row_count) / prev_rows if prev_rows else 0
            if ratio > limits.max_row_count_delta_ratio:
                changes.append(
                    _change_type(
                        kind="sharp_row_drop",
                        country_code=cc,
                        before=prev_rows,
                        after=row_count,
                        message=f"Row count for {cc} dropped by {ratio:.0%}",
                    )
                )

        current_categories = _selected_categories_from_derived(output.derived)

        lost_categories = prev_categories - current_categories
        for category in sorted(lost_categories):
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

        if prev_status in {"complete", "partial"} and output.derived.status == "unclassified":
            changes.append(
                _change_type(
                    kind="classified_to_unclassified",
                    country_code=cc,
                    before=prev_status,
                    after="unclassified",
                    message=f"Derived data for {cc} became unclassified",
                )
            )

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

    return ChangeReport(
        schema_version=1,
        changes=changes,
    )


def enforce_regression(
    report: ChangeReport,
    fail_on_regression: bool = False,
) -> None:
    """Raise RegressionError if the report contains regressions and enforcement is enabled."""
    if not report.has_regression:
        return
    if fail_on_regression:
        raise RegressionError(
            "Regression detected:\n"
            + "\n".join(
                f"- {c.kind}: {c.message}"
                for c in report.changes
                if c.kind
                in {
                    "removed_country",
                    "discovered_to_missing",
                    "supported_to_transient",
                    "supported_to_unsupported",
                    "removed_table",
                    "removed_all_tables",
                    "lost_core_category",
                    "structural_regression",
                    "sharp_table_drop",
                    "sharp_row_drop",
                    "sharp_country_drop",
                    "classified_to_unclassified",
                }
            )
        )
    logger.warning("Regression detected but not enforced: %d changes", len(report.changes))
