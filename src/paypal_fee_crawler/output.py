"""Atomic, deterministic output generation for the PayPal fee crawler."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import shutil
import subprocess  # nosec B404
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .constants import MANAGED_PATHS
from .derived_categories import _selected_categories_from_derived
from .exceptions import ValidationError as CrawlerValidationError
from .hashing import _country_output_hash
from .models import (
    ChangeReport,
    ClassifierMetadata,
    CoreFeeDerived,
    CoreFeeFixedFeeSchedule,
    CoreFeeInternationalSurchargeSchedule,
    CoreFeeRateReference,
    CoreFeeResolvedRate,
    CoreFeeRule,
    CoreFees,
    CountryIndex,
    CountryIndexEntry,
    CountryManifest,
    CountryOutput,
    CrawlCache,
    CrawlCacheEntry,
    CrawlState,
    CrawlStateEntry,
    Market,
    PublicCoreFeeEntry,
    PublicCountryOutput,
    SchemaVersionInfo,
    UnsupportedCountry,
)
from .validation import (
    generate_core_fees_schema,
    generate_country_schema,
    generate_index_schema,
    generate_manifest_schema,
    validate_output_tree,
)

logger = logging.getLogger(__name__)


def _to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses, Pydantic models and enums to JSON-compatible plain objects."""
    if isinstance(obj, BaseModel):
        return _to_jsonable(obj.model_dump(mode="json"))
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (set, frozenset)):
        return sorted(_to_jsonable(v) for v in obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return _to_jsonable(dataclasses.asdict(obj))
    return str(obj)


def _normalize_public_output(obj: Any, parent_key: str | None = None) -> Any:
    """Recursively normalize public output data for deterministic serialization.

    Lists of dicts are sorted by a canonical JSON key so that non-semantic
    ordering differences do not change the published artifacts. Scalar lists
    are also sorted unless they carry positional meaning (e.g. ``normalized_cells``
    or ``section_path``).
    """
    if isinstance(obj, dict):
        return {k: _normalize_public_output(v, parent_key=k) for k, v in obj.items()}
    if isinstance(obj, list):
        normalized = [_normalize_public_output(item) for item in obj]
        if not normalized:
            return normalized
        if parent_key in {"normalized_cells", "section_path"}:
            return normalized
        if all(isinstance(item, dict) for item in normalized):
            return sorted(
                normalized,
                key=lambda d: json.dumps(d, sort_keys=True, ensure_ascii=False),
            )
        if all(isinstance(item, (str, int, float, bool, type(None))) for item in normalized):
            return sorted(normalized, key=lambda x: (x is None, type(x).__name__, x))
        return normalized
    return obj


def _serialize(obj: Any) -> str:
    """Serialize an object to deterministic JSON with stable ordering."""
    return (
        json.dumps(
            obj,
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n"
    )


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_serialize(data), encoding="utf-8")


def _is_same_file(dst: Path, src: Path) -> bool:
    if not dst.exists():
        return False
    if dst.stat().st_size != src.stat().st_size:
        return False
    return dst.read_text(encoding="utf-8") == src.read_text(encoding="utf-8")


@dataclass
class _JournalEntry:
    """One managed-path operation in the publication transaction.

    The entry is appended to the journal before the first filesystem mutation.
    This is essential: if the backup succeeds but installing the staged path
    fails, rollback still knows how to restore the original live path.
    """

    managed_name: str
    live_path: Path
    backup_path: Path
    staged_path: Path | None
    action: str
    original_existed: bool
    backup_created: bool = False
    live_installed: bool = False
    finalized: bool = False


class OutputPublisher:
    """Publish crawler output atomically with deterministic, schema-validated files."""

    # These are the only paths the crawler owns.  The output directory itself may
    # be the root of a git repository and must never be renamed or deleted.
    MANAGED_PATHS = MANAGED_PATHS

    # Paths that may legitimately differ between runs (e.g. the crawl report
    # records whether the run changed the data) and must not feed back into the
    # "changed" flag itself.
    IGNORED_CHANGED_PATHS = frozenset({"meta/crawl-report.json", "change-report.json"})

    def __init__(
        self,
        output_dir: Path | str,
        staging_dir: Path | str | None = None,
        timestamp: str | None = None,
        keep_diagnostics: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.staging_dir = Path(staging_dir) if staging_dir else None
        # If no timestamp is provided, generated_at falls back to the page source
        # update date when available; otherwise it remains null.
        self.timestamp = timestamp
        self.keep_diagnostics = keep_diagnostics

    def _make_staging(self) -> Path:
        if self.staging_dir and self.staging_dir.is_relative_to(self.output_dir):
            # Only use a staging directory that lives inside the output tree;
            # otherwise atomic rename of managed subdirectories cannot be
            # guaranteed on all platforms.
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            return self.staging_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=".staging-", dir=str(self.output_dir)))

    def _compute_content_sha256(self, data: dict[str, Any]) -> str:
        return _country_output_hash(data)

    def _run_generated_at(self) -> str | None:
        """Return the configured run timestamp."""
        return self.timestamp

    def _write_crawler_revision(self, staging: Path, crawler_revision: str | None) -> None:
        """Write the crawler Git revision used to generate this data."""
        if crawler_revision:
            _write_json(
                staging / "meta" / "crawler-revision.json",
                {"crawler_revision": crawler_revision, "generated_at": self.timestamp},
            )

    def publish_change_report(self, staging: Path, change_report: ChangeReport | None) -> None:
        """Publish the computed change report, overwriting any stale previous report."""
        if change_report is None:
            change_report = ChangeReport()
        change_report = change_report.model_copy(update={"generated_at": self.timestamp})
        _write_json(
            staging / "change-report.json",
            _normalize_public_output(change_report.model_dump(mode="json")),
        )

    def generate_readme(self, staging: Path) -> None:
        """Regenerate README.md from the staged artifacts."""
        script_path = self.output_dir / "scripts" / "generate_readme.py"
        if not script_path.exists():
            # Fall back to the bundled script adjacent to the crawler package.
            script_path = Path(__file__).resolve().parent.parent.parent.parent / "scripts" / "generate_readme.py"
        if not script_path.exists():
            return
        try:
            subprocess.run(  # nosec
                [sys.executable, str(script_path), str(staging)],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            logger.warning("README generation failed: %s", exc)

    def write_crawl_report(self, output_dir: Path, report: Any) -> None:
        """Write the crawl execution report to meta/crawl-report.json."""
        path = output_dir / "meta" / "crawl-report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _build_state_entry(
        self,
        output: CountryOutput,
        artifact_sha256: str,
        existing_state: CrawlState | None,
    ) -> CrawlStateEntry:
        existing_entry = self._existing_state_entry(output, existing_state, artifact_sha256)
        if existing_entry is not None:
            table_fingerprints = list(existing_entry.table_fingerprints)
            classifier_version = existing_entry.classifier_version
        else:
            table_fingerprints = self._table_fingerprints_for_output(output)
            classifier_version = None

        return CrawlStateEntry(
            raw_content_sha256=output.source.content_sha256,
            artifact_sha256=artifact_sha256,
            classifier_version=classifier_version,
            derived_status=output.derived.status,
            selected_categories=sorted(_selected_categories_from_derived(output.derived)),
            table_count=len(output.tables),
            row_count=sum(len(table.rows) for table in output.tables),
            table_fingerprints=table_fingerprints,
            source_url=output.source.canonical_url or output.source.requested_url,
            source_updated_at=output.source.page_updated_at,
        )

    def _existing_state_entry(
        self,
        output: CountryOutput,
        existing_state: CrawlState | None,
        artifact_sha256: str,
    ) -> CrawlStateEntry | None:
        """Return the matching prior state entry if this output is unchanged."""
        if existing_state is None:
            return None
        existing_entry = existing_state.markets.get(output.market.paypal_market_code)
        if (
            existing_entry is not None
            and existing_entry.artifact_sha256 == artifact_sha256
            and existing_entry.raw_content_sha256 == output.source.content_sha256
        ):
            return existing_entry
        return None

    def _table_fingerprints_for_output(self, output: CountryOutput) -> list[str]:
        """Return fresh table fingerprints for a newly generated output."""
        fingerprints: list[str] = []
        for table in output.tables:
            table_dump = table.model_dump(mode="json", exclude_none=True)
            table_hash = hashlib.sha256(
                json.dumps(table_dump, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
            ).hexdigest()
            fingerprints.append(f"sha256:{table_hash}")
        return fingerprints

    def _to_core_fee_derived(self, derived: Any) -> CoreFeeDerived:
        """Return a compact calculator-only copy of a derived fee result."""
        rules: list[CoreFeeRule] = []
        for rule in derived.transaction_fee_rules:
            rate_ref = None
            if rule.rate_reference is not None:
                resolved = None
                if rule.rate_reference.resolved_rate is not None:
                    resolved = CoreFeeResolvedRate(
                        percentage=rule.rate_reference.resolved_rate.percentage,
                        fixed_fee_schedule=rule.rate_reference.resolved_rate.fixed_fee_schedule,
                        international_surcharge_schedule=rule.rate_reference.resolved_rate.international_surcharge_schedule,
                        maximum_fee_schedule=rule.rate_reference.resolved_rate.maximum_fee_schedule,
                        rule_id=rule.rate_reference.resolved_rate.rule_id,
                    )
                rate_ref = CoreFeeRateReference(
                    reference=rule.rate_reference.reference,
                    resolved_rate=resolved,
                )
            rules.append(
                CoreFeeRule(
                    id=rule.id,
                    variant_id=rule.variant_id,
                    label=rule.label,
                    percentage=rule.percentage,
                    fixed_fee_schedule=rule.fixed_fee_schedule,
                    international_surcharge_schedule=rule.international_surcharge_schedule,
                    maximum_fee_schedule=rule.maximum_fee_schedule,
                    rate_reference=rate_ref,
                    conditions=rule.conditions,
                    calculation_status=rule.calculation_status,
                    fee_components=rule.fee_components,
                )
            )

        fixed: dict[str, CoreFeeFixedFeeSchedule] = {
            name: CoreFeeFixedFeeSchedule(entries=schedule.entries)
            for name, schedule in derived.fixed_fee_schedules.items()
        }
        intl: dict[str, CoreFeeInternationalSurchargeSchedule] = {
            name: CoreFeeInternationalSurchargeSchedule(entries=schedule.entries)
            for name, schedule in derived.international_surcharge_schedules.items()
        }
        maximum: dict[str, CoreFeeFixedFeeSchedule] = {
            name: CoreFeeFixedFeeSchedule(entries=schedule.entries)
            for name, schedule in derived.maximum_fee_schedules.items()
        }
        return CoreFeeDerived(
            status=derived.status,
            transaction_fee_rules=rules,
            fixed_fee_schedules=fixed,
            international_surcharge_schedules=intl,
            maximum_fee_schedules=maximum,
            currency_conversion=derived.currency_conversion,
        )

    def _load_existing_cache(self) -> CrawlCache:
        """Load the previous crawl cache so 304/reused runs retain cache headers."""
        path = self.output_dir / "meta" / "crawl-cache.json"
        if not path.exists():
            return CrawlCache()
        return CrawlCache.model_validate_json(path.read_text(encoding="utf-8"))

    def _load_existing_state(self) -> CrawlState | None:
        """Load the previous crawl state so unchanged runs can reuse table fingerprints."""
        path = self.output_dir / "meta" / "crawl-state.json"
        if not path.exists():
            return None
        try:
            return CrawlState.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:  # nosec B112 # noqa: S112
            return None

    def publish(
        self,
        outputs: dict[str, CountryOutput],
        markets: list[Market],
        unsupported: list[UnsupportedCountry],
        change_report: ChangeReport | None = None,
        diagnostics: dict[str, Any] | None = None,
        classifier_metadata: ClassifierMetadata | None = None,
        crawler_revision: str | None = None,
        transient_failures: list[UnsupportedCountry] | None = None,
    ) -> tuple[bool, Path]:
        """Write all output files to a staging directory and return (changed, staging_path)."""
        staging = self._make_staging()
        json_dir = staging / "json"
        meta_dir = staging / "meta"
        schemas_dir = staging / "schemas"
        json_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        schemas_dir.mkdir(parents=True, exist_ok=True)

        index_entries: list[CountryIndexEntry] = []
        core_entries: list[PublicCoreFeeEntry] = []

        existing_cache = self._load_existing_cache()
        cache_entries: dict[str, CrawlCacheEntry] = {
            market_code: entry for market_code, entry in existing_cache.markets.items() if market_code in outputs
        }

        existing_state = self._load_existing_state()
        state_entries: dict[str, CrawlStateEntry] = {}
        run_generated_at = self._run_generated_at()

        for cc in sorted(outputs.keys()):
            output = outputs[cc]
            public = PublicCountryOutput.from_internal(output)
            # generated_at and crawled_at are the run timestamp; source_updated_at
            # and cms_updated_at are preserved from the source page.
            run_timestamp = self.timestamp or output.generated_at
            if run_timestamp:
                public = public.model_copy(
                    update={
                        "generated_at": run_timestamp,
                        "crawled_at": run_timestamp,
                    }
                )

            path = json_dir / f"{output.market.url_slug}.json"
            country_data = _normalize_public_output(public.model_dump(mode="json", exclude_none=True))
            content_hash = self._compute_content_sha256(country_data)
            _write_json(path, country_data)

            if output.source.etag or output.source.last_modified:
                cache_entries[output.market.paypal_market_code] = CrawlCacheEntry(
                    etag=output.source.etag,
                    last_modified=output.source.last_modified,
                    content_sha256=output.source.content_sha256,
                )

            index_entries.append(
                CountryIndexEntry(
                    paypal_market_code=output.market.paypal_market_code,
                    iso_country_code=output.market.iso_country_code,
                    locale=output.market.locale,
                    data_url=f"json/{output.market.url_slug}.json",
                    source_url=output.source.canonical_url or output.source.requested_url,
                    source_updated_at=output.source.page_updated_at,
                    crawled_at=run_timestamp,
                    derived_status=output.derived.status,
                    content_sha256=content_hash,
                )
            )
            core_entries.append(
                PublicCoreFeeEntry(
                    paypal_market_code=output.market.paypal_market_code,
                    iso_country_code=output.market.iso_country_code,
                    derived_status=output.derived.status,
                    derived=self._to_core_fee_derived(output.derived),
                )
            )

            state_entries[output.market.paypal_market_code] = self._build_state_entry(
                output, content_hash, existing_state
            )

        index = CountryIndex(generated_at=run_generated_at, countries=index_entries)
        index_data = _normalize_public_output(index.model_dump(mode="json", exclude_none=True))
        _write_json(json_dir / "index.json", index_data)

        core_fees = CoreFees(generated_at=run_generated_at, countries=core_entries)
        core_data = _normalize_public_output(core_fees.model_dump(mode="json", exclude_none=True))
        _write_json(json_dir / "core-fees.json", core_data)

        manifest = CountryManifest(
            generated_at=run_generated_at,
            markets=markets,
            unsupported=unsupported,
            transient_failures=transient_failures or [],
        )
        manifest_data = _normalize_public_output(manifest.model_dump(mode="json", exclude_none=True))
        manifest_data["generated_at"] = manifest.generated_at
        _write_json(meta_dir / "countries.json", manifest_data)
        _write_json(
            meta_dir / "unsupported-countries.json",
            _normalize_public_output(
                {
                    "schema_version": 1,
                    "unsupported": [u.model_dump(mode="json", exclude_none=True) for u in unsupported],
                }
            ),
        )
        _write_json(
            meta_dir / "transient-failures.json",
            _normalize_public_output(
                {
                    "schema_version": 1,
                    "transient_failures": [
                        u.model_dump(mode="json", exclude_none=True) for u in (transient_failures or [])
                    ],
                }
            ),
        )
        _write_json(
            meta_dir / "schema-version.json",
            SchemaVersionInfo(
                description="Public schema for PayPal fee data v1",
            ).model_dump(mode="json", exclude_none=True),
        )

        _write_json(schemas_dir / "paypal-fees-v1.schema.json", generate_country_schema())
        _write_json(schemas_dir / "core-fees-v1.schema.json", generate_core_fees_schema())
        _write_json(schemas_dir / "index-v1.schema.json", generate_index_schema())
        _write_json(schemas_dir / "manifest-v1.schema.json", generate_manifest_schema())

        _write_json(
            meta_dir / "crawl-cache.json",
            _normalize_public_output(CrawlCache(markets=cache_entries).model_dump(mode="json", exclude_none=True)),
        )

        _write_json(
            meta_dir / "crawl-state.json",
            _normalize_public_output(
                CrawlState(generated_at=run_generated_at, markets=state_entries).model_dump(mode="json")
            ),
        )

        if classifier_metadata is not None:
            _write_json(
                meta_dir / "classifier-version.json",
                _normalize_public_output(classifier_metadata.model_dump(mode="json", exclude_none=True)),
            )

        accepted_path = self.output_dir / "meta" / "accepted-regressions.json"
        if accepted_path.exists():
            shutil.copy2(accepted_path, meta_dir / "accepted-regressions.json")

        if self.keep_diagnostics:
            diagnostics_dir = meta_dir / "diagnostics"
            diagnostics_dir.mkdir(parents=True, exist_ok=True)
            for cc in sorted(outputs.keys()):
                run = diagnostics.get(cc) if diagnostics else None
                _write_json(
                    diagnostics_dir / f"{cc.lower()}.json",
                    _normalize_public_output(
                        {
                            "schema_version": 1,
                            "generated_at": outputs[cc].source.page_updated_at or self.timestamp,
                            "normalized_output": outputs[cc].model_dump(mode="json", exclude_none=True),
                            "classification_run": _to_jsonable(run) if run is not None else None,
                        }
                    ),
                )

        self._write_crawler_revision(staging, crawler_revision)
        self.publish_change_report(staging, change_report)

        return staging != self.output_dir, staging

    def commit(self, staging: Path) -> tuple[bool, list[str]]:
        """Atomically replace only managed paths with the staged tree.

        This method is safe when ``output_dir`` is the root of a git repository:
        only ``MANAGED_PATHS`` are touched.  Staging is cross-file validated
        before any live path is modified.  Backups remain available until the
        live tree has also passed validation.
        """
        changed_files = self._list_changed_files(staging)
        if not changed_files and self._output_dir_exists_and_matches(staging):
            self.rollback(staging)
            return False, []

        errors = validate_output_tree(staging)
        if errors:
            self.rollback(staging)
            raise CrawlerValidationError("Staging output failed cross-file validation:\n" + "\n".join(errors))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        journal: list[_JournalEntry] = []
        finalized = False

        try:
            for name in self.MANAGED_PATHS:
                src = staging / name
                dst = self.output_dir / name
                backup = dst.with_name(f"{dst.name}.old")
                src_exists = src.exists()
                dst_exists = dst.exists()

                if dst_exists and src_exists:
                    action = "replaced"
                elif dst_exists and not src_exists:
                    action = "removed"
                elif not dst_exists and src_exists:
                    action = "added"
                else:
                    action = "none"

                entry = _JournalEntry(
                    managed_name=name,
                    live_path=dst,
                    backup_path=backup,
                    staged_path=src if src_exists else None,
                    action=action,
                    original_existed=dst_exists,
                )

                # Register before mutating anything.  If backup succeeds and the
                # next rename fails, rollback will still restore the backup.
                journal.append(entry)

                if action == "none":
                    continue

                self._remove_path(backup)

                if dst_exists:
                    os.rename(dst, backup)
                    entry.backup_created = True

                if src_exists:
                    os.rename(src, dst)
                    entry.live_installed = True

            final_errors = validate_output_tree(self.output_dir)
            if final_errors:
                raise CrawlerValidationError("Live output failed cross-file validation:\n" + "\n".join(final_errors))

            finalized = True
            for entry in journal:
                entry.finalized = True

            self._cleanup_backups_best_effort(journal)
            self.rollback(staging)
            # The change-report and crawl-report are generated summaries, not data
            # changes; a crawl that only updates these reports (e.g. clearing a
            # stale regression) should still be considered unchanged.
            data_changed = bool(set(changed_files) - self.IGNORED_CHANGED_PATHS)
            return data_changed, changed_files

        except Exception as exc:
            if not finalized:
                self._rollback_live(journal)
            self.rollback(staging)
            if isinstance(exc, CrawlerValidationError):
                raise
            raise CrawlerValidationError(f"Failed to publish output: {exc}") from exc

    def _cleanup_backups_best_effort(self, journal: list[_JournalEntry]) -> None:
        """Remove backups after a successful, validated commit.

        Backup cleanup is intentionally best-effort.  Once the live tree has
        passed final validation, a cleanup failure must not trigger rollback of a
        good publication.  Leftover ``*.old`` paths can be removed on a later run.
        """
        for entry in journal:
            if entry.backup_created and entry.backup_path.exists():
                try:
                    self._remove_path(entry.backup_path)
                except Exception as exc:  # pragma: no cover - platform dependent
                    logger.warning("Could not remove publication backup %s: %s", entry.backup_path, exc)

    def _rollback_live(self, journal: list[_JournalEntry]) -> None:
        """Restore managed live paths to their pre-transaction state."""
        failed: list[str] = []

        for entry in reversed(journal):
            if entry.action == "none":
                continue

            live = entry.live_path
            backup = entry.backup_path

            # Only remove the live path if we know a mutation actually happened.
            # If the journal entry was appended but neither backup nor live swap
            # completed, the original live path must be left untouched.
            mutation_happened = entry.live_installed or entry.backup_created

            if mutation_happened and live.exists():
                try:
                    self._remove_path(live)
                except Exception as exc:
                    failed.append(f"Could not remove live path {live}: {exc}")
                    continue

            if entry.original_existed:
                if entry.backup_created and backup.exists():
                    try:
                        os.rename(backup, live)
                    except Exception as exc:
                        failed.append(f"Could not restore backup {backup} to {live}: {exc}")
                elif mutation_happened and not live.exists():
                    failed.append(f"Backup missing for {live}; original state cannot be restored")
            else:
                # The path did not exist before the transaction.  Ensure any
                # installed path is gone; no backup should be restored.
                if entry.live_installed and live.exists():
                    try:
                        self._remove_path(live)
                    except Exception as exc:
                        failed.append(f"Could not remove added path {live}: {exc}")

        if failed:
            logger.error("Rollback completed with errors:\n%s", "\n".join(failed))
            raise CrawlerValidationError("Rollback completed with errors:\n" + "\n".join(failed))

    def _remove_path(self, path: Path) -> None:
        """Remove a file or directory tree, ignoring missing paths."""
        if not path.exists():
            return
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)

    def _list_changed_files(self, staging: Path) -> list[str]:
        """Return relative paths of managed files that differ from published output."""
        changed: list[str] = []
        for name in self.MANAGED_PATHS:
            src = staging / name
            if not src.exists():
                continue
            if src.is_dir():
                for src_file in src.rglob("*"):
                    if not src_file.is_file():
                        continue
                    rel = src_file.relative_to(staging)
                    dst = self.output_dir / rel
                    if not _is_same_file(dst, src_file):
                        changed.append(str(rel))
            else:
                rel = src.relative_to(staging)
                dst = self.output_dir / rel
                if not _is_same_file(dst, src):
                    changed.append(str(rel))

        if self.output_dir.exists():
            for name in self.MANAGED_PATHS:
                dst = self.output_dir / name
                if not dst.exists():
                    continue
                if dst.is_dir():
                    for dst_file in dst.rglob("*"):
                        if not dst_file.is_file():
                            continue
                        rel = dst_file.relative_to(self.output_dir)
                        if not (staging / rel).exists():
                            changed.append(str(rel))
                else:
                    rel = dst.relative_to(self.output_dir)
                    if not (staging / rel).exists():
                        changed.append(str(rel))
        return sorted(set(changed))

    def _output_dir_exists_and_matches(self, staging: Path) -> bool:
        """Return whether the output directory exists and matches staging exactly."""
        if not self.output_dir.exists():
            return False
        return not self._list_changed_files(staging)

    def rollback(self, staging: Path) -> None:
        """Clean up the staging directory on failure or when no change is published."""
        if staging.exists() and staging != self.output_dir and not self._is_managed_path(staging):
            shutil.rmtree(staging, ignore_errors=True)

    def _is_managed_path(self, path: Path) -> bool:
        """Return whether the path is one of the live managed paths."""
        try:
            rel = path.relative_to(self.output_dir)
        except ValueError:
            return False
        parts = rel.parts
        if not parts:
            return False
        return parts[0] in self.MANAGED_PATHS or (len(parts) == 1 and parts[0] in self.MANAGED_PATHS)
