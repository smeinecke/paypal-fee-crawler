"""Atomic, deterministic output generation for the PayPal fee crawler."""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import ValidationError as CrawlerValidationError
from .models import (
    ChangeReport,
    CoreFeeEntry,
    CoreFees,
    CountryIndex,
    CountryIndexEntry,
    CountryManifest,
    CountryOutput,
    Market,
    SchemaVersionInfo,
    UnsupportedCountry,
)
from .regression import _country_output_hash
from .validation import (
    generate_core_fees_schema,
    generate_country_schema,
    generate_index_schema,
    generate_manifest_schema,
    validate_output_tree,
)

logger = logging.getLogger(__name__)


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


def _is_same_file(path: Path, content: str) -> bool:
    if not path.exists():
        return False
    return path.read_text(encoding="utf-8") == content


@dataclass
class _JournalEntry:
    """Single step of an atomic publication transaction."""

    managed_name: str
    live_path: Path
    backup_path: Path
    staged_path: Path | None
    action: str
    original_existed: bool
    backup_created: bool = False
    swapped: bool = False


class OutputPublisher:
    """Publish crawler output atomically with deterministic, schema-validated files."""

    # Managed paths are the only ones that may be created, replaced, or removed.
    # The output directory itself is never renamed, so this is safe to run at the
    # root of a git repository.
    MANAGED_PATHS = ("json", "meta", "schemas", "change-report.json")

    def __init__(
        self,
        output_dir: Path | str,
        staging_dir: Path | str | None = None,
        timestamp: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.staging_dir = Path(staging_dir) if staging_dir else None
        # If no timestamp is provided, canonical output is still deterministic;
        # generated_at is written as null. The caller is responsible for supplying
        # a stable run timestamp if it wants explicit timestamps in the output.
        self.timestamp = timestamp

    def _make_staging(self) -> Path:
        if self.staging_dir and self.staging_dir.is_relative_to(self.output_dir):
            # Only use a staging directory that lives inside the output tree;
            # otherwise atomic rename of managed subdirectories cannot be guaranteed.
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            return self.staging_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=".staging-", dir=str(self.output_dir)))

    def _compute_content_sha256(self, data: dict[str, Any]) -> str:
        return _country_output_hash(data)

    def publish(
        self,
        outputs: dict[str, CountryOutput],
        markets: list[Market],
        unsupported: list[UnsupportedCountry],
        change_report: ChangeReport | None = None,
    ) -> tuple[bool, Path]:
        """Write all output files to a staging directory and return (changed, staging_path)."""
        staging = self._make_staging()
        json_dir = staging / "json"
        meta_dir = staging / "meta"
        schemas_dir = staging / "schemas"

        json_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)
        schemas_dir.mkdir(parents=True, exist_ok=True)

        # Per-country files.
        index_entries: list[CountryIndexEntry] = []
        core_entries: list[CoreFeeEntry] = []
        for cc in sorted(outputs.keys()):
            output = outputs[cc]
            data = output.model_dump(mode="json")
            data["generated_at"] = self.timestamp
            content_hash = self._compute_content_sha256(data)
            # Update source metadata with deterministic hash.
            source = dict(data["source"])
            source["content_sha256"] = content_hash
            data["source"] = source
            # Write file.
            path = json_dir / f"{output.market.url_slug}.json"
            _write_json(path, data)
            index_entries.append(
                CountryIndexEntry(
                    paypal_market_code=output.market.paypal_market_code,
                    iso_country_code=output.market.iso_country_code,
                    locale=output.market.locale,
                    data_url=f"json/{output.market.url_slug}.json",
                    source_url=output.source.canonical_url or output.source.requested_url,
                    source_updated_at=output.source.page_updated_at,
                    derived_status=output.derived.status,
                    content_sha256=content_hash,
                )
            )
            core_entries.append(
                CoreFeeEntry(
                    paypal_market_code=output.market.paypal_market_code,
                    iso_country_code=output.market.iso_country_code,
                    derived_status=output.derived.status,
                    derived=output.derived,
                )
            )

        # Index and core fees.
        index = CountryIndex(schema_version=1, generated_at=self.timestamp, countries=index_entries)
        _write_json(json_dir / "index.json", index.model_dump(mode="json"))
        core_fees = CoreFees(schema_version=1, generated_at=self.timestamp, countries=core_entries)
        _write_json(json_dir / "core-fees.json", core_fees.model_dump(mode="json"))

        # Manifests and metadata.
        manifest = CountryManifest(
            schema_version=1,
            generated_at=self.timestamp,
            markets=markets,
            unsupported=unsupported,
        )
        _write_json(meta_dir / "countries.json", manifest.model_dump(mode="json"))
        _write_json(
            meta_dir / "unsupported-countries.json",
            {"schema_version": 1, "unsupported": [u.model_dump(mode="json") for u in unsupported]},
        )
        _write_json(
            meta_dir / "schema-version.json",
            SchemaVersionInfo(
                schema_version=1,
                schema_path="schemas/paypal-fees-v1.schema.json",
                schemas=[
                    "schemas/paypal-fees-v1.schema.json",
                    "schemas/core-fees-v1.schema.json",
                    "schemas/index-v1.schema.json",
                    "schemas/manifest-v1.schema.json",
                ],
                description="Initial schema for PayPal fee data",
            ).model_dump(mode="json"),
        )

        # Schemas.
        _write_json(schemas_dir / "paypal-fees-v1.schema.json", generate_country_schema())
        _write_json(schemas_dir / "core-fees-v1.schema.json", generate_core_fees_schema())
        _write_json(schemas_dir / "index-v1.schema.json", generate_index_schema())
        _write_json(schemas_dir / "manifest-v1.schema.json", generate_manifest_schema())

        # Change report. Only write it when there are changes; an empty report would
        # otherwise perturb the deterministic output tree for no-change runs. If the
        # live output already has a change report, carry it forward unchanged so the
        # managed path is not treated as removed.
        if change_report is not None:
            if change_report.changes:
                _write_json(staging / "change-report.json", change_report.model_dump(mode="json"))
            elif (self.output_dir / "change-report.json").exists():
                shutil.copy2(self.output_dir / "change-report.json", staging / "change-report.json")

        return staging != self.output_dir, staging

    def commit(self, staging: Path) -> tuple[bool, list[str]]:
        """Atomically replace managed paths with the staging tree.

        Only the paths listed in ``MANAGED_PATHS`` are touched. The output
        directory itself is never renamed, so this is safe to run at the root
        of a git repository. Staging is validated before any live path is
        modified; the live tree is validated again before backups are deleted.
        If the swap fails, the previous files are restored from the journal.
        """
        changed_files = self._list_changed_files(staging)
        if not changed_files and self._output_dir_exists_and_matches(staging):
            # No change detected; discard the staging directory.
            self.rollback(staging)
            return False, []

        # Pre-publication validation: ensure staging is a valid self-consistent
        # output tree before modifying live files.
        errors = validate_output_tree(staging)
        if errors:
            self.rollback(staging)
            raise CrawlerValidationError("Staging output failed cross-file validation:\n" + "\n".join(errors))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        journal: list[_JournalEntry] = []
        try:
            for name in self.MANAGED_PATHS:
                src = staging / name
                dst = self.output_dir / name
                backup = dst.with_name(f"{dst.name}.old")
                entry = _JournalEntry(
                    managed_name=name,
                    live_path=dst,
                    backup_path=backup,
                    staged_path=src if src.exists() else None,
                    action="none",
                    original_existed=dst.exists(),
                )

                if dst.exists() and src.exists():
                    # Replace existing live path with staged content.
                    self._remove_path(backup)
                    os.rename(dst, backup)
                    entry.action = "replaced"
                    entry.backup_created = True
                elif dst.exists() and not src.exists():
                    # Remove obsolete live path.
                    self._remove_path(backup)
                    os.rename(dst, backup)
                    entry.action = "removed"
                    entry.backup_created = True
                elif not dst.exists() and src.exists():
                    # Add a new managed path.
                    entry.action = "added"
                # else: neither exists -> nothing to do.

                if src.exists():
                    os.rename(src, dst)
                    entry.swapped = True

                journal.append(entry)

            # Final integrity validation on the live tree before deleting backups.
            final_errors = validate_output_tree(self.output_dir)
            if final_errors:
                raise CrawlerValidationError("Live output failed cross-file validation:\n" + "\n".join(final_errors))

            # Successful swap: delete backups.
            for entry in journal:
                if entry.backup_created and entry.backup_path.exists():
                    self._remove_path(entry.backup_path)

            self.rollback(staging)
            return bool(changed_files), changed_files

        except Exception as exc:
            self._rollback_live(journal)
            self.rollback(staging)
            if isinstance(exc, CrawlerValidationError):
                raise
            raise CrawlerValidationError(f"Failed to publish output: {exc}") from exc

    def _rollback_live(self, journal: list[_JournalEntry]) -> None:
        """Restore the live managed tree to its pre-transaction state.

        Rollback is performed in reverse journal order. Newly created paths are
        removed; replaced paths are restored from their backups; removed paths
        are restored from their backups.
        """
        failed: list[str] = []
        for entry in reversed(journal):
            if entry.action == "none" or not entry.swapped:
                continue

            live = entry.live_path
            backup = entry.backup_path

            if entry.action == "added":
                if live.exists() and not entry.original_existed:
                    try:
                        self._remove_path(live)
                    except Exception as exc:
                        failed.append(f"Could not remove newly created {live}: {exc}")
                continue

            # Replaced or removed: remove the new live path and restore the backup.
            try:
                if live.exists():
                    self._remove_path(live)
            except Exception as exc:
                failed.append(f"Could not remove new live path {live}: {exc}")
                continue

            if entry.backup_created and backup.exists():
                try:
                    os.rename(backup, live)
                except Exception as exc:
                    failed.append(f"Could not restore backup {backup} to {live}: {exc}")
            elif entry.original_existed and not backup.exists():
                failed.append(f"Backup missing for {live}; original state cannot be restored")

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
        """Return relative paths of managed files that differ from the published output."""
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
                    content = src_file.read_text(encoding="utf-8")
                    if not _is_same_file(dst, content):
                        changed.append(str(rel))
            else:
                rel = src.relative_to(staging)
                dst = self.output_dir / rel
                content = src.read_text(encoding="utf-8")
                if not _is_same_file(dst, content):
                    changed.append(str(rel))
        # Also detect managed files that are present in output but missing in staging.
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
        return changed

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
