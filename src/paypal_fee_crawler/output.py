"""Atomic, deterministic output generation for the PayPal fee crawler."""

from __future__ import annotations

import contextlib
import json
import logging
import os
import shutil
import tempfile
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
    validate_all_output,
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

        # Change report.
        if change_report is not None:
            _write_json(staging / "change-report.json", change_report.model_dump(mode="json"))

        return staging != self.output_dir, staging

    def commit(self, staging: Path) -> tuple[bool, list[str]]:
        """Atomically replace managed paths with the staging tree.

        Only the paths listed in ``MANAGED_PATHS`` are touched. The output
        directory itself is never renamed, so this is safe to run at the root
        of a git repository. Staging is validated before any live path is
        modified. If the swap fails, the previous files remain in place.
        """
        changed_files = self._list_changed_files(staging)
        if not changed_files and self._output_dir_exists_and_matches(staging):
            # No change detected; discard the staging directory.
            self.rollback(staging)
            return False, []

        # Pre-publication validation: ensure staging is a valid self-consistent
        # output tree before modifying live files.
        errors = validate_all_output(staging, schema_only=True)
        if errors:
            self.rollback(staging)
            raise CrawlerValidationError("Staging output failed validation:\n" + "\n".join(errors))

        backup_suffix = ".old"
        backup_paths: list[Path] = []
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            for name in self.MANAGED_PATHS:
                src = staging / name
                dst = self.output_dir / name
                if not src.exists():
                    # If a managed path is not in staging, remove the old one if it exists.
                    if dst.exists():
                        backup = dst.with_name(f"{dst.name}{backup_suffix}")
                        if backup.exists():
                            shutil.rmtree(backup, ignore_errors=True) if backup.is_dir() else backup.unlink(
                                missing_ok=True
                            )
                        os.rename(dst, backup)
                        backup_paths.append(backup)
                    continue

                # Move old version to backup and rename staging into place.
                if dst.exists():
                    backup = dst.with_name(f"{dst.name}{backup_suffix}")
                    if backup.exists():
                        shutil.rmtree(backup, ignore_errors=True) if backup.is_dir() else backup.unlink(missing_ok=True)
                    os.rename(dst, backup)
                    backup_paths.append(backup)
                os.rename(src, dst)

            # Successful swap: remove backups.
            for backup in backup_paths:
                if backup.exists():
                    if backup.is_dir():
                        shutil.rmtree(backup, ignore_errors=True)
                    else:
                        backup.unlink(missing_ok=True)
        except Exception:
            # On failure, attempt to restore the previous paths.
            for backup in backup_paths:
                if backup.exists():
                    live = backup.with_name(backup.name[: -len(backup_suffix)])
                    if not live.exists():
                        with contextlib.suppress(Exception):
                            os.rename(backup, live)
            raise
        finally:
            self.rollback(staging)

        return bool(changed_files), changed_files

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
