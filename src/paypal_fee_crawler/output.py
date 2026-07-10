"""Atomic, deterministic output generation for the PayPal fee crawler."""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

    def __init__(
        self,
        output_dir: Path | str,
        staging_dir: Path | str | None = None,
        timestamp: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.staging_dir = Path(staging_dir) if staging_dir else None
        self.timestamp = timestamp or datetime.now(timezone.utc).replace(microsecond=0).isoformat()  # noqa: UP017

    def _make_staging(self) -> Path:
        if self.staging_dir:
            self.staging_dir.mkdir(parents=True, exist_ok=True)
            return self.staging_dir
        return Path(tempfile.mkdtemp(prefix="paypal-fee-crawler-"))

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
        """Atomically replace published output with the staging tree.

        Uses a directory swap so that the public directory is always in a
        consistent state: either the old output or the new output, never a
        partial mix. If the swap fails, the previous output remains in place.
        """
        changed_files = self._list_changed_files(staging)
        if not changed_files and self._output_dir_exists_and_matches(staging):
            # No change detected; discard the staging directory.
            self.rollback(staging)
            return False, []

        backup_dir = self.output_dir.with_name(f"{self.output_dir.name}.old")
        # Remove any stale backup from a previous aborted run.
        if backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)

        try:
            if self.output_dir.exists():
                self.output_dir.rename(backup_dir)
            staging.rename(self.output_dir)
            # Successful swap: remove the backup.
            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)
        except Exception:
            # On failure, attempt to restore the previous directory.
            if backup_dir.exists() and not self.output_dir.exists():
                with contextlib.suppress(Exception):
                    backup_dir.rename(self.output_dir)
            raise

        self.rollback(staging)
        return bool(changed_files), changed_files

    def _list_changed_files(self, staging: Path) -> list[str]:
        """Return relative paths of files that differ from the published output."""
        changed: list[str] = []
        for src in staging.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(staging)
            dst = self.output_dir / rel
            content = src.read_text(encoding="utf-8")
            if not _is_same_file(dst, content):
                changed.append(str(rel))
        # Also detect files that are present in output but missing in staging.
        if self.output_dir.exists():
            for dst in self.output_dir.rglob("*"):
                if not dst.is_file():
                    continue
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
        if self.staging_dir is None and staging.exists() and staging != self.output_dir:
            shutil.rmtree(staging, ignore_errors=True)
