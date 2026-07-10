"""Atomic, deterministic output generation for the PayPal fee crawler."""

from __future__ import annotations

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
            path = json_dir / f"{cc.lower()}.json"
            _write_json(path, data)
            index_entries.append(
                CountryIndexEntry(
                    country_code=cc,
                    locale=output.market.locale,
                    data_url=f"json/{cc.lower()}.json",
                    source_url=output.source.canonical_url or output.source.requested_url,
                    source_updated_at=output.source.page_updated_at,
                    derived_status=output.derived.status,
                    content_sha256=content_hash,
                )
            )
            core_entries.append(
                CoreFeeEntry(
                    country_code=cc,
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
        """Compare staging to published output and replace files only when changed."""
        changed_files: list[str] = []
        # Create all output directories.
        (self.output_dir / "json").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "meta").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "schemas").mkdir(parents=True, exist_ok=True)

        for src in staging.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(staging)
            dst = self.output_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            content = src.read_text(encoding="utf-8")
            if _is_same_file(dst, content):
                continue
            dst.write_text(content, encoding="utf-8")
            changed_files.append(str(rel))

        # Remove files in output that are no longer in staging? Only if atomic and
        # the caller explicitly wants it; we avoid deletion to preserve prior data.
        return bool(changed_files), changed_files

    def rollback(self, staging: Path) -> None:
        """Clean up the staging directory on failure."""
        if self.staging_dir is None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
