"""Validation of crawled output and schemas."""

from __future__ import annotations

import contextlib
import json
import logging
import re
import subprocess  # nosec B404
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .classify import _fee_components_for_rule
from .exceptions import ValidationError as CrawlerValidationError
from .models import (
    CoreFees,
    CountryIndex,
    CountryManifest,
    CountryOutput,
    CrawlReport,
    PublicCountryOutput,
    TransactionFeeRule,
    UnsupportedCountry,
)
from .normalize import CURRENCY_CODES, normalize_decimal_string
from .regression import _country_output_hash

logger = logging.getLogger(__name__)

_MANAGED_ROOTS = {"json", "meta", "schemas", "change-report.json"}
_PROTECTED_ROOTS = {".git", ".github", "crawler"}
_PROTECTED_FILES = {"README.md", "README", "LICENSE", ".gitmodules"}


def load_json_schema(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _validate_currency_codes(data: Any, errors: list[str]) -> None:
    """Recursively check that all currency codes are valid ISO 4217."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "currency" and isinstance(value, str) and value.upper() not in CURRENCY_CODES:
                errors.append(f"Invalid currency code: {value}")
            # Fixed-fee schedules use currency codes as extra object keys.
            if (
                isinstance(value, str)
                and len(key) == 3
                and key.upper() not in CURRENCY_CODES
                and key.lower() not in {"id", "eur", "usd"}
            ):
                # Heuristic: a 3-letter uppercase key inside a schedule dict.
                pass
            _validate_currency_codes(value, errors)
    elif isinstance(data, list):
        for item in data:
            _validate_currency_codes(item, errors)


def _validate_fixed_fee_schedules(derived: Any, label: str, errors: list[str]) -> None:
    for schedule_name, schedule in derived.fixed_fee_schedules.items():
        for currency, amount in schedule.entries.items():
            if currency.upper() not in CURRENCY_CODES:
                errors.append(f"{label} fixed fee schedule {schedule_name} has invalid currency {currency}")
            try:
                normalize_decimal_string(amount)
            except ValueError:
                errors.append(f"{label} fixed fee schedule {schedule_name} has invalid amount {amount}")


def _validate_table_plausibility(output: CountryOutput, errors: list[str]) -> None:
    if not output.tables:
        errors.append("No fee tables found")
        return
    if not any(table.rows for table in output.tables):
        errors.append("No table rows found")

    has_token = any(
        token.kind in {"percentage", "money", "number"}
        for table in output.tables
        for row in table.rows
        for cell in row.cells
        for token in cell.tokens
    )
    if not has_token:
        errors.append("No pricing token or plausible fee value found")

    for table in output.tables:
        for row in table.rows:
            for cell in row.cells:
                for token in cell.tokens:
                    if token.kind == "percentage":
                        try:
                            value = normalize_decimal_string(token.value or "0")
                        except ValueError:
                            continue
                        abs_val = value[1:] if value.startswith("-") else value
                        if float(abs_val) > 100:
                            errors.append(f"Implausible percentage: {token.raw}")


def _validate_transaction_rules(derived: Any, label: str) -> list[str]:
    """Validate transaction rules for schedule references, calculability and uniqueness."""
    errors: list[str] = []
    seen_rule_keys: dict[str, TransactionFeeRule] = {}

    for rule in derived.transaction_fee_rules:
        if rule.fixed_fee_schedule and rule.fixed_fee_schedule not in derived.fixed_fee_schedules:
            errors.append(f"{label} rule {rule.id} references missing fixed-fee schedule {rule.fixed_fee_schedule}")
        if (
            rule.international_surcharge_schedule
            and rule.international_surcharge_schedule not in derived.international_surcharge_schedules
        ):
            errors.append(
                f"{label} rule {rule.id} references missing international surcharge schedule "
                f"{rule.international_surcharge_schedule}"
            )

        resolved = rule.rate_reference.resolved_rate if rule.rate_reference else None
        if resolved:
            if resolved.fixed_fee_schedule and resolved.fixed_fee_schedule not in derived.fixed_fee_schedules:
                errors.append(
                    f"{label} rule {rule.id} has unresolved nested fixed-fee reference {resolved.fixed_fee_schedule}"
                )
            if (
                resolved.international_surcharge_schedule
                and resolved.international_surcharge_schedule not in derived.international_surcharge_schedules
            ):
                errors.append(
                    f"{label} rule {rule.id} has unresolved nested international-surcharge reference "
                    f"{resolved.international_surcharge_schedule}"
                )

        has_legacy_component = bool(
            rule.percentage is not None
            or rule.fixed_fee_schedule is not None
            or rule.international_surcharge_schedule is not None
            or (rule.rate_reference and rule.rate_reference.resolved_rate is not None)
        )
        if rule.calculation_status == "calculable" and not rule.fee_components and not has_legacy_component:
            errors.append(f"{label} rule {rule.id} is marked calculable but has no fee components")

        conditions = rule.conditions or {}
        key = json.dumps(
            [
                rule.id,
                rule.variant_id,
                sorted(conditions.keys()),
                [conditions[k] for k in sorted(conditions)],
            ]
        )
        if key in seen_rule_keys:
            other = seen_rule_keys[key]
            if (
                other.percentage != rule.percentage
                or other.fixed_fee_schedule != rule.fixed_fee_schedule
                or other.international_surcharge_schedule != rule.international_surcharge_schedule
            ):
                errors.append(f"{label} rule {rule.id} has duplicate semantic identity with different fee definition")
        else:
            seen_rule_keys[key] = rule

    return errors


def _strict_publication_errors(derived: Any, label: str) -> list[str]:
    """Return blocking semantic defects that make a derived-fee result unsafe to publish.

    Strict validation catches conflicting identities, dangling references,
    invalid calculable rules, unsupported fee shapes and schedule conflicts.
    Partial or unclassified markets are intentionally allowed because they are
    still useful data; use ``_require_all_complete_errors`` to reject them.
    """
    errors: list[str] = []
    status = getattr(derived, "status", None)
    diagnostics = getattr(derived, "diagnostics", None) or []
    diagnostic_types = {d.type for d in diagnostics}
    for diagnostic_type in sorted(diagnostic_types):
        if diagnostic_type in {
            "conflicting_rule_identity",
            "ambiguous_identity",
            "unresolved_reference",
            "unsupported_fee_shape",
            "inappropriate_inheritance",
        }:
            errors.append(f"{label} has {diagnostic_type} diagnostic(s)")

    coverage = getattr(derived, "coverage_summary", None)
    if coverage:
        if coverage.conflicts:
            errors.append(f"{label} has {coverage.conflicts} schedule conflict(s)")
        if coverage.missing_required_schedules:
            errors.append(f"{label} has {coverage.missing_required_schedules} missing required schedule(s)")
        if coverage.unresolved_references or coverage.unresolved_nested_references:
            errors.append(f"{label} has unresolved reference(s)")
        if coverage.ambiguous_identities:
            errors.append(f"{label} has {coverage.ambiguous_identities} ambiguous identity conflict(s)")
        if coverage.unsupported_fee_shapes:
            errors.append(f"{label} has {coverage.unsupported_fee_shapes} unsupported fee shape(s)")
        # A market marked complete must not leave fee candidates unresolved.
        if status == "complete" and coverage.unclassified_fee_candidates:
            errors.append(
                f"{label} is complete but has {coverage.unclassified_fee_candidates} unresolved fee candidate(s)"
            )

    # Any rule that is calculable must actually carry usable fee components.
    for rule in getattr(derived, "transaction_fee_rules", []) or []:
        if rule.calculation_status == "calculable" and not _fee_components_for_rule(rule):
            errors.append(f"{label} rule {rule.id} is calculable but has no fee components")

    return errors


def _require_all_complete_errors(derived: Any, label: str) -> list[str]:
    """Return errors when a country is not fully complete and clean."""
    errors: list[str] = []
    status = getattr(derived, "status", None)
    if status != "complete":
        errors.append(f"{label} is not complete (status={status})")
    diagnostics = getattr(derived, "diagnostics", None) or []
    if diagnostics:
        types = sorted({d.type for d in diagnostics})
        errors.append(f"{label} has {len(diagnostics)} diagnostic(s): {', '.join(types)}")
    unclassified = getattr(derived, "unclassified_fee_rows", None) or []
    if unclassified:
        errors.append(f"{label} has {len(unclassified)} unclassified row(s)")
    ambiguous = getattr(derived, "ambiguous_rows", None) or []
    if ambiguous:
        errors.append(f"{label} has {len(ambiguous)} ambiguous row(s)")
    coverage = getattr(derived, "coverage_summary", None)
    if coverage:
        if coverage.unknown_apm_methods:
            errors.append(f"{label} has {coverage.unknown_apm_methods} unknown APM method(s)")
        if coverage.unclassified_fee_candidates:
            errors.append(f"{label} has {coverage.unclassified_fee_candidates} unclassified fee candidate(s)")
    return errors


def _complete_derived_errors(derived: Any, label: str) -> list[str]:
    """Return consistency errors for a derived-fee result.

    A complete result must expose at least the core commercial transaction rules
    and have corresponding fixed-fee schedules.  Schedules referenced by rules
    must exist and schedules must not contain duplicate regions or dangling
    references.
    """
    errors: list[str] = []

    if derived.status == "complete":
        core_ids = {"paypal_checkout", "goods_and_services", "other_commercial"}
        found_ids = {rule.id for rule in derived.transaction_fee_rules}
        if not core_ids & found_ids:
            errors.append(f"{label} marked complete without any core commercial transaction rule")
        if not derived.fixed_fee_schedules:
            errors.append(f"{label} marked complete without fixed-fee schedules")

    errors.extend(_validate_transaction_rules(derived, label))

    for schedule_name, schedule in derived.international_surcharge_schedules.items():
        regions = [entry.payer_region for entry in schedule.entries]
        if len(regions) != len(set(regions)):
            errors.append(f"{label} international surcharge schedule {schedule_name} has duplicate regions")
        for entry in schedule.entries:
            if not entry.payer_region:
                errors.append(f"{label} international surcharge schedule {schedule_name} has entry without region")

    # Fixed-fee schedules must not contain duplicate currency keys (fragments may
    # overlap, but a conflict would be a data-quality error).
    for schedule_name, schedule in derived.fixed_fee_schedules.items():
        if len(schedule.entries) != len(set(schedule.entries.keys())):
            errors.append(f"{label} fixed fee schedule {schedule_name} has duplicate currency keys")

    return errors


def validate_country_output(
    data: dict[str, Any],
    schema_only: bool = False,
    strict: bool = False,
    require_all_complete: bool = False,
) -> list[str]:
    """Validate an internal per-country JSON object (allows classifier fields)."""
    errors: list[str] = []
    try:
        output = CountryOutput.model_validate(data)
    except ValidationError as exc:
        for error in exc.errors():
            errors.append(f"Schema validation: {error['loc']}: {error['msg']}")
        return errors

    _validate_currency_codes(data, errors)
    errors.extend(_complete_derived_errors(output.derived, f"Country {output.market.paypal_market_code}"))
    if strict:
        errors.extend(_strict_publication_errors(output.derived, f"Country {output.market.paypal_market_code}"))
    if require_all_complete:
        errors.extend(_require_all_complete_errors(output.derived, f"Country {output.market.paypal_market_code}"))
    if not schema_only:
        _validate_table_plausibility(output, errors)
    return errors


def validate_public_country_output(
    data: dict[str, Any],
    schema_only: bool = False,
    strict: bool = False,
    require_all_complete: bool = False,
) -> list[str]:
    """Validate a public per-country JSON object (rejects internal fields)."""
    errors: list[str] = []
    try:
        output = PublicCountryOutput.model_validate(data)
    except ValidationError as exc:
        for error in exc.errors():
            errors.append(f"Schema validation: {error['loc']}: {error['msg']}")
        return errors

    if not schema_only:
        _validate_currency_codes(data, errors)
        errors.extend(_complete_derived_errors(output.derived, f"Country {output.market.paypal_market_code}"))
        if strict:
            errors.extend(_strict_publication_errors(output.derived, f"Country {output.market.paypal_market_code}"))
        if require_all_complete:
            errors.extend(_require_all_complete_errors(output.derived, f"Country {output.market.paypal_market_code}"))
    return errors


def validate_country_index(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        CountryIndex.model_validate(data)
    except ValidationError as exc:
        for error in exc.errors():
            errors.append(f"Schema validation: {error['loc']}: {error['msg']}")
    return errors


def validate_core_fees(data: dict[str, Any], strict: bool = False, require_all_complete: bool = False) -> list[str]:
    errors: list[str] = []
    try:
        core = CoreFees.model_validate(data)
    except ValidationError as exc:
        for error in exc.errors():
            errors.append(f"Schema validation: {error['loc']}: {error['msg']}")
        return errors
    for entry in core.countries:
        errors.extend(_complete_derived_errors(entry.derived, f"Core-fee entry {entry.paypal_market_code}"))
        if strict:
            errors.extend(_strict_publication_errors(entry.derived, f"Core-fee entry {entry.paypal_market_code}"))
        if require_all_complete:
            errors.extend(_require_all_complete_errors(entry.derived, f"Core-fee entry {entry.paypal_market_code}"))
    return errors


def validate_country_manifest(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        CountryManifest.model_validate(data)
    except ValidationError as exc:
        for error in exc.errors():
            errors.append(f"Schema validation: {error['loc']}: {error['msg']}")
    return errors


def validate_file(
    path: Path | str,
    schema_type: str,
    schema_only: bool = False,
    strict: bool = False,
    require_all_complete: bool = False,
) -> list[str]:
    """Validate a JSON file on disk.

    ``schema_type`` is one of ``country``, ``index``, ``core_fees``, ``manifest``.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if schema_type == "country":
        return validate_public_country_output(
            data, schema_only=schema_only, strict=strict, require_all_complete=require_all_complete
        )
    if schema_type == "index":
        return validate_country_index(data)
    if schema_type == "core_fees":
        return validate_core_fees(data, strict=strict, require_all_complete=require_all_complete)
    if schema_type == "manifest":
        return validate_country_manifest(data)
    raise CrawlerValidationError(f"Unknown schema type: {schema_type}")


def generate_country_schema() -> dict[str, Any]:
    """Generate the JSON schema for per-country output."""
    schema = PublicCountryOutput.model_json_schema(mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/smeinecke/paypal-fee-data/schemas/paypal-fees-v1.schema.json"
    return schema


def generate_core_fees_schema() -> dict[str, Any]:
    """Generate the JSON schema for the consolidated core fees file."""
    schema = CoreFees.model_json_schema(mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/smeinecke/paypal-fee-data/schemas/core-fees-v1.schema.json"
    return schema


def generate_index_schema() -> dict[str, Any]:
    """Generate the JSON schema for the country index file."""
    schema = CountryIndex.model_json_schema(mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/smeinecke/paypal-fee-data/schemas/index-v1.schema.json"
    return schema


def generate_manifest_schema() -> dict[str, Any]:
    """Generate the JSON schema for the country manifest file."""
    schema = CountryManifest.model_json_schema(mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/smeinecke/paypal-fee-data/schemas/manifest-v1.schema.json"
    return schema


def validate_all_output(
    output_dir: Path | str,
    schema_only: bool = False,
    strict: bool = False,
    require_all_complete: bool = False,
) -> list[str]:
    """Validate every generated JSON file in the output directory."""
    output_dir = Path(output_dir)
    errors: list[str] = []

    for path in output_dir.glob("json/*.json"):
        if path.name in {"index.json", "core-fees.json"}:
            continue
        file_errors = validate_file(
            path, "country", schema_only=schema_only, strict=strict, require_all_complete=require_all_complete
        )
        if file_errors:
            errors.append(f"{path}: " + "; ".join(file_errors))

    index_path = output_dir / "json" / "index.json"
    if index_path.exists():
        file_errors = validate_file(index_path, "index")
        if file_errors:
            errors.append(f"{index_path}: " + "; ".join(file_errors))

    core_path = output_dir / "json" / "core-fees.json"
    if core_path.exists():
        file_errors = validate_file(core_path, "core_fees", strict=strict, require_all_complete=require_all_complete)
        if file_errors:
            errors.append(f"{core_path}: " + "; ".join(file_errors))

    manifest_path = output_dir / "meta" / "countries.json"
    if manifest_path.exists():
        file_errors = validate_file(manifest_path, "manifest")
        if file_errors:
            errors.append(f"{manifest_path}: " + "; ".join(file_errors))

    return errors


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _iter_managed_paths(root: Path) -> list[Path]:
    paths: list[Path] = []
    for name in _MANAGED_ROOTS:
        path = root / name
        if not path.exists():
            continue
        if path.is_dir():
            paths.extend(path.rglob("*"))
        else:
            paths.append(path)
    return paths


def _validate_generated_path_safety(root: Path, errors: list[str]) -> None:
    """Validate generated paths without rejecting unrelated repo-root files."""
    for path in _iter_managed_paths(root):
        rel = path.relative_to(root)
        if path.is_symlink():
            errors.append(f"Symlink not allowed in output tree: {rel}")
        if ".." in rel.parts:
            errors.append(f"Path traversal detected: {rel}")
        first = rel.parts[0] if rel.parts else ""
        if first in _PROTECTED_ROOTS or first in _PROTECTED_FILES:
            errors.append(f"Output targets protected repository path: {rel}")


def _validate_required_files(root: Path) -> list[str]:
    required = [
        root / "json" / "index.json",
        root / "json" / "core-fees.json",
        root / "meta" / "countries.json",
        root / "meta" / "schema-version.json",
        root / "meta" / "crawl-state.json",
    ]
    return [f"Missing required file: {_safe_rel(path, root)}" for path in required if not path.exists()]


def _validate_schema_files(root: Path) -> list[str]:
    schema_files = [
        "paypal-fees-v1.schema.json",
        "core-fees-v1.schema.json",
        "index-v1.schema.json",
        "manifest-v1.schema.json",
    ]
    errors: list[str] = []
    for name in schema_files:
        if not (root / "schemas" / name).exists():
            errors.append(f"Missing schema file: schemas/{name}")
    return errors


def _load_tree_model(
    path: Path,
    model_class: type,
    label: str,
    errors: list[str],
) -> Any | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return model_class.model_validate(data)
    except Exception as exc:  # nosec B112 # noqa: S112
        errors.append(f"{label}: {exc}")
        return None


def _load_tree_models(
    root: Path,
    errors: list[str],
) -> tuple[CountryIndex | None, CoreFees | None, CountryManifest | None]:
    index = _load_tree_model(root / "json" / "index.json", CountryIndex, "json/index.json", errors)
    core = _load_tree_model(root / "json" / "core-fees.json", CoreFees, "json/core-fees.json", errors)
    manifest = _load_tree_model(root / "meta" / "countries.json", CountryManifest, "meta/countries.json", errors)
    return index, core, manifest


def _validate_manifest_duplicates(manifest: CountryManifest, errors: list[str]) -> None:
    market_codes = [m.paypal_market_code for m in manifest.markets]
    if len(market_codes) != len(set(market_codes)):
        errors.append("Duplicate PayPal market codes in manifest")
    slugs = [m.url_slug for m in manifest.markets]
    if len(slugs) != len(set(slugs)):
        errors.append("Duplicate URL slugs in manifest")


def _validate_index_manifest_consistency(index: CountryIndex, manifest: CountryManifest, errors: list[str]) -> None:
    supported = {entry.paypal_market_code for entry in index.countries}
    unsupported = {u.paypal_market_code for u in manifest.unsupported}
    if supported & unsupported:
        errors.append("Supported and unsupported market sets overlap")


def _load_country_files(
    root: Path,
    errors: list[str],
) -> dict[str, tuple[Path, PublicCountryOutput, dict[str, Any]]]:
    country_files: dict[str, tuple[Path, PublicCountryOutput, dict[str, Any]]] = {}
    for path in (root / "json").glob("*.json"):
        if path.name in {"index.json", "core-fees.json"}:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            country = PublicCountryOutput.model_validate(data)
        except Exception as exc:  # nosec B112 # noqa: S112
            errors.append(f"{path}: {exc}")
            continue
        cc = country.market.paypal_market_code
        if cc in country_files:
            errors.append(f"Duplicate country file for {cc}")
        country_files[cc] = (path, country, data)
    return country_files


def _validate_country_file(
    entry: Any,
    country_files: dict[str, tuple[Path, PublicCountryOutput, dict[str, Any]]],
    root: Path,
    errors: list[str],
) -> None:
    cc = entry.paypal_market_code
    if cc not in country_files:
        errors.append(f"Index entry {cc} has no country file")
        return
    path, country, data = country_files[cc]
    slug = cc.lower()
    expected_data_url = f"json/{slug}.json"
    if entry.data_url != expected_data_url:
        errors.append(f"Index data_url for {cc} is {entry.data_url}, expected {expected_data_url}")
    rel = path.relative_to(root)
    if str(rel) != entry.data_url:
        errors.append(f"Country file path {rel} does not match index data_url {entry.data_url}")
    if path.name != f"{slug}.json":
        errors.append(f"Filename {path.name} does not match market slug {slug}")
    if cc != country.market.paypal_market_code:
        errors.append(f"Index market code {cc} disagrees with country file {country.market.paypal_market_code}")

    expected_hash = _country_output_hash(data)
    if entry.content_sha256 != expected_hash:
        errors.append(f"Index content hash for {cc} does not match country file hash")

    country_errors = validate_public_country_output(data, schema_only=False)
    if country_errors:
        errors.append(f"{path}: " + "; ".join(country_errors))

    _validate_fixed_fee_schedules(country.derived, str(cc), errors)


def _validate_index_country_consistency(
    index: CountryIndex,
    country_files: dict[str, tuple[Path, PublicCountryOutput, dict[str, Any]]],
    root: Path,
    errors: list[str],
) -> None:
    if len(index.countries) != len(country_files):
        errors.append(f"Index lists {len(index.countries)} countries but {len(country_files)} country files exist")

    index_data_urls = {entry.data_url for entry in index.countries}
    if len(index_data_urls) != len(index.countries):
        errors.append("Duplicate data URLs in index")

    for entry in index.countries:
        _validate_country_file(entry, country_files, root, errors)


def _validate_core_fees(core: CoreFees, supported: set[str], errors: list[str]) -> None:
    core_codes = {entry.paypal_market_code for entry in core.countries}
    if core_codes - supported:
        errors.append("Core-fees file contains markets not listed in the supported index")
    if supported - core_codes:
        for cc in sorted(supported - core_codes):
            errors.append(f"Supported country {cc} missing from core-fees file")

    for entry in core.countries:
        if entry.derived_status not in {"complete", "partial", "unclassified"}:
            errors.append(f"Invalid derived status in core fees for {entry.paypal_market_code}")
        if entry.derived_status != entry.derived.status:
            errors.append(f"Core-fee status mismatch for {entry.paypal_market_code}")
        errors.extend(_complete_derived_errors(entry.derived, f"Core-fee entry {entry.paypal_market_code}"))


def _validate_supported_coverage(
    country_files: dict[str, tuple[Path, PublicCountryOutput, dict[str, Any]]],
    supported: set[str],
    errors: list[str],
) -> None:
    for cc in country_files:
        if cc not in supported:
            errors.append(f"Country file {cc} is not listed in the supported index")


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _format_dt(value: str | None) -> str:
    if not value:
        return "—"
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return value


def _derive_publication_stats(output_dir: Path) -> dict[str, str]:
    index = _load_json(output_dir / "json" / "index.json")
    countries_meta = _load_json(output_dir / "meta" / "countries.json")
    unsupported = _load_json(output_dir / "meta" / "unsupported-countries.json")

    countries = index.get("countries", [])
    total_countries = len(countries)
    status_counts: dict[str, int] = {}
    for country in countries:
        status = country.get("derived_status") or "unknown"
        status_counts[status] = status_counts.get(status, 0) + 1

    transaction_rule_count = 0
    currency_conversion_count = 0
    inherited_schedule_objects = 0
    inherited_schedule_references = 0
    rule_categories: set[str] = set()
    for country_path in (output_dir / "json").glob("*.json"):
        if country_path.name in ("index.json", "core-fees.json"):
            continue
        country_data = _load_json(country_path)
        derived = country_data.get("derived", {})
        transaction_rules = derived.get("transaction_fee_rules") or []
        transaction_rule_count += len(transaction_rules)
        for rule in transaction_rules:
            rule_categories.add(rule.get("id", "unknown"))
        if derived.get("fixed_fee_schedules"):
            rule_categories.add("fixed_fee_schedules")
        if derived.get("international_surcharge_schedules"):
            rule_categories.add("international_surcharge_schedules")
        if derived.get("currency_conversion"):
            currency_conversion_count += 1
            rule_categories.add("currency_conversion")
        coverage = derived.get("coverage_summary") or {}
        inherited_schedule_objects += coverage.get("inherited_schedule_objects", 0)
        inherited_schedule_references += coverage.get("inherited_schedule_references", 0)

    regions: set[str] = set()
    for market in countries_meta.get("markets", []):
        region = market.get("region")
        if region:
            regions.add(region)

    latest_update = None
    for country in countries:
        updated = country.get("crawled_at") or country.get("generated_at")
        if updated:
            try:
                candidate = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                if latest_update is None or candidate > latest_update:
                    latest_update = candidate
            except Exception:
                pass

    core_fees = _load_json(output_dir / "json" / "core-fees.json")
    generated_at = index.get("generated_at") or core_fees.get("generated_at")
    if generated_at and latest_update is None:
        with contextlib.suppress(Exception):
            latest_update = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))

    unsupported_count = 0
    if isinstance(unsupported, dict):
        unsupported_count = len(unsupported.get("unsupported", []))
    elif isinstance(unsupported, list):
        unsupported_count = len(unsupported)

    status_order = ["complete", "partial", "unclassified", "failed"]
    status_parts = []
    for status in status_order:
        count = status_counts.get(status, 0)
        if count:
            status_parts.append(f"{count} {status}")
    for status, count in sorted(status_counts.items()):
        if status not in status_order:
            status_parts.append(f"{count} {status}")
    status_str = ", ".join(status_parts) if status_parts else "—"

    return {
        "Countries": f"**{total_countries}**",
        "Derivation status": status_str,
        "Transaction fee rules": f"**{transaction_rule_count:,}**",
        "Currency conversion entries": f"**{currency_conversion_count:,}**",
        "Total core entries": f"**{transaction_rule_count + currency_conversion_count:,}**",
        "Inherited schedule objects": f"{inherited_schedule_objects:,}",
        "Inherited schedule references": f"{inherited_schedule_references:,}",
        "Rule categories": ", ".join(sorted(rule_categories)) or "—",
        "Regions": f"{len(regions)} ({', '.join(sorted(regions)) or '—'})",
        "Unsupported countries": str(unsupported_count),
        "Last crawled": _format_dt(latest_update.isoformat().replace("+00:00", "Z") if latest_update else None),
    }


def _validate_readme_metrics(output_dir: Path, errors: list[str]) -> None:
    readme_path = output_dir / "README.md"
    if not readme_path.exists():
        errors.append("README.md is missing")
        return
    content = readme_path.read_text(encoding="utf-8")
    match = re.search(r"<!-- STATS_START -->\n(.*?)<!-- STATS_END -->", content, re.DOTALL)
    if not match:
        errors.append("README.md does not contain STATS markers")
        return
    table = match.group(1)
    actual: dict[str, str] = {}
    for line in table.splitlines():
        if not line.startswith("| "):
            continue
        parts = [p.strip() for p in line[2:-2].split("|")]
        if len(parts) >= 2 and parts[0] not in ("Metric", "---"):
            actual[parts[0]] = parts[1]

    expected = _derive_publication_stats(output_dir)
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            errors.append(f"README.md metric '{key}' is {actual.get(key)!r}, expected {expected_value!r}")


def _crawler_submodule_revision(data_dir: Path) -> str | None:
    crawler_dir = data_dir / "crawler"
    if not (crawler_dir / ".git").exists():
        return None
    try:
        result = subprocess.run(  # nosec
            ["git", "-C", str(crawler_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception as exc:
        logger.debug("Cannot read crawler submodule revision: %s", exc)
    return None


def _is_full_git_hash(value: str) -> bool:
    if len(value) != 40:
        return False
    return all(c in "0123456789abcdef" for c in value.lower())


def _validate_crawler_revision(data_dir: Path, errors: list[str]) -> None:
    revision_path = data_dir / "meta" / "crawler-revision.json"
    if not revision_path.exists():
        errors.append("meta/crawler-revision.json is missing")
        return
    try:
        data = _load_json(revision_path)
    except Exception as exc:
        errors.append(f"meta/crawler-revision.json is malformed: {exc}")
        return
    if not isinstance(data, dict):
        errors.append("meta/crawler-revision.json must be an object")
        return
    metadata_rev = data.get("crawler_revision")
    if metadata_rev is None:
        errors.append("meta/crawler-revision.json is missing crawler_revision")
        return
    if not isinstance(metadata_rev, str) or not _is_full_git_hash(metadata_rev):
        errors.append(f"meta/crawler-revision.json crawler_revision is not a full 40-character Git hash: {metadata_rev!r}")
        return
    submodule_rev = _crawler_submodule_revision(data_dir)
    if submodule_rev is None:
        errors.append("crawler submodule is not a git checkout; cannot verify crawler-revision.json")
        return
    if metadata_rev != submodule_rev:
        errors.append(
            f"meta/crawler-revision.json crawler_revision ({metadata_rev}) does not match crawler submodule ({submodule_rev})"
        )


def _validate_change_report(data_dir: Path, errors: list[str]) -> None:
    report_path = data_dir / "change-report.json"
    if not report_path.exists():
        errors.append("change-report.json is missing")
        return
    try:
        data = _load_json(report_path)
    except Exception as exc:
        errors.append(f"change-report.json is malformed: {exc}")
        return
    if not isinstance(data, dict):
        errors.append("change-report.json must be an object")
        return
    if "has_regression" not in data:
        errors.append("change-report.json is missing has_regression")
        return
    if data["has_regression"] is not False:
        errors.append(f"change-report.json has_regression must be false, got {data['has_regression']!r}")


def _validate_crawl_report(data_dir: Path, errors: list[str]) -> None:
    report_path = data_dir / "meta" / "crawl-report.json"
    if not report_path.exists():
        errors.append("meta/crawl-report.json is missing")
        return
    try:
        data = _load_json(report_path)
    except Exception as exc:
        errors.append(f"meta/crawl-report.json is malformed: {exc}")
        return
    if not isinstance(data, dict):
        errors.append("meta/crawl-report.json must be an object")
        return
    try:
        report = CrawlReport.model_validate(data)
    except Exception as exc:
        errors.append(f"meta/crawl-report.json is not a valid CrawlReport: {exc}")
        return
    if report.exit_code != 0:
        errors.append(f"meta/crawl-report.json exit_code is not 0: {report.exit_code}")


def _validate_completeness(data_dir: Path, errors: list[str]) -> None:
    manifest_path = data_dir / "meta" / "countries.json"
    index_path = data_dir / "json" / "index.json"
    unsupported_path = data_dir / "meta" / "unsupported-countries.json"
    transient_path = data_dir / "meta" / "transient-failures.json"

    manifest: CountryManifest | None = None
    if manifest_path.exists():
        try:
            manifest = CountryManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"meta/countries.json: cannot validate: {exc}")
            return
    else:
        errors.append("meta/countries.json is missing")
        return

    index: CountryIndex | None = None
    if index_path.exists():
        try:
            index = CountryIndex.model_validate_json(index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"json/index.json: cannot validate: {exc}")

    unsupported: list[UnsupportedCountry] = []
    if unsupported_path.exists():
        try:
            data = _load_json(unsupported_path)
            items = data.get("unsupported", []) if isinstance(data, dict) else data
            unsupported = [UnsupportedCountry.model_validate(item) for item in items]
        except Exception as exc:
            errors.append(f"meta/unsupported-countries.json: cannot validate: {exc}")

    transient: list[UnsupportedCountry] = []
    if transient_path.exists():
        try:
            data = _load_json(transient_path)
            items = data.get("transient_failures", []) if isinstance(data, dict) else data
            transient = [UnsupportedCountry.model_validate(item) for item in items]
        except Exception as exc:
            errors.append(f"meta/transient-failures.json: cannot validate: {exc}")

    discovered: set[str] = set()
    if manifest:
        for market in manifest.markets:
            if market.paypal_market_code:
                discovered.add(market.paypal_market_code)
        for item in manifest.unsupported:
            if item.paypal_market_code:
                discovered.add(item.paypal_market_code)
        for item in manifest.transient_failures:
            if item.paypal_market_code:
                discovered.add(item.paypal_market_code)

    supported = {entry.paypal_market_code for entry in (index.countries if index else [])}
    unsupported_set = {u.paypal_market_code for u in unsupported if u.paypal_market_code}
    transient_set = {t.paypal_market_code for t in transient if t.paypal_market_code}

    for cc in sorted(discovered):
        states: list[str] = []
        if cc in supported:
            states.append("supported")
        if cc in unsupported_set:
            states.append("unsupported")
        if cc in transient_set:
            states.append("transient")
        if len(states) != 1:
            errors.append(f"{cc}: expected exactly one state, got {states}")

    for cc in sorted(supported):
        if cc not in discovered:
            errors.append(f"{cc}: supported country {cc} is not in the discovered market set")
    for cc in sorted(unsupported_set):
        if cc not in discovered:
            errors.append(f"{cc}: unsupported country {cc} is not in the discovered market set")
    for cc in sorted(transient_set):
        if cc not in discovered:
            errors.append(f"{cc}: transient country {cc} is not in the discovered market set")


def validate_output_tree(
    root: Path | str,
    strict: bool = False,
    require_all_complete: bool = False,
) -> list[str]:
    """Validate schemas and cross-file relationships in the generated tree.

    ``root`` may be either a staging directory containing only generated files or
    the root of the data git repository.  Validation intentionally ignores
    unrelated non-managed files such as ``.git`` and ``README.md`` when they are
    already present in the repository root.

    When ``strict`` is True, blocking semantic checks are applied (conflicting
    identities, dangling references, invalid calculable rules, unsupported fee
    shapes). When ``require_all_complete`` is True, every country must be
    complete and have no classifier diagnostics or unresolved candidates.
    """
    root = Path(root)
    errors: list[str] = []

    errors.extend(_validate_required_files(root))
    errors.extend(_validate_schema_files(root))
    _validate_generated_path_safety(root, errors)

    if errors:
        return errors

    index, core, manifest = _load_tree_models(root, errors)
    if index is None or core is None or manifest is None:
        return errors

    _validate_index_manifest_consistency(index, manifest, errors)
    _validate_manifest_duplicates(manifest, errors)

    supported = {entry.paypal_market_code for entry in index.countries}
    country_files = _load_country_files(root, errors)
    _validate_index_country_consistency(index, country_files, root, errors)
    _validate_core_fees(core, supported, errors)
    _validate_supported_coverage(country_files, supported, errors)

    if strict or require_all_complete:
        for cc, (_path, output, _raw) in country_files.items():
            if strict:
                errors.extend(_strict_publication_errors(output.derived, f"Country {cc}"))
            if require_all_complete:
                errors.extend(_require_all_complete_errors(output.derived, f"Country {cc}"))
        for entry in core.countries:
            if strict:
                errors.extend(_strict_publication_errors(entry.derived, f"Core-fee entry {entry.paypal_market_code}"))
            if require_all_complete:
                errors.extend(_require_all_complete_errors(entry.derived, f"Core-fee entry {entry.paypal_market_code}"))

    # Strict publication-readiness checks.
    if strict:
        _validate_change_report(root, errors)
        _validate_crawl_report(root, errors)
        _validate_crawler_revision(root, errors)
        _validate_readme_metrics(root, errors)

    if require_all_complete:
        _validate_completeness(root, errors)

    return errors
