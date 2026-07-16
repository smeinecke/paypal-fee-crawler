"""Validation of crawled output and schemas."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .exceptions import ValidationError as CrawlerValidationError
from .models import (
    CoreFees,
    CountryIndex,
    CountryManifest,
    CountryOutput,
    PublicCountryOutput,
    TransactionFeeRule,
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
    """Return errors that block publication of a derived-fee result.

    In strict mode a publication-ready country must be complete, have no
    classifier diagnostics, no unclassified or ambiguous rows, and no remaining
    unknown APM methods or schedule conflicts.
    """
    errors: list[str] = []
    status = getattr(derived, "status", None)
    if status != "complete":
        errors.append(f"{label} is not complete (status={status})")
    diagnostics = getattr(derived, "diagnostics", None) or []
    if diagnostics:
        types = sorted(set(d.type for d in diagnostics))
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
        if coverage.conflicts:
            errors.append(f"{label} has {coverage.conflicts} schedule conflict(s)")
        if coverage.missing_required_schedules:
            errors.append(f"{label} has {coverage.missing_required_schedules} missing required schedule(s)")
        if coverage.inherited_schedules:
            errors.append(f"{label} has {coverage.inherited_schedules} inherited schedule(s)")
        if coverage.unresolved_references or coverage.unresolved_nested_references:
            errors.append(f"{label} has unresolved reference(s)")
        if coverage.ambiguous_identities:
            errors.append(f"{label} has {coverage.ambiguous_identities} ambiguous identity conflict(s)")
        if coverage.unsupported_fee_shapes:
            errors.append(f"{label} has {coverage.unsupported_fee_shapes} unsupported fee shape(s)")
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


def validate_country_output(data: dict[str, Any], schema_only: bool = False, strict: bool = False) -> list[str]:
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
    if not schema_only:
        _validate_table_plausibility(output, errors)
    return errors


def validate_public_country_output(
    data: dict[str, Any], schema_only: bool = False, strict: bool = False
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
    return errors


def validate_country_index(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        CountryIndex.model_validate(data)
    except ValidationError as exc:
        for error in exc.errors():
            errors.append(f"Schema validation: {error['loc']}: {error['msg']}")
    return errors


def validate_core_fees(data: dict[str, Any], strict: bool = False) -> list[str]:
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
    path: Path | str, schema_type: str, schema_only: bool = False, strict: bool = False
) -> list[str]:
    """Validate a JSON file on disk.

    ``schema_type`` is one of ``country``, ``index``, ``core_fees``, ``manifest``.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if schema_type == "country":
        return validate_public_country_output(data, schema_only=schema_only, strict=strict)
    if schema_type == "index":
        return validate_country_index(data)
    if schema_type == "core_fees":
        return validate_core_fees(data, strict=strict)
    if schema_type == "manifest":
        return validate_country_manifest(data)
    raise CrawlerValidationError(f"Unknown schema type: {schema_type}")


def generate_country_schema() -> dict[str, Any]:
    """Generate the JSON schema for per-country output."""
    schema = PublicCountryOutput.model_json_schema(mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/smeinecke/paypal-fee-data/schemas/paypal-fees-v4.schema.json"
    return schema


def generate_core_fees_schema() -> dict[str, Any]:
    """Generate the JSON schema for the consolidated core fees file."""
    schema = CoreFees.model_json_schema(mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/smeinecke/paypal-fee-data/schemas/core-fees-v4.schema.json"
    return schema


def generate_index_schema() -> dict[str, Any]:
    """Generate the JSON schema for the country index file."""
    schema = CountryIndex.model_json_schema(mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/smeinecke/paypal-fee-data/schemas/index-v4.schema.json"
    return schema


def generate_manifest_schema() -> dict[str, Any]:
    """Generate the JSON schema for the country manifest file."""
    schema = CountryManifest.model_json_schema(mode="serialization")
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/smeinecke/paypal-fee-data/schemas/manifest-v4.schema.json"
    return schema


def validate_all_output(output_dir: Path | str, schema_only: bool = False, strict: bool = False) -> list[str]:
    """Validate every generated JSON file in the output directory."""
    output_dir = Path(output_dir)
    errors: list[str] = []

    for path in output_dir.glob("json/*.json"):
        if path.name in {"index.json", "core-fees.json"}:
            continue
        file_errors = validate_file(path, "country", schema_only=schema_only, strict=strict)
        if file_errors:
            errors.append(f"{path}: " + "; ".join(file_errors))

    index_path = output_dir / "json" / "index.json"
    if index_path.exists():
        file_errors = validate_file(index_path, "index")
        if file_errors:
            errors.append(f"{index_path}: " + "; ".join(file_errors))

    core_path = output_dir / "json" / "core-fees.json"
    if core_path.exists():
        file_errors = validate_file(core_path, "core_fees", strict=strict)
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
        "paypal-fees-v4.schema.json",
        "core-fees-v4.schema.json",
        "index-v4.schema.json",
        "manifest-v4.schema.json",
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


def validate_output_tree(root: Path | str, strict: bool = False) -> list[str]:
    """Validate schemas and cross-file relationships in the generated tree.

    ``root`` may be either a staging directory containing only generated files or
    the root of the data git repository.  Validation intentionally ignores
    unrelated non-managed files such as ``.git`` and ``README.md`` when they are
    already present in the repository root.

    When ``strict`` is True, publication-level checks are applied: every country
    must be complete and have no classifier diagnostics.
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

    if strict:
        for cc, (_path, output, _raw) in country_files.items():
            errors.extend(_strict_publication_errors(output.derived, f"Country {cc}"))
        for entry in core.countries:
            errors.extend(_strict_publication_errors(entry.derived, f"Core-fee entry {entry.paypal_market_code}"))

    return errors
