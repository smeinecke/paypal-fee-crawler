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
)
from .normalize import CURRENCY_CODES, normalize_decimal_string
from .regression import _country_output_hash

logger = logging.getLogger(__name__)


def load_json_schema(path: Path | str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _validate_currency_codes(data: Any, errors: list[str]) -> None:
    """Recursively check that all currency codes are valid ISO 4217."""
    if isinstance(data, dict):
        for key, value in data.items():
            if key == "currency" and isinstance(value, str) and value.upper() not in CURRENCY_CODES:
                errors.append(f"Invalid currency code: {value}")
            _validate_currency_codes(value, errors)
    elif isinstance(data, list):
        for item in data:
            _validate_currency_codes(item, errors)


def _validate_table_plausibility(output: CountryOutput, errors: list[str]) -> None:
    if not output.tables:
        errors.append("No fee tables found")
        return
    if not any(table.rows for table in output.tables):
        errors.append("No table rows found")
    # Check for at least one pricing token or plausible fee value.
    has_token = any(
        token.kind in {"percentage", "money", "number"}
        for table in output.tables
        for row in table.rows
        for cell in row.cells
        for token in cell.tokens
    )
    if not has_token:
        errors.append("No pricing token or plausible fee value found")
    # Check percentage limits (0-100, with a small tolerance for negative adjustments).
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


def validate_country_output(data: dict[str, Any], schema_only: bool = False) -> list[str]:
    """Validate a single per-country JSON object against the schema and plausibility rules."""
    errors: list[str] = []
    try:
        output = CountryOutput.model_validate(data)
    except ValidationError as exc:
        for error in exc.errors():
            errors.append(f"Schema validation: {error['loc']}: {error['msg']}")
        return errors

    _validate_currency_codes(data, errors)
    if not schema_only:
        _validate_table_plausibility(output, errors)
    return errors


def validate_country_index(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        CountryIndex.model_validate(data)
    except ValidationError as exc:
        for error in exc.errors():
            errors.append(f"Schema validation: {error['loc']}: {error['msg']}")
    return errors


def validate_core_fees(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        CoreFees.model_validate(data)
    except ValidationError as exc:
        for error in exc.errors():
            errors.append(f"Schema validation: {error['loc']}: {error['msg']}")
    return errors


def validate_country_manifest(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    try:
        CountryManifest.model_validate(data)
    except ValidationError as exc:
        for error in exc.errors():
            errors.append(f"Schema validation: {error['loc']}: {error['msg']}")
    return errors


def validate_file(path: Path | str, schema_type: str, schema_only: bool = False) -> list[str]:
    """Validate a JSON file on disk. schema_type is one of country, index, core_fees, manifest."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if schema_type == "country":
        return validate_country_output(data, schema_only=schema_only)
    if schema_type == "index":
        return validate_country_index(data)
    if schema_type == "core_fees":
        return validate_core_fees(data)
    if schema_type == "manifest":
        return validate_country_manifest(data)
    raise CrawlerValidationError(f"Unknown schema type: {schema_type}")


def generate_country_schema() -> dict[str, Any]:
    """Generate the JSON schema for per-country output."""
    schema = CountryOutput.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/smeinecke/paypal-fee-data/schemas/paypal-fees-v1.schema.json"
    return schema


def generate_core_fees_schema() -> dict[str, Any]:
    """Generate the JSON schema for the consolidated core fees file."""
    schema = CoreFees.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/smeinecke/paypal-fee-data/schemas/core-fees-v1.schema.json"
    return schema


def generate_index_schema() -> dict[str, Any]:
    """Generate the JSON schema for the country index file."""
    schema = CountryIndex.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/smeinecke/paypal-fee-data/schemas/index-v1.schema.json"
    return schema


def generate_manifest_schema() -> dict[str, Any]:
    """Generate the JSON schema for the country manifest file."""
    schema = CountryManifest.model_json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/smeinecke/paypal-fee-data/schemas/manifest-v1.schema.json"
    return schema


def validate_all_output(output_dir: Path | str, schema_only: bool = False) -> list[str]:
    """Validate every generated JSON file in the output directory."""
    output_dir = Path(output_dir)
    errors: list[str] = []
    for path in output_dir.glob("json/*.json"):
        if path.name in {"index.json", "core-fees.json"}:
            continue
        file_errors = validate_file(path, "country", schema_only=schema_only)
        if file_errors:
            errors.append(f"{path}: " + "; ".join(file_errors))
    index_path = output_dir / "json" / "index.json"
    if index_path.exists():
        file_errors = validate_file(index_path, "index")
        if file_errors:
            errors.append(f"{index_path}: " + "; ".join(file_errors))
    core_path = output_dir / "json" / "core-fees.json"
    if core_path.exists():
        file_errors = validate_file(core_path, "core_fees")
        if file_errors:
            errors.append(f"{core_path}: " + "; ".join(file_errors))
    manifest_path = output_dir / "meta" / "countries.json"
    if manifest_path.exists():
        file_errors = validate_file(manifest_path, "manifest")
        if file_errors:
            errors.append(f"{manifest_path}: " + "; ".join(file_errors))
    return errors


def validate_output_tree(root: Path | str) -> list[str]:
    """Validate both schemas and cross-file relationships in the output tree.

    This is the gate used before atomic publication. It returns a list of human-
    readable error strings; an empty list means the tree is internally consistent.
    """
    root = Path(root)
    errors: list[str] = []

    required = [
        root / "json" / "index.json",
        root / "json" / "core-fees.json",
        root / "meta" / "countries.json",
        root / "meta" / "schema-version.json",
    ]
    for path in required:
        if not path.exists():
            errors.append(f"Missing required file: {path.relative_to(root)}")

    schema_files = [
        "paypal-fees-v1.schema.json",
        "core-fees-v1.schema.json",
        "index-v1.schema.json",
        "manifest-v1.schema.json",
    ]
    for name in schema_files:
        if not (root / "schemas" / name).exists():
            errors.append(f"Missing schema file: schemas/{name}")

    if errors:
        return errors

    try:
        index_data = json.loads((root / "json" / "index.json").read_text(encoding="utf-8"))
        index = CountryIndex.model_validate(index_data)
    except Exception as exc:  # nosec B112 # noqa: S112
        errors.append(f"json/index.json: {exc}")
        return errors

    try:
        core_data = json.loads((root / "json" / "core-fees.json").read_text(encoding="utf-8"))
        core = CoreFees.model_validate(core_data)
    except Exception as exc:  # nosec B112 # noqa: S112
        errors.append(f"json/core-fees.json: {exc}")
        return errors

    try:
        manifest_data = json.loads((root / "meta" / "countries.json").read_text(encoding="utf-8"))
        manifest = CountryManifest.model_validate(manifest_data)
    except Exception as exc:  # nosec B112 # noqa: S112
        errors.append(f"meta/countries.json: {exc}")
        return errors

    supported = {entry.paypal_market_code for entry in index.countries}
    unsupported = {u.paypal_market_code for u in manifest.unsupported}
    if supported & unsupported:
        errors.append("Supported and unsupported market sets overlap")

    market_codes = [m.paypal_market_code for m in manifest.markets]
    if len(market_codes) != len(set(market_codes)):
        errors.append("Duplicate PayPal market codes in manifest")
    slugs = [m.url_slug for m in manifest.markets]
    if len(slugs) != len(set(slugs)):
        errors.append("Duplicate URL slugs in manifest")

    country_files: dict[str, tuple[Path, CountryOutput, dict[str, Any]]] = {}
    for path in (root / "json").glob("*.json"):
        if path.name in {"index.json", "core-fees.json"}:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            country = CountryOutput.model_validate(data)
        except Exception as exc:  # nosec B112 # noqa: S112
            errors.append(f"{path}: {exc}")
            continue
        cc = country.market.paypal_market_code
        if cc in country_files:
            errors.append(f"Duplicate country file for {cc}")
        country_files[cc] = (path, country, data)

    if len(index.countries) != len(country_files):
        errors.append(f"Index lists {len(index.countries)} countries but {len(country_files)} country files exist")

    index_data_urls = {entry.data_url for entry in index.countries}
    if len(index_data_urls) != len(index.countries):
        errors.append("Duplicate data URLs in index")

    for entry in index.countries:
        cc = entry.paypal_market_code
        if cc not in country_files:
            errors.append(f"Index entry {cc} has no country file")
            continue
        path, country, data = country_files[cc]
        expected_data_url = f"json/{country.market.url_slug}.json"
        if entry.data_url != expected_data_url:
            errors.append(f"Index data_url for {cc} is {entry.data_url}, expected {expected_data_url}")
        rel = path.relative_to(root)
        if str(rel) != entry.data_url:
            errors.append(f"Country file path {rel} does not match index data_url {entry.data_url}")
        if path.name != f"{country.market.url_slug}.json":
            errors.append(f"Filename {path.name} does not match market slug {country.market.url_slug}")
        if cc != country.market.paypal_market_code:
            errors.append(f"Index market code {cc} disagrees with country file {country.market.paypal_market_code}")
        if not country.tables:
            errors.append(f"Country {cc} has no fee tables")
        expected_hash = _country_output_hash(data)
        if entry.content_sha256 != expected_hash:
            errors.append(f"Index content hash for {cc} does not match country file hash")
        country_errors = validate_country_output(data, schema_only=False)
        if country_errors:
            errors.append(f"{path}: " + "; ".join(country_errors))
        currencies = [fee.currency for fee in country.derived.commercial_fixed_fees]
        if len(currencies) != len(set(currencies)):
            errors.append(f"Duplicate fixed-fee currency entries for {cc}")
        if country.derived.status == "complete" and (
            not country.derived.standard_commercial or not country.derived.commercial_fixed_fees
        ):
            errors.append(f"Country {cc} marked complete without standard commercial and fixed fees")

    core_codes = {entry.paypal_market_code for entry in core.countries}
    if core_codes - supported:
        errors.append("Core-fees file contains markets not listed in the supported index")
    if supported - core_codes:
        for cc in sorted(supported - core_codes):
            errors.append(f"Supported country {cc} missing from core-fees file")
    for entry in core.countries:
        if entry.derived_status not in {"complete", "partial", "unclassified"}:
            errors.append(f"Invalid derived status in core fees for {entry.paypal_market_code}")
        if entry.derived_status == "complete" and (
            not entry.derived.standard_commercial or not entry.derived.commercial_fixed_fees
        ):
            errors.append(f"Core-fee entry {entry.paypal_market_code} marked complete without required categories")

    for cc in country_files:
        if cc not in supported:
            errors.append(f"Country file {cc} is not listed in the supported index")

    for path in root.rglob("*"):
        if path.is_symlink():
            errors.append(f"Symlink not allowed in output tree: {path.relative_to(root)}")
        rel = path.relative_to(root)
        if ".." in rel.parts:
            errors.append(f"Path traversal detected: {rel}")
        first = rel.parts[0] if rel.parts else ""
        if first in {".git", ".github"} or first in {"README", "LICENSE"}:
            errors.append(f"Output targets protected repository path: {rel}")

    return errors
