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


def validate_country_output(data: dict[str, Any]) -> list[str]:
    """Validate a single per-country JSON object against the schema and plausibility rules."""
    errors: list[str] = []
    try:
        output = CountryOutput.model_validate(data)
    except ValidationError as exc:
        for error in exc.errors():
            errors.append(f"Schema validation: {error['loc']}: {error['msg']}")
        return errors

    _validate_currency_codes(data, errors)
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


def validate_file(path: Path | str, schema_type: str) -> list[str]:
    """Validate a JSON file on disk. schema_type is one of country, index, core_fees, manifest."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if schema_type == "country":
        return validate_country_output(data)
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


def validate_all_output(output_dir: Path | str) -> list[str]:
    """Validate every generated JSON file in the output directory."""
    output_dir = Path(output_dir)
    errors: list[str] = []
    for path in output_dir.glob("json/*.json"):
        if path.name in {"index.json", "core-fees.json"}:
            continue
        file_errors = validate_file(path, "country")
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
