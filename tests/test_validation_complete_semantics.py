from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paypal_fee_crawler.regression import _country_output_hash
from paypal_fee_crawler.validation import validate_output_tree, validate_public_country_output


def _country(derived: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "generated_at": None,
        "market": {
            "paypal_market_code": "DE",
            "iso_country_code": "DE",
            "country_name": "Germany",
            "region": "europe",
            "locale": "de_DE",
            "languages": [],
        },
        "source": {
            "requested_url": "https://www.paypal.com/de/business/paypal-business-fees",
            "canonical_url": "https://www.paypal.com/de/business/paypal-business-fees",
        },
        "sections": [],
        "tables": [
            {
                "document_id": "FEETB16",
                "headers": [],
                "rows": [
                    {
                        "cells": [
                            {
                                "text": "2,99% + Festgebühr",
                                "tokens": [{"raw": "2,99%", "kind": "percentage", "value": "2.99"}],
                                "links": [],
                            }
                        ]
                    }
                ],
            }
        ],
        "derived": derived,
        "warnings": [],
    }


def _complete_derived() -> dict[str, Any]:
    return {
        "status": "complete",
        "standard_commercial": {"percentage": "2.99"},
        "commercial_fixed_fees": [
            {"currency": "EUR", "amount": "0.39"},
            {"currency": "USD", "amount": "0.49"},
        ],
        "international_surcharges": [
            {"region": "EEA", "percentage_points": "0"},
            {"region": "GB", "percentage_points": "1.29"},
            {"region": "OTHER", "percentage_points": "1.99"},
        ],
        "currency_conversion": {"spread_percentage": "3"},
        "international_surcharge_exposed": True,
        "currency_conversion_exposed": True,
    }


def test_complete_country_requires_international_and_fx_categories_when_exposed() -> None:
    derived = _complete_derived()
    derived["international_surcharges"] = []
    derived["currency_conversion"] = None
    derived["international_surcharge_exposed"] = True
    derived["currency_conversion_exposed"] = True

    errors = validate_public_country_output(_country(derived), schema_only=False)

    assert any("international surcharges" in error for error in errors)
    assert any("currency conversion" in error for error in errors)


def test_complete_country_accepts_missing_unexposed_categories() -> None:
    derived = _complete_derived()
    derived["international_surcharges"] = []
    derived["currency_conversion"] = None
    derived["international_surcharge_exposed"] = False
    derived["currency_conversion_exposed"] = False

    errors = validate_public_country_output(_country(derived), schema_only=False)

    assert not errors


def test_complete_country_accepts_all_required_categories() -> None:
    errors = validate_public_country_output(_country(_complete_derived()), schema_only=False)
    assert not errors


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_minimal_tree(root: Path, country: dict[str, Any]) -> None:
    country_hash = _country_output_hash(country)
    country["source"]["artifact_sha256"] = country_hash

    _write_json(root / "json" / "de.json", country)
    _write_json(
        root / "json" / "index.json",
        {
            "schema_version": 2,
            "generated_at": None,
            "countries": [
                {
                    "paypal_market_code": "DE",
                    "iso_country_code": "DE",
                    "locale": "de_DE",
                    "data_url": "json/de.json",
                    "source_url": "https://www.paypal.com/de/business/paypal-business-fees",
                    "source_updated_at": None,
                    "derived_status": country["derived"]["status"],
                    "content_sha256": country_hash,
                }
            ],
        },
    )
    _write_json(
        root / "json" / "core-fees.json",
        {
            "schema_version": 2,
            "generated_at": None,
            "countries": [
                {
                    "paypal_market_code": "DE",
                    "iso_country_code": "DE",
                    "derived_status": country["derived"]["status"],
                    "derived": country["derived"],
                }
            ],
        },
    )
    _write_json(
        root / "meta" / "countries.json",
        {
            "schema_version": 2,
            "generated_at": None,
            "markets": [
                {
                    "paypal_market_code": "DE",
                    "iso_country_code": "DE",
                    "country_name": "Germany",
                    "region": "europe",
                    "locale": "de_DE",
                    "languages": [],
                }
            ],
            "unsupported": [],
            "fee_page_urls": {},
        },
    )
    _write_json(root / "meta" / "schema-version.json", {"schema_version": 2, "schema_path": "schemas/paypal-fees-v2.schema.json"})
    for schema_name in [
        "paypal-fees-v2.schema.json",
        "core-fees-v2.schema.json",
        "index-v2.schema.json",
        "manifest-v2.schema.json",
    ]:
        _write_json(root / "schemas" / schema_name, {"type": "object"})


def test_output_tree_ignores_existing_repo_root_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / ".git").mkdir(parents=True)
    (root / ".git" / "sentinel").write_text("git", encoding="utf-8")
    (root / ".github" / "workflows").mkdir(parents=True)
    (root / ".github" / "workflows" / "update.yml").write_text("workflow", encoding="utf-8")
    (root / "README.md").write_text("readme", encoding="utf-8")
    (root / "LICENSE").write_text("license", encoding="utf-8")
    (root / "crawler").mkdir()
    (root / "crawler" / "sentinel").write_text("crawler", encoding="utf-8")

    _write_minimal_tree(root, _country(_complete_derived()))

    assert validate_output_tree(root) == []


def test_output_tree_rejects_complete_without_required_categories(tmp_path: Path) -> None:
    country = _country(_complete_derived())
    country["derived"]["currency_conversion"] = None
    country["derived"]["currency_conversion_exposed"] = True
    _write_minimal_tree(tmp_path, country)

    errors = validate_output_tree(tmp_path)

    assert any("currency conversion" in error for error in errors)
