from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from paypal_fee_crawler.models import CrawlReport
from paypal_fee_crawler.regression import _country_output_hash
from paypal_fee_crawler.validation import (
    _derive_publication_stats,
    validate_output_tree,
    validate_public_country_output,
)


def _git_init_with_commit(path: Path) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Test",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    subprocess.run(["git", "init", "-q", str(path)], check=True, env=env)
    subprocess.run(["git", "-C", str(path), "commit", "--allow-empty", "-m", "init", "-q"], check=True, env=env)
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return result.stdout.strip()


def _country(derived: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": None,
        "market": {
            "paypal_market_code": "DE",
            "iso_country_code": "DE",
            "country_name": "Germany",
            "locale": "de_DE",
        },
        "derived": derived,
    }


def _complete_derived() -> dict[str, Any]:
    return {
        "status": "complete",
        "transaction_fee_rules": [
            {
                "id": "paypal_checkout",
                "label": "PayPal Checkout",
                "percentage": "2.99",
                "fixed_fee_schedule": "commercial",
            },
            {
                "id": "goods_and_services",
                "label": "Goods and Services",
                "percentage": "2.49",
                "fixed_fee_schedule": "goods_and_services",
            },
        ],
        "fixed_fee_schedules": {
            "commercial": {"entries": {"EUR": "0.39", "USD": "0.49"}},
            "goods_and_services": {"entries": {"EUR": "0.35"}},
        },
        "international_surcharge_schedules": {
            "commercial": {
                "entries": [
                    {"payer_region": "EEA", "percentage_points": "0"},
                    {"payer_region": "GB", "percentage_points": "1.29"},
                    {"payer_region": "OTHER", "percentage_points": "1.99"},
                ]
            }
        },
        "currency_conversion": {"spread_percentage": "3"},
    }


def test_complete_country_requires_core_rules_and_fixed_schedules() -> None:
    derived = _complete_derived()
    derived["transaction_fee_rules"] = []
    derived["fixed_fee_schedules"] = {}

    errors = validate_public_country_output(_country(derived), schema_only=False)
    assert any("core commercial" in error.lower() for error in errors)
    assert any("fixed-fee" in error.lower() for error in errors)


def test_complete_country_accepts_missing_currency_conversion() -> None:
    derived = _complete_derived()
    derived["currency_conversion"] = None

    errors = validate_public_country_output(_country(derived), schema_only=False)
    assert not errors


def test_complete_country_accepts_all_required_categories() -> None:
    errors = validate_public_country_output(_country(_complete_derived()), schema_only=False)
    assert not errors


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _render_readme_table(stats: dict[str, str]) -> str:
    lines = ["| Metric | Value |", "|--------|------:|"]
    for key, value in stats.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    return "\n".join(lines)


def _write_minimal_tree(root: Path, country: dict[str, Any]) -> None:
    crawler_dir = root / "crawler"
    crawler_rev = _git_init_with_commit(crawler_dir)
    country_hash = _country_output_hash(country)

    _write_json(root / "json" / "de.json", country)
    _write_json(
        root / "json" / "index.json",
        {
            "schema_version": 1,
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
    # core-fees.json uses a compact derived model that omits diagnostics and
    # coverage summary, so strip those for the core copy.
    core_allowed = {
        "status",
        "transaction_fee_rules",
        "fixed_fee_schedules",
        "international_surcharge_schedules",
        "maximum_fee_schedules",
        "currency_conversion",
    }
    core_derived = {k: v for k, v in country["derived"].items() if k in core_allowed}
    _write_json(
        root / "json" / "core-fees.json",
        {
            "schema_version": 1,
            "generated_at": None,
            "countries": [
                {
                    "paypal_market_code": "DE",
                    "iso_country_code": "DE",
                    "derived_status": country["derived"]["status"],
                    "derived": core_derived,
                }
            ],
        },
    )
    _write_json(
        root / "meta" / "countries.json",
        {
            "schema_version": 1,
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
            "transient_failures": [],
            "fee_page_urls": {},
        },
    )
    _write_json(
        root / "meta" / "unsupported-countries.json",
        {"schema_version": 1, "unsupported": []},
    )
    _write_json(
        root / "meta" / "transient-failures.json",
        {"schema_version": 1, "transient_failures": []},
    )
    _write_json(
        root / "meta" / "schema-version.json",
        {
            "schema_version": 1,
            "schema_path": "schemas/paypal-fees-v1.schema.json",
            "schemas": [
                "schemas/paypal-fees-v1.schema.json",
                "schemas/core-fees-v1.schema.json",
                "schemas/index-v1.schema.json",
                "schemas/manifest-v1.schema.json",
            ],
        },
    )
    _write_json(
        root / "meta" / "crawl-state.json",
        {"schema_version": 1, "generated_at": None, "markets": {}},
    )
    _write_json(
        root / "change-report.json",
        {"schema_version": 1, "changes": [], "generated_at": None, "has_regression": False},
    )
    _write_json(
        root / "meta" / "crawl-report.json",
        CrawlReport(exit_code=0, changed=False, countries_processed=1).model_dump(mode="json"),
    )
    _write_json(
        root / "meta" / "crawler-revision.json",
        {"crawler_revision": crawler_rev, "generated_at": None},
    )
    for schema_name in [
        "paypal-fees-v1.schema.json",
        "core-fees-v1.schema.json",
        "index-v1.schema.json",
        "manifest-v1.schema.json",
    ]:
        _write_json(root / "schemas" / schema_name, {"type": "object"})

    # Generate a README whose metrics match the staged data so strict validation
    # does not fail for fixture trees that are otherwise valid.
    stats = _derive_publication_stats(root)
    readme = f"# Test data\n\n## Statistics\n\n<!-- STATS_START -->\n{_render_readme_table(stats)}<!-- STATS_END -->\n"
    (root / "README.md").write_text(readme, encoding="utf-8")


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
    country["derived"]["transaction_fee_rules"] = []
    country["derived"]["fixed_fee_schedules"] = {}
    _write_minimal_tree(tmp_path, country)

    errors = validate_output_tree(tmp_path)
    assert any("core commercial" in error.lower() for error in errors)


def test_strict_allows_partial_market_with_unclassified_candidates(tmp_path: Path) -> None:
    """Strict semantic validation permits partial markets that have no conflicts."""
    derived = _complete_derived()
    derived["status"] = "partial"
    derived["coverage_summary"] = {"unclassified_fee_candidates": 3}
    _write_minimal_tree(tmp_path, _country(derived))

    assert validate_output_tree(tmp_path, strict=True) == []


def test_require_all_complete_rejects_partial_markets(tmp_path: Path) -> None:
    """--require-all-complete rejects intentionally partial markets."""
    derived = _complete_derived()
    derived["status"] = "partial"
    _write_minimal_tree(tmp_path, _country(derived))

    errors = validate_output_tree(tmp_path, require_all_complete=True)
    assert any("not complete" in error.lower() for error in errors)


def test_strict_rejects_conflicting_rule_identities(tmp_path: Path) -> None:
    """Strict validation fails when the same identity carries different fees."""
    derived = _complete_derived()
    derived["diagnostics"] = [{"type": "conflicting_rule_identity", "message": "conflict"}]
    _write_minimal_tree(tmp_path, _country(derived))

    errors = validate_output_tree(tmp_path, strict=True)
    assert any("conflicting_rule_identity" in error for error in errors)


def test_strict_rejects_unresolved_reference(tmp_path: Path) -> None:
    """Strict validation fails when a reference cannot be resolved."""
    derived = _complete_derived()
    derived["coverage_summary"] = {
        "unresolved_references": 1,
        "unresolved_nested_references": 0,
        "unclassified_fee_candidates": 0,
    }
    _write_minimal_tree(tmp_path, _country(derived))

    errors = validate_output_tree(tmp_path, strict=True)
    assert any("unresolved reference" in error.lower() for error in errors)


def test_strict_rejects_regression_report(tmp_path: Path) -> None:
    """A change report with regressions makes the tree not publication-ready."""
    _write_minimal_tree(tmp_path, _country(_complete_derived()))
    _write_json(
        tmp_path / "change-report.json",
        {"schema_version": 1, "changes": [], "has_regression": True},
    )

    errors = validate_output_tree(tmp_path, strict=True)
    assert any("regression" in error.lower() for error in errors)
