"""Tests for publication-readiness checks: crawler revision, reports, and README sync."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from paypal_fee_crawler.crawler import _crawler_revision
from paypal_fee_crawler.models import CountryOutput, CrawlReport, Market, Source, UnsupportedCountry
from paypal_fee_crawler.output import OutputPublisher
from paypal_fee_crawler.validation import (
    _crawler_submodule_revision,
    _derive_publication_stats,
    validate_output_tree,
)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _render_readme_table(stats: dict[str, str]) -> str:
    lines = ["| Metric | Value |", "|--------|------:|"]
    for key, value in stats.items():
        lines.append(f"| {key} | {value} |")
    lines.append("")
    return "\n".join(lines)


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


def _write_minimal_tree(
    root: Path,
    country: dict[str, Any],
    crawler_rev: str | None = None,
    readme_stats: dict[str, str] | None = None,
    skip_readme: bool = False,
) -> None:
    from paypal_fee_crawler.regression import _country_output_hash

    crawler_dir = root / "crawler"
    actual_crawler_rev = _git_init_with_commit(crawler_dir)
    if crawler_rev is None:
        crawler_rev = actual_crawler_rev

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

    if not skip_readme:
        stats = readme_stats if readme_stats is not None else _derive_publication_stats(root)
        readme = (
            f"# Test data\n\n## Statistics\n\n<!-- STATS_START -->\n{_render_readme_table(stats)}<!-- STATS_END -->\n"
        )
        (root / "README.md").write_text(readme, encoding="utf-8")


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


def test_crawler_revision_prefers_crawler_dir(tmp_path: Path) -> None:
    """_crawler_revision returns the HEAD of the supplied crawler checkout."""
    rev = _git_init_with_commit(tmp_path)
    assert _crawler_revision(tmp_path) == rev


def test_crawler_revision_falls_back_to_source_root() -> None:
    """_crawler_revision falls back to the source checkout when no crawler dir is supplied."""
    rev = _crawler_revision(None)
    assert rev is not None
    assert len(rev) == 40
    assert all(c in "0123456789abcdef" for c in rev.lower())


def test_crawler_revision_falls_back_when_dir_missing(tmp_path: Path) -> None:
    """_crawler_revision falls back to the source checkout when the supplied dir is not a git repo."""
    missing = tmp_path / "not-a-repo"
    missing.mkdir()
    rev = _crawler_revision(missing)
    assert rev is not None
    assert len(rev) == 40


def test_crawler_submodule_revision_reads_git_head(tmp_path: Path) -> None:
    """The validation helper reads the crawler submodule HEAD."""
    crawler_dir = tmp_path / "crawler"
    rev = _git_init_with_commit(crawler_dir)
    assert _crawler_submodule_revision(tmp_path) == rev


def test_crawler_submodule_revision_missing_git(tmp_path: Path) -> None:
    """The validation helper returns None when the crawler dir is not a git checkout."""
    assert _crawler_submodule_revision(tmp_path) is None


def test_strict_validation_passes_for_clean_publication(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path, _country(_complete_derived()))
    assert validate_output_tree(tmp_path, strict=True) == []


def test_missing_change_report_fails_strict_validation(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path, _country(_complete_derived()))
    (tmp_path / "change-report.json").unlink()
    errors = validate_output_tree(tmp_path, strict=True)
    assert any("change-report.json is missing" in e for e in errors)


def test_missing_has_regression_fails_strict_validation(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path, _country(_complete_derived()))
    _write_json(tmp_path / "change-report.json", {"schema_version": 1, "changes": []})
    errors = validate_output_tree(tmp_path, strict=True)
    assert any("missing has_regression" in e.lower() for e in errors)


def test_has_regression_true_fails_strict_validation(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path, _country(_complete_derived()))
    _write_json(
        tmp_path / "change-report.json",
        {"schema_version": 1, "changes": [], "has_regression": True},
    )
    errors = validate_output_tree(tmp_path, strict=True)
    assert any("has_regression must be false" in e.lower() for e in errors)


def test_missing_crawl_report_fails_strict_validation(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path, _country(_complete_derived()))
    (tmp_path / "meta" / "crawl-report.json").unlink()
    errors = validate_output_tree(tmp_path, strict=True)
    assert any("meta/crawl-report.json is missing" in e for e in errors)


def test_crawler_revision_mismatch_fails_strict_validation(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path, _country(_complete_derived()), crawler_rev="b" * 40)
    errors = validate_output_tree(tmp_path, strict=True)
    assert any("does not match crawler submodule" in e for e in errors)


def test_malformed_crawler_revision_fails_strict_validation(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path, _country(_complete_derived()), crawler_rev="short")
    errors = validate_output_tree(tmp_path, strict=True)
    assert any("not a full 40-character Git hash" in e for e in errors)


def test_missing_crawler_revision_file_fails_strict_validation(tmp_path: Path) -> None:
    _write_minimal_tree(tmp_path, _country(_complete_derived()))
    (tmp_path / "meta" / "crawler-revision.json").unlink()
    errors = validate_output_tree(tmp_path, strict=True)
    assert any("meta/crawler-revision.json is missing" in e for e in errors)


def test_incomplete_supported_data_passes_require_all_complete(tmp_path: Path) -> None:
    """A supported market that is partial is still accepted by --require-all-complete."""
    derived = _complete_derived()
    derived["status"] = "partial"
    _write_minimal_tree(tmp_path, _country(derived))
    errors = validate_output_tree(tmp_path, require_all_complete=True)
    assert not any("not complete" in e.lower() for e in errors)


def test_missing_supported_country_file_fails_validation(tmp_path: Path) -> None:
    """A supported country missing its JSON file is reported."""
    _write_minimal_tree(tmp_path, _country(_complete_derived()))
    (tmp_path / "json" / "de.json").unlink()
    errors = validate_output_tree(tmp_path)
    assert any("has no country file" in e.lower() for e in errors)


def test_unsupported_markets_are_explicit(tmp_path: Path) -> None:
    """An unsupported market must be recorded in unsupported-countries.json."""
    derived = _complete_derived()
    _write_minimal_tree(tmp_path, _country(derived))
    # Add France to the manifest but not to the index or unsupported file.
    manifest = json.loads((tmp_path / "meta" / "countries.json").read_text(encoding="utf-8"))
    manifest["markets"].append(
        {
            "paypal_market_code": "FR",
            "iso_country_code": "FR",
            "country_name": "France",
            "region": "europe",
            "locale": "fr_FR",
            "languages": [],
        }
    )
    _write_json(tmp_path / "meta" / "countries.json", manifest)
    errors = validate_output_tree(tmp_path, require_all_complete=True)
    assert any("FR: expected exactly one state" in e for e in errors)


def test_unsupported_market_recorded_passes(tmp_path: Path) -> None:
    """An unsupported market listed in unsupported-countries.json is accepted."""
    derived = _complete_derived()
    _write_minimal_tree(tmp_path, _country(derived))
    manifest = json.loads((tmp_path / "meta" / "countries.json").read_text(encoding="utf-8"))
    manifest["markets"].append(
        {
            "paypal_market_code": "FR",
            "iso_country_code": "FR",
            "country_name": "France",
            "region": "europe",
            "locale": "fr_FR",
            "languages": [],
        }
    )
    manifest["unsupported"].append(
        UnsupportedCountry(
            paypal_market_code="FR",
            iso_country_code="FR",
            country_name="France",
            reason="no public fee page",
        ).model_dump(mode="json", exclude_none=True)
    )
    _write_json(tmp_path / "meta" / "countries.json", manifest)
    unsupported = json.loads((tmp_path / "meta" / "unsupported-countries.json").read_text(encoding="utf-8"))
    unsupported["unsupported"].append(
        UnsupportedCountry(
            paypal_market_code="FR",
            iso_country_code="FR",
            country_name="France",
            reason="no public fee page",
        ).model_dump(mode="json", exclude_none=True)
    )
    _write_json(tmp_path / "meta" / "unsupported-countries.json", unsupported)
    errors = validate_output_tree(tmp_path, require_all_complete=True)
    assert not any("FR: expected exactly one state" in e for e in errors)


def test_readme_metrics_mismatch_fails_strict_validation(tmp_path: Path) -> None:
    wrong_stats = {"Countries": "999"}
    _write_minimal_tree(tmp_path, _country(_complete_derived()), readme_stats=wrong_stats)
    errors = validate_output_tree(tmp_path, strict=True)
    assert any("README.md metric" in e for e in errors)


def test_output_publisher_writes_crawler_revision_and_crawl_report(tmp_path: Path) -> None:
    """OutputPublisher writes the crawler revision, transient failures, and a separate crawl report."""
    publisher = OutputPublisher(tmp_path, timestamp="2025-01-01T00:00:00+00:00")

    output = CountryOutput(
        schema_version=1,
        generated_at="2025-01-01T00:00:00+00:00",
        market=Market(paypal_market_code="DE", iso_country_code="DE", country_name="Germany"),
        source=Source(requested_url="https://example.com/de"),
    )
    _, staging = publisher.publish(
        {"DE": output},
        [Market(paypal_market_code="DE", iso_country_code="DE", country_name="Germany")],
        [],
        change_report=CrawlReport(exit_code=0, changed=False, countries_processed=1),
        crawler_revision="c" * 40,
        transient_failures=[
            UnsupportedCountry(
                paypal_market_code="FR",
                iso_country_code="FR",
                country_name="France",
                reason="transient failure during crawl",
                temporary=True,
            )
        ],
    )
    assert (staging / "meta" / "crawler-revision.json").exists()
    assert (staging / "change-report.json").exists()
    assert (staging / "meta" / "transient-failures.json").exists()

    report = CrawlReport(exit_code=0, changed=False, countries_processed=1)
    publisher.write_crawl_report(tmp_path, report)
    assert (tmp_path / "meta" / "crawl-report.json").exists()
