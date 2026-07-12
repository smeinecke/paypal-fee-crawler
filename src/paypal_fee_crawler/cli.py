"""Command-line interface for the PayPal fee crawler."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from datetime import UTC
from pathlib import Path
from typing import Any

import click

from .classify import CLASSIFIER_VERSION
from .comparison import compare_classifiers
from .crawler import Crawler
from .exceptions import (
    ConfigurationError,
    CountryDiscoveryError,
    CrawlerError,
    ExitCode,
    NetworkError,
    ParserError,
    RegressionError,
)
from .exceptions import (
    ValidationError as CrawlerValidationError,
)
from .models import ClassifierMode, CrawlConfiguration, CrawlReport
from .validation import validate_all_output

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _build_config(
    output: str | None,
    staging_dir: str | None,
    country: tuple[str, ...],
    countries: str | None,
    timeout: float,
    max_workers: int,
    user_agent: str | None,
    atomic: bool,
    fail_on_regression: bool,
    fail_on_warning: bool,
    allow_country_drop: bool,
    refresh_country_manifest: bool,
    keep_diagnostics: bool,
    verbose: bool,
    request_delay: float,
    timestamp: str | None,
    transient_policy: str = "fail",
    classifier_mode: ClassifierMode = ClassifierMode.LEGACY,
) -> CrawlConfiguration:
    selected_countries: list[str] | None = None
    if country:
        selected_countries = list(country)
    if countries:
        selected_countries = [c.strip().upper() for c in countries.split(",") if c.strip()]
    return CrawlConfiguration(
        output_dir=output,
        staging_dir=staging_dir,
        timestamp=timestamp,
        countries=selected_countries,
        timeout=timeout,
        max_workers=max_workers,
        user_agent=user_agent,
        atomic=atomic,
        fail_on_regression=fail_on_regression,
        fail_on_warning=fail_on_warning,
        allow_country_drop=allow_country_drop,
        refresh_country_manifest=refresh_country_manifest,
        keep_diagnostics=keep_diagnostics,
        verbose=verbose,
        request_delay=request_delay,
        transient_policy=transient_policy,
        classifier_mode=classifier_mode,
    )


@click.group()
@click.version_option(version="0.1.0", prog_name="paypal-fee-crawler")
def main() -> None:
    """PayPal fee data crawler."""
    pass


@main.command()
@click.option("--output", required=True, type=click.Path(), help="Output directory for generated data.")
@click.option("--staging-dir", type=click.Path(), help="Staging directory (default: temp dir).")
@click.option("--country", multiple=True, help="Country code to crawl (can be repeated).")
@click.option("--countries", help="Comma-separated list of country codes to crawl.")
@click.option("--timeout", default=30.0, type=float, help="HTTP timeout in seconds.")
@click.option("--max-workers", default=3, type=int, help="Maximum concurrent requests.")
@click.option("--request-delay", default=0.5, type=float, help="Delay between requests in seconds.")
@click.option("--user-agent", help="Custom user agent string.")
@click.option("--atomic/--no-atomic", default=True, help="Publish atomically.")
@click.option("--fail-on-regression/--no-fail-on-regression", default=False, help="Fail on regression.")
@click.option("--fail-on-warning", is_flag=True, help="Fail on parser warnings.")
@click.option("--allow-country-drop", is_flag=True, help="Allow previously supported countries to drop.")
@click.option("--refresh-country-manifest", is_flag=True, help="Refresh country manifest even on discovery failure.")
@click.option("--keep-diagnostics", is_flag=True, help="Keep diagnostic artifacts on failure.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.option(
    "--classifier-mode",
    type=click.Choice([m.value for m in ClassifierMode]),
    default=ClassifierMode.LEGACY.value,
    help="Active classifier mode: legacy, shadow, or structural.",
)
@click.option("--report", type=click.Path(), help="Write machine-readable JSON report to this path.")
@click.option("--timestamp", help="Deterministic timestamp for generated output (ISO 8601).")
@click.option(
    "--transient-policy",
    type=click.Choice(["fail", "reuse-previous"]),
    default="fail",
    help="Behavior when a previously supported country cannot be refreshed.",
)
def crawl(
    output: str,
    staging_dir: str | None,
    country: tuple[str, ...],
    countries: str | None,
    timeout: float,
    max_workers: int,
    request_delay: float,
    user_agent: str | None,
    atomic: bool,
    fail_on_regression: bool,
    fail_on_warning: bool,
    allow_country_drop: bool,
    refresh_country_manifest: bool,
    keep_diagnostics: bool,
    verbose: bool,
    classifier_mode: str,
    report: str | None,
    timestamp: str | None,
    transient_policy: str,
) -> None:
    """Crawl PayPal fee pages and publish deterministic JSON output."""
    _configure_logging(verbose)
    config = _build_config(
        output=output,
        staging_dir=staging_dir,
        country=country,
        countries=countries,
        timeout=timeout,
        max_workers=max_workers,
        user_agent=user_agent,
        atomic=atomic,
        fail_on_regression=fail_on_regression,
        fail_on_warning=fail_on_warning,
        allow_country_drop=allow_country_drop,
        refresh_country_manifest=refresh_country_manifest,
        keep_diagnostics=keep_diagnostics,
        verbose=verbose,
        request_delay=request_delay,
        timestamp=timestamp,
        transient_policy=transient_policy,
        classifier_mode=ClassifierMode(classifier_mode),
    )

    async def _run() -> CrawlReport:
        async with Crawler(config) as crawler:
            return await crawler.crawl()

    try:
        result = asyncio.run(_run())
    except CountryDiscoveryError as exc:
        logger.error("Country discovery failed: %s", exc)
        sys.exit(ExitCode.PARSER_FAILURE)
    except NetworkError as exc:
        logger.error("Network failure: %s", exc)
        sys.exit(ExitCode.NETWORK_FAILURE)
    except ParserError as exc:
        logger.error("Parser failure: %s", exc)
        sys.exit(ExitCode.PARSER_FAILURE)
    except CrawlerValidationError as exc:
        logger.error("Validation failure: %s", exc)
        sys.exit(ExitCode.VALIDATION_FAILURE)
    except RegressionError as exc:
        logger.error("Regression failure: %s", exc)
        sys.exit(ExitCode.REGRESSION_FAILURE)
    except ConfigurationError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(ExitCode.CONFIGURATION_ERROR)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(ExitCode.UNEXPECTED_ERROR)

    if report:
        Path(report).write_text(
            json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    status = "changed" if result.changed else "unchanged"
    click.echo(
        f"Crawl complete ({status}): {result.countries_processed} countries processed, "
        f"{len(result.countries_failed)} failed, {len(result.countries_unsupported)} unsupported."
    )
    sys.exit(result.exit_code)


@main.command()
@click.option("--timeout", default=30.0, type=float, help="HTTP timeout in seconds.")
@click.option("--max-workers", default=3, type=int, help="Maximum concurrent requests.")
@click.option("--request-delay", default=0.5, type=float, help="Delay between requests in seconds.")
@click.option("--user-agent", help="Custom user agent string.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def discover_countries(
    timeout: float,
    max_workers: int,
    request_delay: float,
    user_agent: str | None,
    verbose: bool,
) -> None:
    """Discover and print the PayPal country/market manifest."""
    _configure_logging(verbose)
    config = CrawlConfiguration(
        timeout=timeout,
        max_workers=max_workers,
        user_agent=user_agent,
        request_delay=request_delay,
        verbose=verbose,
    )

    async def _run() -> list[Any]:
        async with Crawler(config) as crawler:
            markets = await crawler.discover()
            return [m.model_dump(mode="json") for m in markets]

    try:
        data = asyncio.run(_run())
        click.echo(json.dumps(data, indent=2, sort_keys=True))
    except Exception as exc:
        logger.error("Discovery failed: %s", exc)
        sys.exit(ExitCode.PARSER_FAILURE)


@main.command()
@click.argument("country_code")
@click.option("--output", type=click.Path(), help="Output directory to write the single country JSON.")
@click.option("--timeout", default=30.0, type=float, help="HTTP timeout in seconds.")
@click.option("--request-delay", default=0.5, type=float, help="Delay between requests in seconds.")
@click.option("--user-agent", help="Custom user agent string.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.option("--timestamp", help="Deterministic timestamp for generated output (ISO 8601).")
def crawl_country(
    country_code: str,
    output: str | None,
    timeout: float,
    request_delay: float,
    user_agent: str | None,
    verbose: bool,
    timestamp: str | None,
) -> None:
    """Crawl a single PayPal market and print or save the result."""
    _configure_logging(verbose)
    config = CrawlConfiguration(
        countries=[country_code.upper()],
        output_dir=output or "./out",
        timestamp=timestamp,
        timeout=timeout,
        request_delay=request_delay,
        user_agent=user_agent,
        verbose=verbose,
        atomic=False,
    )

    async def _run() -> CrawlReport:
        async with Crawler(config) as crawler:
            return await crawler.crawl()

    try:
        result = asyncio.run(_run())
    except CrawlerError as exc:
        logger.error("Crawl failed: %s", exc)
        sys.exit(ExitCode.PARSER_FAILURE)
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        sys.exit(ExitCode.UNEXPECTED_ERROR)

    click.echo(
        f"Crawl complete: {result.countries_processed} countries processed, "
        f"{len(result.countries_failed)} failed, {len(result.countries_unsupported)} unsupported."
    )
    sys.exit(result.exit_code)


@main.command()
@click.argument("output_dir", type=click.Path(exists=True))
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def validate(output_dir: str, verbose: bool) -> None:
    """Validate all generated JSON files in the output directory."""
    _configure_logging(verbose)
    errors = validate_all_output(output_dir)
    if errors:
        for error in errors:
            click.echo(error)
        sys.exit(ExitCode.VALIDATION_FAILURE)
    click.echo("All generated files are valid.")
    sys.exit(ExitCode.SUCCESS_NO_CHANGE)


@main.command()
@click.argument("old_file", type=click.Path(exists=True))
@click.argument("new_file", type=click.Path(exists=True))
def diff(old_file: str, new_file: str) -> None:
    """Show a structured diff between two country JSON files."""
    old_data = json.loads(Path(old_file).read_text(encoding="utf-8"))
    new_data = json.loads(Path(new_file).read_text(encoding="utf-8"))
    changes: dict[str, Any] = {}
    for key in set(old_data.keys()) | set(new_data.keys()):
        if old_data.get(key) != new_data.get(key):
            changes[key] = {"before": old_data.get(key), "after": new_data.get(key)}
    click.echo(json.dumps(changes, indent=2, sort_keys=True, default=str))


@main.command()
@click.argument("html_file", type=click.Path(exists=True))
def inspect(html_file: str) -> None:
    """Extract and print the CMS render context from a local HTML file."""
    from .cms_context import extract_cms_context

    text = Path(html_file).read_text(encoding="utf-8")
    try:
        data = extract_cms_context(text)
        click.echo(json.dumps(data, indent=2, sort_keys=True, default=str))
    except Exception as exc:
        click.echo(f"Failed to extract CMS context: {exc}")
        sys.exit(ExitCode.PARSER_FAILURE)


@main.command("compare-classifiers")
@click.argument("json_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output", type=click.Path(), help="Output directory for comparison reports.")
@click.option("--country", multiple=True, help="Country code to compare (can be repeated).")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def compare_classifiers_cmd(
    json_dir: str,
    output: str | None,
    country: tuple[str, ...],
    verbose: bool,
) -> None:
    """Compare legacy and structural classifiers over existing country output.

    Loads each CountryOutput from JSON_DIR, re-classifies its tables with both
    engines, and writes a deterministic JSON and Markdown report.
    """
    _configure_logging(verbose)
    countries = set(country) if country else None
    output_path = output or "."
    try:
        report = compare_classifiers(Path(json_dir), Path(output_path), countries)
    except Exception as exc:
        logger.error("Comparison failed: %s", exc)
        sys.exit(ExitCode.PARSER_FAILURE)
    click.echo(f"Comparison report written to {report.json_path}")


@main.command("promote-classifiers")
@click.argument("gold_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--output", type=click.Path(), help="Output directory for promotion report.")
@click.option("--run-tests/--no-tests", default=True, help="Run the pytest fixture test suite.")
@click.option("--bump", is_flag=True, help="Bump the structural classifier version in classify.py on success.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
def promote_classifiers(
    gold_dir: str,
    output: str | None,
    run_tests: bool,
    bump: bool,
    verbose: bool,
) -> None:
    """Promote the structural classifier if the gold corpus and tests pass.

    Loads each CountryOutput from GOLD_DIR and compares legacy vs structural
    outputs.  Any status change, category change, or value change blocks
    promotion.  When ``--run-tests`` is set, the pytest fixture test suite is
    also required to pass.  On success, a promotion report is written and the
    classifier version may be bumped with ``--bump``.
    """
    _configure_logging(verbose)
    output_path = Path(output or ".")

    report = compare_classifiers(Path(gold_dir), output_path)
    summary = report.summary
    errors: list[str] = []
    if summary.status_changed:
        errors.append(f"status changed for {summary.status_changed} countries")
    if summary.categories_changed:
        errors.append(f"categories changed for {summary.categories_changed} countries")
    if summary.value_changes:
        errors.append(f"{summary.value_changes} value changes")
    if report.comparisons and any(not c.selected_categories_match for c in report.comparisons):
        errors.append("some countries have mismatched selected categories")
    if report.comparisons and any(not c.status_match for c in report.comparisons):
        errors.append("some countries have mismatched status")

    if run_tests:
        logger.info("Running fixture test suite")
        result = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], capture_output=True, text=True)
        if result.returncode != 0:
            errors.append("pytest fixture test suite failed")
            logger.error("pytest output:\n%s", result.stdout + result.stderr)

    if errors:
        for error in errors:
            logger.error("Promotion blocked: %s", error)
        sys.exit(ExitCode.PARSER_FAILURE)

    promotion_report = {
        "schema_version": 1,
        "approved_at": _iso_now(),
        "legacy_classifier_version": "legacy",
        "structural_classifier_version": CLASSIFIER_VERSION,
        "summary": {
            "total_countries": summary.total_countries,
            "status_changed": summary.status_changed,
            "categories_changed": summary.categories_changed,
            "value_changes": summary.value_changes,
            "total_observations": summary.total_observations,
        },
    }
    output_path.mkdir(parents=True, exist_ok=True)
    promotion_path = output_path / "promotion-report.json"
    promotion_path.write_text(json.dumps(promotion_report, indent=2, sort_keys=True), encoding="utf-8")

    click.echo(f"Promotion report written to {promotion_path}")
    click.echo(f"Structural classifier version: {CLASSIFIER_VERSION}")

    if bump:
        _bump_classifier_version(output_path)
        click.echo("Bumped structural classifier version in classify.py")


def _iso_now() -> str:
    """Return an ISO 8601 UTC timestamp."""
    from datetime import datetime

    return datetime.now(UTC).isoformat()


def _bump_classifier_version(output_path: Path) -> None:
    """Bump the structural classifier version constant in classify.py."""
    classify_path = Path(__file__).with_name("classify.py")
    text = classify_path.read_text(encoding="utf-8")
    prefix = "CLASSIFIER_VERSION = \""
    start = text.index(prefix) + len(prefix)
    end = text.index('"', start)
    current = text[start:end]
    parts = current.split("-")
    if parts and parts[-1].isdigit():
        parts[-1] = str(int(parts[-1]) + 1)
    else:
        parts.append("2")
    new = "-".join(parts)
    classify_path.write_text(text[:start] + new + text[end:], encoding="utf-8")


if __name__ == "__main__":
    main()
