"""Command-line interface for the PayPal fee crawler."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

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
from .models import CrawlConfiguration, CrawlReport
from .validation import validate_all_output, validate_output_tree

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _http_options(f: Any) -> Any:
    """Composite decorator for HTTP-related CLI options."""
    f = click.option("--timeout", default=30.0, type=float, help="HTTP timeout in seconds.")(f)
    f = click.option("--max-workers", default=3, type=int, help="Maximum concurrent requests.")(f)
    f = click.option("--request-delay", default=0.5, type=float, help="Delay between requests in seconds.")(f)
    f = click.option("--user-agent", help="Custom user agent string.")(f)
    return f


def _cache_options(f: Any) -> Any:
    """Composite decorator for HTTP-cache CLI options."""
    f = click.option(
        "--cache-dir", type=click.Path(), envvar="PAYPAL_FEE_CRAWLER_CACHE_DIR", help="HTTP cache directory."
    )(f)
    f = click.option(
        "--cache-ttl-hours",
        default=24.0,
        type=float,
        envvar="PAYPAL_FEE_CRAWLER_CACHE_TTL_HOURS",
        help="Cache entry TTL in hours.",
    )(f)
    f = click.option(
        "--no-cache", is_flag=True, envvar="PAYPAL_FEE_CRAWLER_NO_CACHE", help="Bypass HTTP cache reads and writes."
    )(f)
    f = click.option(
        "--refresh-cache",
        is_flag=True,
        envvar="PAYPAL_FEE_CRAWLER_REFRESH_CACHE",
        help="Force network revalidation/download and update cached entries.",
    )(f)
    return f


def _resolve_cache_dir(no_cache: bool, cache_dir: str | None) -> str | None:
    if no_cache:
        return None
    if cache_dir:
        return cache_dir
    return str(Path.cwd() / ".cache" / "paypal-fee-crawler" / "http")


def _build_config(**kwargs: Any) -> CrawlConfiguration:
    """Build a ``CrawlConfiguration`` from CLI option kwargs."""
    if not kwargs.get("atomic", True):
        logger.warning("--no-atomic is deprecated and has no effect; atomic publication is always used.")
    if kwargs.get("refresh_country_manifest"):
        logger.warning("--refresh-country-manifest is deprecated and has no effect.")

    country: tuple[str, ...] = kwargs.pop("country", ())
    countries: str | None = kwargs.pop("countries", None)
    selected_countries: list[str] | None = None
    if country:
        selected_countries = list(country)
    if countries:
        selected_countries = [c.strip().upper() for c in countries.split(",") if c.strip()]

    no_cache: bool = kwargs.get("no_cache", False)
    cache_dir: str | None = kwargs.pop("cache_dir", None)
    output: str | None = kwargs.pop("output", None)
    staging_dir: str | None = kwargs.pop("staging_dir", None)
    timestamp: str | None = kwargs.pop("timestamp", None)

    config_kwargs = {
        "output_dir": output,
        "staging_dir": staging_dir,
        "timestamp": timestamp,
        "countries": selected_countries,
        "cache_dir": _resolve_cache_dir(no_cache, cache_dir),
    }
    config_kwargs.update({k: v for k, v in kwargs.items() if k in CrawlConfiguration.model_fields})
    return CrawlConfiguration(**config_kwargs)


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
@_http_options
@click.option("--atomic/--no-atomic", default=True, help="Publish atomically.")
@click.option("--fail-on-regression/--no-fail-on-regression", default=False, help="Fail on regression.")
@click.option("--fail-on-warning", is_flag=True, help="Fail on parser warnings.")
@click.option("--allow-country-drop", is_flag=True, help="Allow previously supported countries to drop.")
@click.option("--refresh-country-manifest", is_flag=True, help="Refresh country manifest even on discovery failure.")
@click.option("--keep-diagnostics", is_flag=True, help="Keep diagnostic artifacts on failure.")
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.option("--report", type=click.Path(), help="Write machine-readable JSON report to this path.")
@click.option("--timestamp", help="Deterministic timestamp for generated output (ISO 8601).")
@click.option(
    "--transient-policy",
    type=click.Choice(["fail", "reuse-previous"]),
    default="fail",
    help="Behavior when a previously supported country cannot be refreshed.",
)
@_cache_options
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
    report: str | None,
    timestamp: str | None,
    transient_policy: str,
    cache_dir: str | None,
    cache_ttl_hours: float,
    no_cache: bool,
    refresh_cache: bool,
) -> None:
    """Crawl PayPal fee pages and publish deterministic JSON output."""
    _configure_logging(verbose)
    # ``atomic`` and ``refresh_country_manifest`` are deprecated flags consumed
    # by ``_build_config``; keep them referenced so static analysers see usage.
    _ = atomic, refresh_country_manifest
    config_kwargs = {k: v for k, v in locals().items() if k != "report"}
    config = _build_config(**config_kwargs)

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
@_http_options
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@_cache_options
def discover_countries(
    timeout: float,
    max_workers: int,
    request_delay: float,
    user_agent: str | None,
    verbose: bool,
    cache_dir: str | None,
    cache_ttl_hours: float,
    no_cache: bool,
    refresh_cache: bool,
) -> None:
    """Discover and print the PayPal country/market manifest."""
    _configure_logging(verbose)
    config = _build_config(
        timeout=timeout,
        max_workers=max_workers,
        user_agent=user_agent,
        request_delay=request_delay,
        verbose=verbose,
        cache_dir=cache_dir,
        cache_ttl_hours=cache_ttl_hours,
        no_cache=no_cache,
        refresh_cache=refresh_cache,
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
@_http_options
@click.option("--verbose", is_flag=True, help="Enable verbose logging.")
@click.option("--timestamp", help="Deterministic timestamp for generated output (ISO 8601).")
@_cache_options
def crawl_country(
    country_code: str,
    output: str | None,
    timeout: float,
    max_workers: int,
    request_delay: float,
    user_agent: str | None,
    verbose: bool,
    timestamp: str | None,
    cache_dir: str | None,
    cache_ttl_hours: float,
    no_cache: bool,
    refresh_cache: bool,
) -> None:
    """Crawl a single PayPal market and print or save the result."""
    _configure_logging(verbose)
    config = _build_config(
        countries=country_code.upper(),
        output=output or "./out",
        timestamp=timestamp,
        timeout=timeout,
        max_workers=max_workers,
        request_delay=request_delay,
        user_agent=user_agent,
        verbose=verbose,
        cache_dir=cache_dir,
        cache_ttl_hours=cache_ttl_hours,
        no_cache=no_cache,
        refresh_cache=refresh_cache,
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
@click.option("--strict", is_flag=True, help="Enable strict semantic validation.")
@click.option(
    "--require-all-complete",
    is_flag=True,
    help="Require the publication tree to be complete (all discovered markets have an output).",
)
def validate(output_dir: str, verbose: bool, strict: bool, require_all_complete: bool) -> None:
    """Validate all generated JSON files in the output directory."""
    _configure_logging(verbose)
    errors = validate_all_output(output_dir, strict=strict, require_all_complete=require_all_complete)
    errors.extend(validate_output_tree(output_dir, strict=strict, require_all_complete=require_all_complete))
    errors = sorted(set(errors))
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


if __name__ == "__main__":
    main()
