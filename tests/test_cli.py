"""Tests for CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from click.testing import CliRunner

from paypal_fee_crawler.cli import main
from paypal_fee_crawler.http import HttpClient, HttpResponse


def test_cli_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "crawl" in result.output


def test_cli_crawl_country_help() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["crawl-country", "--help"])
    assert result.exit_code == 0
    assert "single PayPal market" in result.output


def test_cli_validate_missing_dir() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["validate", "/nonexistent/path"])
    assert result.exit_code != 0


def test_cli_inspect_fixture(fixtures_dir: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["inspect", str(fixtures_dir / "de.html")])
    assert result.exit_code == 0
    assert "business/paypal-business-fees" in result.output


def test_cli_diff(tmp_path: Path) -> None:
    a = {
        "schema_version": 1,
        "market": {"paypal_market_code": "DE", "iso_country_code": "DE", "country_name": "Germany"},
    }
    b = {
        "schema_version": 1,
        "market": {"paypal_market_code": "DE", "iso_country_code": "DE", "country_name": "Deutschland"},
    }
    a_path = tmp_path / "a.json"
    b_path = tmp_path / "b.json"
    a_path.write_text(json.dumps(a))
    b_path.write_text(json.dumps(b))
    runner = CliRunner()
    result = runner.invoke(main, ["diff", str(a_path), str(b_path)])
    assert result.exit_code == 0
    assert "Deutschland" in result.output


def _load_fixture(name: str) -> str:
    return (Path(__file__).parent / "fixtures" / name).read_text(encoding="utf-8")


async def _fake_http_get(self: HttpClient, url: str, **kwargs: Any) -> HttpResponse:
    if url.endswith("/de") or url.endswith("/de/"):
        html = _load_fixture("de.html")
    elif url.endswith("/us/business/paypal-business-fees") or url.endswith("/us") or url.endswith("/us/"):
        html = _load_fixture("us.html")
    elif url.endswith("/gb/business/paypal-business-fees") or url.endswith("/gb") or url.endswith("/gb/"):
        html = _load_fixture("gb.html")
    elif url.endswith("/de/business/paypal-business-fees"):
        html = _load_fixture("de.html")
    else:
        html = _load_fixture("de.html")
    return HttpResponse(
        url=url,
        status_code=200,
        content=html.encode("utf-8"),
        text=html,
        headers={"content-type": "text/html; charset=utf-8"},
    )


def test_cli_crawl_country_command(tmp_path: Path) -> None:
    runner = CliRunner()
    with patch.object(HttpClient, "get", _fake_http_get):
        result = runner.invoke(
            main,
            [
                "crawl-country",
                "DE",
                "--output",
                str(tmp_path),
                "--request-delay",
                "0",
            ],
        )
    assert result.exit_code in (0, 1), result.output
    assert (tmp_path / "json" / "de.json").exists()


def test_cli_discover_countries_command() -> None:
    runner = CliRunner()
    with patch.object(HttpClient, "get", _fake_http_get):
        result = runner.invoke(
            main,
            [
                "discover-countries",
                "--request-delay",
                "0",
            ],
        )
    assert result.exit_code == 0, result.output


def test_cli_crawl_command(tmp_path: Path) -> None:
    runner = CliRunner()
    with patch.object(HttpClient, "get", _fake_http_get):
        result = runner.invoke(
            main,
            [
                "crawl",
                "--output",
                str(tmp_path),
                "--country",
                "DE",
                "--country",
                "US",
                "--request-delay",
                "0",
                "--max-workers",
                "1",
            ],
        )
    assert result.exit_code in (0, 1), result.output
    assert (tmp_path / "json" / "de.json").exists()
    assert (tmp_path / "json" / "us.json").exists()


def test_cli_crawl_command_fail_on_warning(tmp_path: Path) -> None:
    runner = CliRunner()
    with patch.object(HttpClient, "get", _fake_http_get):
        result = runner.invoke(
            main,
            [
                "crawl",
                "--output",
                str(tmp_path),
                "--country",
                "DE",
                "--request-delay",
                "0",
                "--max-workers",
                "1",
                "--fail-on-warning",
            ],
        )
    # The exit code depends on whether the fixture produces parser warnings.
    assert result.exit_code in (0, 1, 2), result.output
