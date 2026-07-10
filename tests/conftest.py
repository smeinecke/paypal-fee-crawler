"""Shared pytest fixtures and helpers."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def de_html(fixtures_dir: Path) -> str:
    return (fixtures_dir / "de.html").read_text(encoding="utf-8")


@pytest.fixture
def us_html(fixtures_dir: Path) -> str:
    return (fixtures_dir / "us.html").read_text(encoding="utf-8")


@pytest.fixture
def gb_html(fixtures_dir: Path) -> str:
    return (fixtures_dir / "gb.html").read_text(encoding="utf-8")


@pytest.fixture
def de_real_html(fixtures_dir: Path) -> str:
    return (fixtures_dir / "paypal-de-real.html").read_text(encoding="utf-8")


@pytest.fixture
def us_real_html(fixtures_dir: Path) -> str:
    return (fixtures_dir / "paypal-us-real.html").read_text(encoding="utf-8")


@pytest.fixture
def gb_real_html(fixtures_dir: Path) -> str:
    return (fixtures_dir / "paypal-gb-real.html").read_text(encoding="utf-8")
