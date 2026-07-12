"""Shared pytest fixtures and helpers for the PayPal fee crawler test suite."""

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


@pytest.fixture
def gold_corpus_dir(fixtures_dir: Path) -> Path:
    """Directory containing hand-reviewed gold corpus fixtures."""
    return fixtures_dir / "corpus" / "gold"


@pytest.fixture
def synthetic_corpus_dir(fixtures_dir: Path) -> Path:
    """Directory containing synthetic corpus fixtures."""
    return fixtures_dir / "corpus" / "synthetic"
