"""Tests for the gold and synthetic corpus fixtures."""

from __future__ import annotations

from pathlib import Path

from paypal_fee_crawler.models import CountryOutput


def _load_countries(corpus_dir: Path) -> list[CountryOutput]:
    countries: list[CountryOutput] = []
    for path in sorted(corpus_dir.glob("*.json")):
        countries.append(CountryOutput.model_validate_json(path.read_text(encoding="utf-8")))
    return countries


def test_gold_corpus_fixtures_load(gold_corpus_dir: Path) -> None:
    countries = _load_countries(gold_corpus_dir)
    assert countries
    for country in countries:
        assert country.market.paypal_market_code
        assert country.tables


def test_gold_corpus_legacy_and_structural_agree(gold_corpus_dir: Path) -> None:
    from paypal_fee_crawler.classify import classify_legacy, classify_structural
    from paypal_fee_crawler.comparison import compare_runs

    for country in _load_countries(gold_corpus_dir):
        legacy_run = classify_legacy(country.tables, market_code=country.market.paypal_market_code)
        structural_run = classify_structural(country.tables, market_code=country.market.paypal_market_code)
        comparison = compare_runs(legacy_run, structural_run, country.market)
        assert comparison.selected_categories_match, f"{country.market.paypal_market_code} categories mismatch"
        assert comparison.status_match, f"{country.market.paypal_market_code} status mismatch"
        assert not comparison.value_changes, f"{country.market.paypal_market_code} has value changes"


def test_synthetic_corpus_fixtures_load(synthetic_corpus_dir: Path) -> None:
    countries = _load_countries(synthetic_corpus_dir)
    assert countries
    for country in countries:
        assert country.tables
