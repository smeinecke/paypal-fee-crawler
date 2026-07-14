# Makefile for paypal-fee-crawler

.PHONY: all format check validate test test-unit test-live complexity vulture xenon help

all: validate test-unit

format:
	uv run ruff format --check --diff .

reformat-ruff:
	uv run ruff format .

check:
	uv run ruff check .

fix-ruff:
	uv run ruff check . --fix

fix: reformat-ruff fix-ruff
	@echo "Updated code."

pyright:
	uv run pyright

bandit:
	uv run bandit -c pyproject.toml -r src

vulture:
	uv run vulture . --exclude .venv,migrations,tests --make-whitelist

complexity:
	uv run radon cc . -a -nc

xenon:
	uv run xenon -b D -m C -a C .

test:
	uv run pytest

test-unit:
	uv run pytest tests/ -m "not live"

test-live:
	uv run pytest tests/ -m live

validate: format check pyright bandit complexity vulture xenon
	@echo "Validation passed."

help:
	@echo "Available targets:"
	@echo "  all           - Run validation and unit tests (default)"
	@echo "  format        - Check code formatting with ruff"
	@echo "  reformat-ruff - Format code with ruff"
	@echo "  check         - Run ruff linting"
	@echo "  fix-ruff      - Auto-fix ruff issues"
	@echo "  fix           - Run reformat-ruff and fix-ruff"
	@echo "  pyright       - Run type checking"
	@echo "  bandit        - Run security analysis"
	@echo "  complexity    - Run cyclomatic complexity analysis with radon"
	@echo "  vulture       - Run dead code detection with vulture"
	@echo "  xenon         - Run maintainability monitoring with xenon"
	@echo "  test          - Run all tests"
	@echo "  test-unit     - Run unit tests (no live network)"
	@echo "  test-live     - Run live integration tests"
	@echo "  validate      - Run all validation checks"
	@echo "  help          - Show this help message"
