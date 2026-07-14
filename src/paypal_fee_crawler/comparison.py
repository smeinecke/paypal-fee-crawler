"""Legacy classifier comparison helpers (deprecated).

This module previously compared legacy and structural classifier outputs.  With the
rule-based redesign it is retained only as a stub so existing CLI promotion logic
remains importable.  New code should use ``paypal_fee_crawler.classify`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ComparisonResult:
    """Empty placeholder for the legacy classifier comparison result."""

    selected_categories_match: bool = True
    status_match: bool = True
    standard_percentage_match: bool = True
    fixed_fees_match: bool = True
    international_surcharges_match: bool = True
    changes: list[dict[str, Any]] | None = None
    legacy_selected_categories: tuple[str, ...] = ()


def compare_runs(*_args: Any, **_kwargs: Any) -> ComparisonResult:
    """Legacy classifier comparison is no longer supported."""
    return ComparisonResult()


def compare_against_gold(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
    """Legacy gold-corpus comparison is no longer supported."""
    return {"summary": {"total_countries": 0, "mismatches": 0}, "countries": []}


def compare_classifiers(*_args: Any, **_kwargs: Any) -> Path:
    """Legacy classifier comparison report generation is no longer supported."""
    raise NotImplementedError("classifier comparison reports are no longer supported")
