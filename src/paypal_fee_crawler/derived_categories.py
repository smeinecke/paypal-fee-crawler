"""Helpers for summarizing the categories present in a derived fee result."""

from __future__ import annotations

from typing import Any


def _selected_categories_from_derived(derived: Any) -> set[str]:
    """Return the non-empty output categories present in a derived fee result."""
    categories: set[str] = set()
    if getattr(derived, "transaction_fee_rules", None):
        categories.add("transaction_fee_rules")
    if getattr(derived, "fixed_fee_schedules", None):
        categories.add("fixed_fee_schedules")
    if getattr(derived, "international_surcharge_schedules", None):
        categories.add("international_surcharge_schedules")
    if getattr(derived, "currency_conversion", None):
        categories.add("currency_conversion")
    return categories
