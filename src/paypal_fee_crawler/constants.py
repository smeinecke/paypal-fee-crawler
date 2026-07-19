"""Shared crawler constants that multiple modules need without creating cycles."""

from __future__ import annotations

# Output roots that the crawler owns and may modify.
MANAGED_ROOTS = ("json", "meta", "schemas", "change-report.json")

# Full set of managed paths, including generated README updates.
MANAGED_PATHS = (*MANAGED_ROOTS, "README.md")
