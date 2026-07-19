"""Deterministic hashing helpers for crawler output."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def _country_output_hash(data: dict[str, Any]) -> str:
    """Return a deterministic hash of the public business content of a country output."""
    canonical = {
        "market": data.get("market"),
        "source": {
            k: v
            for k, v in (data.get("source") or {}).items()
            if k not in {"etag", "last_modified", "content_sha256", "artifact_sha256"}
        },
        "sections": data.get("sections"),
        "tables": data.get("tables"),
        "derived": data.get("derived"),
    }
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
