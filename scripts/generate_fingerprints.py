#!/usr/bin/env python3
"""Generate and review canonical fingerprints from a normalized country corpus.

Example:
    uv run python scripts/generate_fingerprints.py ../paypal-fee-data/json --output fingerprints/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from paypal_fee_crawler.models import CountryOutput
from paypal_fee_crawler.profiles import build_table_profile
from paypal_fee_crawler.registry import FingerprintBuilder


def _load_tables(json_dir: Path) -> list[tuple[str, str, Any, Any]]:
    """Load (country, table_id, table, profile) tuples from *json_dir*."""
    tables: list[tuple[str, str, Any, Any]] = []
    for path in sorted(json_dir.glob("*.json")):
        try:
            country = CountryOutput.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {path.name}: {exc}")
            continue
        for table in country.tables:
            profile = build_table_profile(table)
            tables.append((country.market.paypal_market_code, table.table_id or "", table, profile))
    return tables


def _group_by_fingerprint(tables: list[tuple[str, str, Any, Any]]) -> dict[str, dict]:
    """Group tables by canonical fingerprint and return reviewable metadata."""
    groups: dict[str, dict] = {}
    for country, table_id, table, profile in tables:
        fingerprint = str(FingerprintBuilder.build(profile, table))
        if fingerprint not in groups:
            components = FingerprintBuilder.components(profile, table)
            groups[fingerprint] = {
                "fingerprint": fingerprint,
                "components": components.__dict__,
                "tables": [],
                "document_ids": [],
                "markets": [],
            }
        groups[fingerprint]["tables"].append({"country": country, "table_id": table_id})
        if table.document_id:
            groups[fingerprint]["document_ids"].append(table.document_id)
        groups[fingerprint]["markets"].append(country)
    return groups


def _write_outputs(groups: dict[str, dict], output_dir: Path) -> None:
    """Write deterministic JSON and Markdown review files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "fingerprints.json"
    data = {
        "fingerprint_version": 1,
        "group_count": len(groups),
        "groups": dict(sorted(groups.items())),
    }
    json_path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    md_path = output_dir / "fingerprints.md"
    lines = [
        "# Fingerprint Review Report",
        "",
        f"Groups: {len(groups)}",
        "",
    ]
    for fingerprint in sorted(groups):
        group = groups[fingerprint]
        markets = sorted(set(group["markets"]))
        doc_ids = sorted(set(group["document_ids"]))
        lines.append(f"## {fingerprint}")
        lines.append(f"- Markets: {', '.join(markets) if markets else 'none'}")
        lines.append(f"- Document IDs: {', '.join(doc_ids) if doc_ids else 'none'}")
        lines.append(f"- Occurrences: {len(group['tables'])}")
        lines.append("")
        for occurrence in group["tables"]:
            lines.append(f"- `{occurrence['country']}` / `{occurrence['table_id']}`")
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate canonical fingerprints from a corpus")
    parser.add_argument("json_dir", type=Path, help="Directory containing CountryOutput JSON files")
    parser.add_argument("--output", type=Path, default=Path("fingerprint-review"), help="Output directory")
    args = parser.parse_args()

    tables = _load_tables(args.json_dir)
    groups = _group_by_fingerprint(tables)
    _write_outputs(groups, args.output)


if __name__ == "__main__":
    main()
