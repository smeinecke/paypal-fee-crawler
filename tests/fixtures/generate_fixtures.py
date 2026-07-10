#!/usr/bin/env python3
"""Generate sanitized synthetic HTML fixtures for offline tests."""

from __future__ import annotations

import json
from pathlib import Path


def _make_token(raw: str, fee_data_key: str | None = None, internal_name: str | None = None) -> dict:
    return {
        "sys": {"id": f"token-{fee_data_key or raw}", "contentType": {"sys": {"id": "cvPricingToken"}}},
        "fields": {
            "internalName": internal_name or fee_data_key or raw,
            "feeDataKey": fee_data_key or raw,
            "value": raw,
            "displayValue": raw,
        },
    }


def _rich_text(text: str, links: list[dict] | None = None) -> dict:
    nodes: list[dict] = []
    if links:
        parts = []
        start = 0
        for link in links:
            idx = text.find(link["text"], start)
            if idx == -1:
                idx = len(text)
            if idx > start:
                parts.append(("text", text[start:idx]))
            parts.append(("link", link))
            start = idx + len(link["text"])
        if start < len(text):
            parts.append(("text", text[start:]))
        for kind, part in parts:
            if kind == "text":
                nodes.append({"nodeType": "text", "value": part, "marks": []})
            else:
                nodes.append(
                    {
                        "nodeType": "hyperlink",
                        "data": {"uri": part["uri"]},
                        "content": [{"nodeType": "text", "value": part["text"], "marks": []}],
                    }
                )
    else:
        nodes.append({"nodeType": "text", "value": text, "marks": []})
    return {"nodeType": "document", "content": [{"nodeType": "paragraph", "content": nodes}]}


def _render_context(country_code: str, country_name: str, locale: str) -> dict:
    # Commercial table with a percentage token and fixed-fee rows.
    commercial_table = {
        "componentType": "FeeTable",
        "componentId": f"ft-commercial-{country_code}",
        "documentId": f"FEETB-{country_code}-01",
        "content": {
            "caption": "Commercial transaction fees",
            "columns": [
                _rich_text("Payment type"),
                _rich_text("Fee"),
            ],
            "rows": [
                {
                    "cells": [
                        _rich_text("Commercial transactions"),
                        _rich_text(
                            "2,99% + 0,39 EUR",
                            links=[{"text": "0,39 EUR", "uri": "#fixed-fee"}],
                        ),
                    ]
                },
                {
                    "cells": [
                        _rich_text("Charity / nonprofit"),
                        _rich_text("1,99% + 0,39 EUR"),
                    ]
                },
            ],
        },
    }
    fixed_fee_table = {
        "componentType": "FeeTable",
        "componentId": f"ft-fixed-{country_code}",
        "documentId": f"FEETB-{country_code}-02",
        "content": {
            "caption": "Fixed fee by received currency",
            "columns": [
                _rich_text("Currency"),
                _rich_text("Fixed fee"),
            ],
            "rows": [
                {"cells": [_rich_text("EUR"), _rich_text("0,39 EUR")]},
                {"cells": [_rich_text("USD"), _rich_text("0,49 USD")]},
                {"cells": [_rich_text("GBP"), _rich_text("0,29 GBP")]},
                {"cells": [_rich_text("CHF"), _rich_text("0,39 CHF")]},
            ],
        },
    }
    international_table = {
        "componentType": "FeeTable",
        "componentId": f"ft-intl-{country_code}",
        "documentId": f"FEETB-{country_code}-03",
        "content": {
            "caption": "International surcharge",
            "columns": [
                _rich_text("Payer region"),
                _rich_text("Surcharge"),
            ],
            "rows": [
                {"cells": [_rich_text("EEA"), _rich_text("0%")]},
                {"cells": [_rich_text("GB"), _rich_text("+1,29%")]},
                {"cells": [_rich_text("Other"), _rich_text("+1,99%")]},
            ],
        },
    }
    conversion_table = {
        "componentType": "FeeTable",
        "componentId": f"ft-conv-{country_code}",
        "documentId": f"FEETB-{country_code}-04",
        "content": {
            "caption": "Currency conversion",
            "columns": [_rich_text("Currency conversion"), _rich_text("Spread")],
            "rows": [
                {"cells": [_rich_text("Currency conversion"), _rich_text("3%")]},
            ],
        },
    }
    # Split table with identical caption to test preservation.
    split_table_a = {
        "componentType": "FeeTable",
        "componentId": f"ft-split-a-{country_code}",
        "documentId": f"FEETB-{country_code}-05",
        "content": {
            "caption": "Currency fixed fees",
            "columns": [_rich_text("Currency"), _rich_text("Fee")],
            "rows": [
                {"cells": [_rich_text("EUR"), _rich_text("0,39 EUR")]},
                {"cells": [_rich_text("USD"), _rich_text("0,49 USD")]},
            ],
        },
    }
    split_table_b = {
        "componentType": "FeeTable",
        "componentId": f"ft-split-b-{country_code}",
        "documentId": f"FEETB-{country_code}-05",
        "content": {
            "caption": "Currency fixed fees",
            "columns": [_rich_text("Currency"), _rich_text("Fee")],
            "rows": [
                {"cells": [_rich_text("GBP"), _rich_text("0,29 GBP")]},
                {"cells": [_rich_text("CHF"), _rich_text("0,39 CHF")]},
            ],
        },
    }
    # FeeTableReference pointing to the split table.
    reference = {
        "componentType": "FeeTableReference",
        "componentId": f"ft-ref-{country_code}",
        "feeTableDocumentId": f"FEETB-{country_code}-05",
    }

    return {
        "pageId": "business/paypal-business-fees",
        "pageName": "PayPal Merchant and Seller Fees",
        "pageTitle": f"PayPal Merchant and Seller Fees - {country_name}",
        "pageUpdatedAt": "2026-04-30",
        "pageReference": {
            "id": "business/paypal-business-fees",
            "pageModel": {
                "middle": [
                    {
                        "componentType": "TextSectionType",
                        "componentId": f"sec-intro-{country_code}",
                        "content": {
                            "heading": _rich_text("Commercial transaction fees"),
                            "body": _rich_text("Fees for receiving domestic payments."),
                        },
                    },
                    {
                        "componentType": "FeeTableSection",
                        "componentId": f"sec-fees-{country_code}",
                        "content": {
                            "heading": _rich_text("Fee tables"),
                            "tables": [
                                commercial_table,
                                fixed_fee_table,
                                international_table,
                                conversion_table,
                                split_table_a,
                                split_table_b,
                                reference,
                            ],
                        },
                    },
                ]
            },
        },
        "countrySelector": {
            "componentType": "CountrySelector",
            "regions": [
                {
                    "region": "Europe",
                    "countries": [
                        {
                            "countryCode": "DE",
                            "countryName": "Germany",
                            "defaultLocale": "de_DE",
                            "languages": [{"language": "de", "languageName": "Deutsch"}],
                        },
                        {
                            "countryCode": "GB",
                            "countryName": "United Kingdom",
                            "defaultLocale": "en_GB",
                            "languages": [{"language": "en", "languageName": "English"}],
                        },
                    ],
                },
                {
                    "region": "North America",
                    "countries": [
                        {
                            "countryCode": "US",
                            "countryName": "United States",
                            "defaultLocale": "en_US",
                            "languages": [{"language": "en", "languageName": "English"}],
                        }
                    ],
                },
            ],
        },
    }


def _make_html(country_code: str, country_name: str, locale: str) -> str:
    ctx = _render_context(country_code, country_name, locale)
    ctx_json = json.dumps(ctx, ensure_ascii=False, separators=(",", ":"))
    return (
        "<!DOCTYPE html><html><head><title>PayPal Fees</title></head><body>"
        f"<script>window.__CMS_ENGINE_RENDER_CONTEXT__ = {ctx_json};</script>"
        "</body></html>"
    )


def main() -> None:
    fixtures = [
        ("de", "Germany", "de_DE"),
        ("us", "United States", "en_US"),
        ("gb", "United Kingdom", "en_GB"),
    ]
    for cc, name, locale in fixtures:
        path = Path(__file__).with_name(f"{cc}.html")
        path.write_text(_make_html(cc, name, locale), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
