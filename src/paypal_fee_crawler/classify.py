"""Derive core merchant fees from normalized tables with conservative confidence."""

from __future__ import annotations

import logging

from .models import (
    CommercialFee,
    CurrencyConversion,
    DerivedFees,
    FixedFees,
    InternationalSurcharge,
    Table,
)
from .normalize import clean_text

logger = logging.getLogger(__name__)


def _section_text(table: Table) -> str:
    parts = list(table.section_path)
    if table.caption:
        parts.append(table.caption)
    return clean_text(" ".join(parts)).lower()


def _first_percentage_token(table: Table) -> str | None:
    for row in table.rows:
        for cell in row.cells:
            for token in cell.tokens:
                if token.kind == "percentage" and token.value:
                    return token.value
    for header in table.headers:
        for token in header.tokens:
            if token.kind == "percentage" and token.value:
                return token.value
    return None


def _collect_money_tokens(table: Table) -> list[FixedFees]:
    """Collect money tokens, optionally paired with a preceding currency column."""
    fees: list[FixedFees] = []
    header_currencies: list[str] = []
    for header in table.headers:
        for token in header.tokens:
            if token.currency:
                header_currencies.append(token.currency)
        if not header_currencies:
            # Try to parse a currency code from the header text itself.
            parts = header.text.split()
            if parts and parts[-1].isalpha() and len(parts[-1]) == 3:
                header_currencies.append(parts[-1].upper())

    for row in table.rows:
        cells = row.cells
        for idx, cell in enumerate(cells):
            for token in cell.tokens:
                if token.kind == "money" and token.amount and token.currency:
                    fees.append(FixedFees(currency=token.currency, amount=token.amount))
                elif token.kind == "number" and token.value:
                    # If a currency is inferable from a header, treat it as money.
                    currency = None
                    if idx < len(header_currencies):
                        currency = header_currencies[idx]
                    else:
                        # Look for currency code in the same cell text.
                        parts = cell.text.split()
                        if parts and parts[-1].isalpha() and len(parts[-1]) == 3:
                            currency = parts[-1].upper()
                    if currency:
                        fees.append(FixedFees(currency=currency, amount=token.value))
    return fees


def _find_surcharge_regions(table: Table) -> list[InternationalSurcharge]:
    """Extract international surcharge by payer region from a table."""
    surcharges: list[InternationalSurcharge] = []
    for row in table.rows:
        region: str | None = None
        percentage: str | None = None
        for cell in row.cells:
            text = cell.text.strip().upper()
            if text in {"EEA", "EU", "GB", "UK", "US", "CA", "OTHER", "REST OF WORLD", "INTERNATIONAL"}:
                region = text if text not in {"UK"} else "GB"
                if region == "EU":
                    region = "EEA"
                if region == "REST OF WORLD":
                    region = "OTHER"
            for token in cell.tokens:
                if token.kind == "percentage" and token.value:
                    percentage = token.value
        if region and percentage:
            surcharges.append(InternationalSurcharge(region=region, percentage_points=percentage))
    return surcharges


def _find_conversion_spread(table: Table) -> str | None:
    for row in table.rows:
        for cell in row.cells:
            for token in cell.tokens:
                if token.kind == "percentage" and token.value:
                    return token.value
    return None


def classify_tables(tables: list[Table]) -> DerivedFees:
    """Derive core fees from normalized tables with explicit confidence status."""
    derived = DerivedFees(status="unclassified")
    if not tables:
        return derived

    commercial_percentage: str | None = None
    commercial_fixed_fees: list[FixedFees] = []
    international_surcharges: list[InternationalSurcharge] = []
    conversion_spread: str | None = None
    unclassified: list[str] = []

    for table in tables:
        text = _section_text(table)
        if not table.rows and not table.headers:
            continue

        is_commercial = any(keyword in text for keyword in ("commercial", "standard", "domestic", "goods and services"))
        is_fixed = any(keyword in text for keyword in ("fixed fee", "fixed fees", "per transaction"))
        is_international = any(keyword in text for keyword in ("international", "cross border", "cross-border"))
        is_conversion = any(keyword in text for keyword in ("currency conversion", "conversion spread", "fx"))
        is_micropayment = "micropayment" in text
        is_donation = "donation" in text
        is_nonprofit = "nonprofit" in text or "non-profit" in text or "charity" in text
        is_chargeback = "chargeback" in text
        is_dispute = "dispute" in text

        if is_commercial:
            percentage = _first_percentage_token(table)
            if percentage:
                commercial_percentage = percentage

        if is_fixed or (is_commercial and not is_international and not is_conversion):
            fees = _collect_money_tokens(table)
            if fees:
                commercial_fixed_fees.extend(fees)

        if is_international:
            surcharges = _find_surcharge_regions(table)
            if surcharges:
                international_surcharges.extend(surcharges)
            elif _first_percentage_token(table):
                # Could not split by region; add a generic OTHER entry.
                international_surcharges.append(
                    InternationalSurcharge(region="OTHER", percentage_points=_first_percentage_token(table))
                )

        if is_conversion:
            spread = _find_conversion_spread(table)
            if spread:
                conversion_spread = spread

        # Mark other known categories as unclassified rather than guessing values.
        if is_micropayment:
            unclassified.append("micropayment")
        if is_donation:
            unclassified.append("donation")
        if is_nonprofit:
            unclassified.append("nonprofit")
        if is_chargeback:
            unclassified.append("chargeback")
        if is_dispute:
            unclassified.append("dispute")

    status = "unclassified"
    if commercial_percentage or commercial_fixed_fees or international_surcharges or conversion_spread:
        status = "partial"
    if commercial_percentage and commercial_fixed_fees:
        status = "complete"

    return DerivedFees(
        status=status,
        standard_commercial=CommercialFee(
            percentage=commercial_percentage,
            fixed_fee_reference="commercial_fixed_fees" if commercial_fixed_fees else None,
        )
        if commercial_percentage
        else None,
        commercial_fixed_fees=commercial_fixed_fees,
        international_surcharges=international_surcharges,
        currency_conversion=CurrencyConversion(spread_percentage=conversion_spread) if conversion_spread else None,
        unclassified_sections=sorted(set(unclassified)),
    )
