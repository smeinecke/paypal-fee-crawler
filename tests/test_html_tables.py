"""Tests for the HTML table extraction fallback."""

from __future__ import annotations

from unittest.mock import patch

from lxml import html

from paypal_fee_crawler.html_tables import (
    extract_html_locale,
    extract_html_pdf_url,
    extract_html_tables,
)
from paypal_fee_crawler.pricing_tokens import tokenize_text


def test_extract_html_tables_basic() -> None:
    html = """
    <html lang="de">
    <head><title>PayPal Fees</title></head>
    <body>
        <h1>Merchant Fees</h1>
        <h2>Domestic</h2>
        <table>
            <caption>Standard fees</caption>
            <thead><tr><th>Type</th><th>Fee</th></tr></thead>
            <tbody>
                <tr><td>Checkout</td><td>2.99% + 0.35 EUR</td></tr>
                <tr><td>QR</td><td>0.90%</td></tr>
            </tbody>
        </table>
    </body>
    </html>
    """
    sections, tables, warnings = extract_html_tables(html)
    assert not warnings
    assert len(sections) >= 2
    assert len(tables) == 1
    table = tables[0]
    assert table.caption == "Standard fees"
    assert table.section_path == ["Merchant Fees", "Domestic"]
    assert len(table.headers) == 2
    assert table.headers[0].text == "Type"
    assert table.headers[1].text == "Fee"
    assert len(table.rows) == 2
    assert table.rows[0].cells[0].text == "Checkout"
    assert table.rows[0].cells[1].text == "2.99% + 0.35 EUR"
    assert table.column_count == 2


def test_extract_html_tables_row_header() -> None:
    html = """
    <h1>International</h1>
    <table>
        <tr><th>Market</th><th>Surcharge</th></tr>
        <tr><td>UK</td><td>+1.29%</td></tr>
    </table>
    """
    sections, tables, warnings = extract_html_tables(html)
    assert not warnings
    assert len(tables) == 1
    table = tables[0]
    assert table.section_path == ["International"]
    assert len(table.headers) == 2
    assert table.headers[1].text == "Surcharge"
    assert len(table.rows) == 1


def test_extract_html_tables_no_rows_skipped() -> None:
    html = """
    <table>
        <thead><tr><th>A</th><th>B</th></tr></thead>
    </table>
    """
    sections, tables, warnings = extract_html_tables(html)
    assert not warnings
    assert len(tables) == 0


def test_extract_html_tables_links() -> None:
    html = """
    <table>
        <tr><th>Resource</th></tr>
        <tr><td><a href="https://example.com/pdf">PDF</a></td></tr>
    </table>
    """
    sections, tables, warnings = extract_html_tables(html)
    assert not warnings
    cell = tables[0].rows[0].cells[0]
    assert cell.text == "PDF"
    assert len(cell.links) == 1
    assert cell.links[0].uri == "https://example.com/pdf"


def test_extract_html_tables_tokenizes_cells() -> None:
    html = """
    <table>
        <tr><th>Fee</th></tr>
        <tr><td>2.49% + 0.35 EUR</td></tr>
    </table>
    """
    sections, tables, warnings = extract_html_tables(html)
    cell = tables[0].rows[0].cells[0]
    assert cell.tokens == tokenize_text("2.49% + 0.35 EUR")


def test_extract_html_tables_bad_html_returns_warning() -> None:
    with patch.object(html, "fromstring", side_effect=ValueError("bad html")):
        sections, tables, warnings = extract_html_tables("<bad>")
    assert len(warnings) == 1
    assert warnings[0].code == "html_parse_error"


def test_extract_html_pdf_url() -> None:
    html = '<a href="https://www.paypal.com/de/fees.pdf">PDF</a>'
    assert extract_html_pdf_url(html) == "https://www.paypal.com/de/fees.pdf"


def test_extract_html_pdf_url_missing() -> None:
    html = '<a href="https://www.paypal.com/de/">Home</a>'
    assert extract_html_pdf_url(html) is None


def test_extract_html_locale() -> None:
    html = '<html lang="de-DE"><body></body></html>'
    assert extract_html_locale(html) == "de-DE"


def test_extract_html_locale_missing() -> None:
    html = "<html><body></body></html>"
    assert extract_html_locale(html) is None
