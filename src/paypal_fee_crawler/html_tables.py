"""HTML table extraction fallback for PayPal fee pages.

PayPal's public fee pages no longer embed the CMS JSON render context. This
module parses the rendered HTML and produces the same ``Table``/``Section``
objects that the CMS component extractor produces, so the existing
classification pipeline can run unchanged.
"""

from __future__ import annotations

import logging

from lxml import html

from .models import Cell, Link, ParserWarning, Row, Section, Table, TableHeader
from .normalize import clean_text
from .pricing_tokens import tokenize_text

logger = logging.getLogger(__name__)


def _extract_text(element: html.HtmlElement, include_links: bool = True) -> tuple[str, list[Link]]:
    """Return normalized text and any links found inside *element*."""
    links: list[Link] = []
    if include_links:
        for anchor in element.iter("a"):
            href = anchor.get("href")
            if href:
                link_text = clean_text(anchor.text_content() or "")
                links.append(Link(text=link_text or None, uri=href))
    text = clean_text(element.text_content() or "")
    return text, links


def _cell(element: html.HtmlElement) -> Cell:
    """Build a normalized Cell from an HTML table cell."""
    text, links = _extract_text(element)
    tokens = tokenize_text(text)
    return Cell(text=text, tokens=tokens, links=links)


def _header_cell(element: html.HtmlElement) -> TableHeader:
    """Build a normalized TableHeader from an HTML table header cell."""
    text, links = _extract_text(element)
    tokens = tokenize_text(text)
    return TableHeader(text=text, tokens=tokens, links=links)


def _extract_table_rows(table: html.HtmlElement) -> tuple[list[TableHeader], list[Row]]:
    """Return (headers, rows) for a single HTML table element."""
    headers: list[TableHeader] = []
    rows: list[Row] = []

    thead = table.find("thead")
    if thead is not None:
        header_row = thead.find("tr")
        if header_row is not None:
            headers = [_header_cell(th) for th in header_row.iter("th")]

    tbody = table.find("tbody")
    if tbody is None:
        tbody = table
    body_rows = tbody.findall("tr")

    if not headers and body_rows:
        # First row may be a header row if it contains <th> cells.
        first_row = body_rows[0]
        header_cells = list(first_row.iter("th"))
        if header_cells:
            headers = [_header_cell(th) for th in header_cells]
            body_rows = body_rows[1:]

    for idx, tr in enumerate(body_rows):
        cells = [_cell(td) for td in tr.iter(("td", "th"))]
        if cells:
            rows.append(Row(row_id=str(idx), cells=cells))

    return headers, rows


def _section_path_for(element: html.HtmlElement) -> list[str]:
    """Return the heading hierarchy that precedes *element* in document order."""
    path: list[str] = []
    heading_levels: dict[int, str] = {}

    preceding = element.xpath(
        "preceding::h1 | preceding::h2 | preceding::h3 | preceding::h4 | preceding::h5 | preceding::h6"
    )
    for heading in preceding:
        level = int(heading.tag[1])
        text = clean_text(heading.text_content() or "")
        if not text:
            continue
        # A new heading at this level replaces any deeper or equal headings.
        for lvl in list(heading_levels):
            if lvl >= level:
                heading_levels.pop(lvl, None)
        heading_levels[level] = text

    # Build path from highest level to lowest.
    for level in sorted(heading_levels):
        path.append(heading_levels[level])
    return path


def _extract_sections(html_tree: html.HtmlElement) -> list[Section]:
    """Extract text sections from headings for page structure."""
    sections: list[Section] = []
    for heading in html_tree.iter(("h1", "h2", "h3", "h4", "h5", "h6")):
        text = clean_text(heading.text_content() or "")
        if not text:
            continue
        sections.append(Section(heading=text))
    return sections


def _parse_html_tree(html_text: str) -> html.HtmlElement | None:
    """Return a parsed lxml tree or None on failure."""
    try:
        return html.fromstring(html_text)
    except Exception:
        return None


def extract_html_tables(
    html_text: str,
    page_url: str | None = None,
    tree: html.HtmlElement | None = None,
) -> tuple[list[Section], list[Table], list[ParserWarning]]:
    """Parse *html_text* and extract normalized tables and sections.

    Returns the same ``(sections, tables, warnings)`` tuple as the CMS
    ``ComponentsExtractor``, making it a drop-in fallback when the CMS render
    context is absent.  A pre-parsed ``tree`` may be supplied to avoid
    re-parsing the same HTML.
    """
    warnings: list[ParserWarning] = []
    if tree is None:
        tree = _parse_html_tree(html_text)
    if tree is None:
        logger.warning("Failed to parse HTML")
        warnings.append(ParserWarning(code="html_parse_error", message="Failed to parse HTML"))
        return [], [], warnings

    sections = _extract_sections(tree)
    tables: list[Table] = []

    for source_order, table_element in enumerate(tree.iter("table")):
        section_path = _section_path_for(table_element)
        headers, rows = _extract_table_rows(table_element)
        if not rows:
            continue

        caption = ""
        caption_el = table_element.find("caption")
        if caption_el is not None:
            caption = clean_text(caption_el.text_content() or "")
        if not caption and section_path:
            caption = section_path[-1]

        tables.append(
            Table(
                component_type="FeeTable",
                component_id=f"html-table-{source_order}",
                document_id=page_url or None,
                caption=caption or None,
                section_path=section_path,
                parent_path=list(section_path),
                source_order=source_order,
                column_count=len(headers) or (len(rows[0].cells) if rows else 0),
                headers=headers,
                rows=rows,
            )
        )

    return sections, tables, warnings


def extract_html_pdf_url(html_text: str, tree: html.HtmlElement | None = None) -> str | None:
    """Find a printable PDF fee schedule link in the raw HTML."""
    if tree is None:
        tree = _parse_html_tree(html_text)
    if tree is None:
        return None
    for anchor in tree.iter("a"):
        href = anchor.get("href")
        if href and "pdf" in href.lower():
            return href
    return None


def extract_html_locale(html_text: str, tree: html.HtmlElement | None = None) -> str | None:
    """Return the page locale from the HTML ``lang`` attribute if present."""
    if tree is None:
        tree = _parse_html_tree(html_text)
    if tree is None:
        return None
    lang = tree.get("lang")
    if isinstance(lang, str) and lang.strip():
        return lang.strip()
    return None
