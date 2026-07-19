from __future__ import annotations

import logging

from .patterns import (
    _APM_EXAMPLE_PHRASE_RE,
    _APM_HEADER_PHRASES,
    _APM_HEADER_TOKENS,
    _APM_METHOD_MATCHERS,
    _APM_PUNCTUATION_RE,
    _APM_SEPARATOR_RE,
    _APM_SPECIAL_METHOD_IDS,
)
from .text_utils import _keyword_match, _norm

logger = logging.getLogger(__name__)


def _tokenize_apm_label(part_norm: str) -> set[str]:
    """Tokenize an APM label part for robust method matching.

    Collapses multi-word method names (e.g. "go pay", "ovo premium") into a
    single token so they can be matched with word boundaries.
    """
    # Normalize punctuation to spaces, then pre-join multi-word brand names.
    joined = _APM_PUNCTUATION_RE.sub(" ", part_norm)
    joined = (
        joined.replace("go pay", "gopay")
        .replace("ovo premium", "ovopremium")
        .replace("floa pay", "floapay")
        .replace("blik pay later", "blikpaylater")
        .replace("naver pay", "naverpay")
        .replace("true money", "truemoney")
        .replace("pago efectivo", "pagoefectivo")
        .replace("mercado pago", "mercadopago")
        .replace("shopee pay", "shopeepay")
        .replace("m pesa", "mpesa")
        .replace("doku wallet", "dokuwallet")
        .replace("jenius pay", "jeniuspay")
        .replace("mixx by yas", "mixxbyyas")
        .replace("pay attitude", "payattitude")
        .replace("promptpay qr", "promptpayqr")
        .replace("bre b", "breb")
        .replace("viet qr", "vietqr")
        .replace("pay safe card", "paysafecard")
        .replace("paysafe card", "paysafecard")
        .replace("oxxo pay", "oxxopay")
        .replace("fiuu cash", "fiuucash")
        .replace("o pay", "opay")
        .replace("air tel", "airtel")
        .replace("kre divo", "kredivo")
        .replace("bankoverschrij ving", "bankoverschrijving")
        .replace("apple pay", "applepay")
    )
    return set(joined.split())


def _extract_apm_methods(label: str) -> tuple[list[str], list[str]]:
    """Extract canonical payment-method IDs from an APM row label.

    Returns (canonical_ids, unknown_segments). The canonical IDs are sorted and
    deduplicated. Unknown segments are raw label parts that did not match any
    known method. Token-based matching avoids false positives like "Republik"
    containing "blik" or "Thailändische Baht" containing "thai".
    """
    norm = _norm(label)
    if not norm:
        return [], []

    parts = [p.strip() for p in _APM_SEPARATOR_RE.split(label) if p.strip()]
    if not parts:
        parts = [label]

    methods: set[str] = set()
    unknowns: list[str] = []

    for part in parts:
        part_norm = _norm(part)
        if not part_norm or len(part_norm) < 3:
            continue

        # Drop introductory phrases like "e.g." or "z.b.".
        part_norm = _APM_EXAMPLE_PHRASE_RE.sub("", part_norm).strip()
        if not part_norm:
            continue

        # Skip generic header phrases ("Alternative payment method", "Alle anderen...").
        if _keyword_match(part_norm, _APM_HEADER_PHRASES, word_boundary=False):
            continue

        tokens = _tokenize_apm_label(part_norm)

        # Skip parts that are only header tokens.
        if tokens & _APM_HEADER_TOKENS and not (tokens - _APM_HEADER_TOKENS):
            continue

        matched: str | None = None
        for method_id, required_groups, forbidden in _APM_METHOD_MATCHERS:
            if tokens & forbidden:
                continue
            if all(tokens & group for group in required_groups):
                matched = method_id
                break

        if matched:
            methods.add(matched)
        else:
            unknowns.append(part)

    return sorted(methods), sorted(set(unknowns))


def _is_apm_special_label(label: str) -> bool:
    """Return True if a row label describes APM special methods.

    These labels list multiple alternative payment methods (e.g. Thai online
    bank transfer, Skrill, BLIK, Kredivo, etc.) and would otherwise be
    misclassified because they contain substrings like "pay later" (from
    "BLIK Pay Later") that collide with invoice_pay_later / pay_later_consumer.
    """
    methods, _ = _extract_apm_methods(label)
    if not methods:
        return False
    # A spurious "online_bank_transfer" match can be triggered by generic tokens
    # such as "on" + "bank" in a withdrawal/return row (e.g. "Bank Return on
    # Withdrawal/Transfer out of PayPal"). Do not treat those as APM special.
    if methods == ["online_bank_transfer"]:
        text = _norm(label)
        if _keyword_match(text, ("withdrawal", "return", "chargeback", "refund"), word_boundary=False):
            return False
    return any(m in _APM_SPECIAL_METHOD_IDS for m in methods)


def _is_international_label(label: str) -> bool:
    """Return True if the label describes an international/cross-border fee."""
    text = _norm(label)
    return _keyword_match(
        text,
        (
            "internationaux",
            "internacionais",
            "internacional",
            "internazionali",
            "foreign",
            "outside",
            "ausland",
            "ausländ",
            "außerhalb",
            "non-eea",
            "non eea",
            "non-eu",
            "non eu",
            "fuera de",
            "fuera",
            "hors",
            "estrangeiro",
            "estrangeira",
            "utenlands",
            "utland",
            "ulkomaan",
            "interna",
            "külföld",
            "külföldi",
            "međunarod",
            "međunarodne",
            "zahraniční",
            "zahraničné",
            "zagraniczny",
            "zagraniczne",
            "zagraniczna",
            "zagranicznych",
            "międzynarodowe",
            "międzynarodowej",
            "międzynarodowych",
            "międzynarodowa",
            "międzynarodowy",
            "mednarodne",
            "mednarodni",
            "kansainvälinen",
            "kansainvalinen",
            "διεθνείς",
            "διεθνεις",
        ),
        word_boundary=False,
    )


def _is_domestic_label(label: str) -> bool:
    """Return True if the label describes a domestic/in-country fee."""
    text = _norm(label)
    if "international" in text:
        return False
    return _keyword_match(
        text,
        (
            "domestic",
            "doméstico",
            "domésticos",
            "doméstica",
            "domésticas",
            "domesticos",
            "domesticas",
            "domestici",
            "domestic",
            "inland",
            "innenland",
            "innenlands",
            "national",
            "nacional",
            "nacionais",
            "national",
            "local",
            "lokal",
            "lokal",
            "nasional",
            "inland",
            "innenlands",
            "inländer",
            "inlander",
            "krajowe",
            "krajowa",
            "krajowych",
            "krajowy",
            "krajových",
            "domácich",
            "domestic",
            "εγχώριες",
            "εγχωριες",
        ),
        word_boundary=False,
    )
