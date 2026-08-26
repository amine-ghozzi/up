"""
French-locale number parser.

Implements the 3-layer parser specified in §22 of the nomenclature revamp plan:

    Layer 1 — OCR error correction (character substitutions, numeric-context only)
    Layer 2 — Locale normalization (parens → neg, dashes → 0, NBSP removal, …)
    Layer 3 — Babel ``parse_decimal`` with a plain-digit fallback

Public surface:

    parse_french_number(raw) -> Decimal | None
    is_numeric_token(text)   -> bool
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Optional

from babel.numbers import parse_decimal
from babel.core import UnknownLocaleError

# ---------------------------------------------------------------------------
# Layer 1 — OCR error correction
# ---------------------------------------------------------------------------

# Substitutions applied only when the token is in numeric context.
# (Applying them blindly would corrupt real labels, e.g. "Obligations".)
_OCR_DIGIT_SUBS = {
    "O": "0",
    "o": "0",
    "l": "1",  # lowercase L → one (context-gated)
    "I": "1",  # uppercase I → one (context-gated)
    "S": "5",  # rare OCR miss
    "B": "8",
}

# A token is "numeric context" when it is mostly digits + allowed separators
# plus at most a handful of OCR-confusable letters.
_NUMERIC_CONTEXT_RE = re.compile(r"^[\d.,\s()OolISB\xa0 ‑–—\-+]+$")
_HAS_DEFINITE_DIGIT_RE = re.compile(r"[2-9]")


def _apply_ocr_fixes(text: str) -> str:
    """Layer 1: swap OCR-confusable glyphs for digits, only in numeric context."""
    if not _NUMERIC_CONTEXT_RE.match(text):
        return text
    # Heuristic gate: require at least one definite digit (2-9) before swapping,
    # OR the whole token must be only confusable chars + separators. This avoids
    # turning "OOO" (unlikely but safe) into "000" without evidence.
    if not _HAS_DEFINITE_DIGIT_RE.search(text):
        # No definite digit: still allow substitution if the token length >= 2
        # and at least one separator is present (e.g. "l.500" → "1.500").
        if not re.search(r"[.,\s]", text):
            return text
    return "".join(_OCR_DIGIT_SUBS.get(ch, ch) for ch in text)


# ---------------------------------------------------------------------------
# Layer 2 — Locale normalization
# ---------------------------------------------------------------------------

_PAREN_RE = re.compile(r"^\((.+)\)$")
_STANDALONE_DASH_RE = re.compile(r"^[\-–—‑]+$")
_NBSP_CHARS = ("\xa0", " ", " ")  # NBSP, narrow NBSP, thin space
_CURRENCY_TRAILERS_RE = re.compile(
    r"\s*(DT|TND|EUR|USD|€|\$|DH|MAD)\s*$", flags=re.IGNORECASE
)
_CURRENCY_LEADERS_RE = re.compile(
    r"^\s*(DT|TND|EUR|USD|€|\$|DH|MAD)\s+", flags=re.IGNORECASE
)


def _normalize_locale(text: str) -> str:
    """Layer 2: parens → negative, dashes → 0, NBSP + currency stripping."""
    s = text.strip()
    if not s:
        return s

    # Standalone dash (possibly multiple) → zero (NCT convention for empty cells)
    if _STANDALONE_DASH_RE.match(s):
        return "0"

    # Parentheses → negative sign (accounting convention)
    paren_match = _PAREN_RE.match(s)
    if paren_match:
        s = "-" + paren_match.group(1).strip()

    # Remove non-breaking spaces and thin spaces (French thousands separators)
    for ch in _NBSP_CHARS:
        s = s.replace(ch, "")

    # Strip currency markers
    s = _CURRENCY_LEADERS_RE.sub("", s)
    s = _CURRENCY_TRAILERS_RE.sub("", s)

    return s.strip()


# ---------------------------------------------------------------------------
# Layer 3 — Locale-aware parsing
# ---------------------------------------------------------------------------


def _fallback_parse(cleaned: str) -> Optional[Decimal]:
    """Plain fallback: strip spaces/dots (thousands), swap comma → dot."""
    stripped = cleaned.replace(" ", "")
    # If it has BOTH "." and ",", treat "." as thousands and "," as decimal
    if "." in stripped and "," in stripped:
        stripped = stripped.replace(".", "").replace(",", ".")
    elif "," in stripped:
        # Only comma → decimal separator in French
        stripped = stripped.replace(",", ".")
    else:
        # Only "." might be decimal OR thousands. If the last dot-group is 3
        # digits and the rest looks like a thousands pattern, treat all dots
        # as thousands. Otherwise treat as decimal.
        if re.match(r"^-?\d{1,3}(\.\d{3})+$", stripped):
            stripped = stripped.replace(".", "")
    try:
        return Decimal(stripped)
    except Exception:  # noqa: BLE001 — Decimal raises InvalidOperation
        return None


def parse_french_number(raw: object) -> Optional[Decimal]:
    """Parse a French-formatted number string to ``Decimal``.

    Handles OCR artifacts (``O`` ↔ ``0``), French thousands/decimal separators
    (``1 500,00`` or ``1.500,00``), parenthesised negatives, and dash-as-zero.

    Returns ``None`` when the input cannot be coerced to a number.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, Decimal)):
        return Decimal(raw)
    if isinstance(raw, float):
        return Decimal(str(raw))
    if not isinstance(raw, str):
        try:
            return Decimal(str(raw))
        except Exception:  # noqa: BLE001
            return None

    text = raw.strip()
    if not text:
        return None

    # Layer 1 — OCR fixes (numeric context only)
    text = _apply_ocr_fixes(text)

    # Layer 2 — Locale normalization (parens, dashes, NBSP, currency)
    text = _normalize_locale(text)
    if text in ("", "0"):
        return Decimal("0") if text == "0" else None

    # Layer 3 — Babel locale-aware parser, with plain fallback on failure
    try:
        return parse_decimal(text, locale="fr_FR")
    except (UnknownLocaleError, Exception):  # noqa: BLE001 — Babel NumberFormatError
        return _fallback_parse(text)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


def is_numeric_token(text: str) -> bool:
    """Return True if ``text`` could plausibly represent a French number."""
    if not isinstance(text, str):
        return False
    candidate = text.strip()
    if not candidate:
        return False
    return parse_french_number(candidate) is not None
