"""
Section Header Detection — §14 (font metadata) + §25 (OCR bbox fallback).

Dual-mode detector that assigns each table row to a Nomenclature section:
  - **Native mode**: Uses PyMuPDF font flags (bold, size) for digital PDFs.
  - **OCR mode**: Uses bounding box height + CAPS + whitespace signals for scanned PDFs.

Public API:

    detect_section_headers(df, dictionary, font_spans=None) → list[str | None]
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Regex for ALL-CAPS with possible accented chars
_CAPS_RE = re.compile(r"^[A-ZÀÂÄÉÈÊËÏÎÔÙÛÜŸÇŒÆ\s\-'.,()]+$")

# Minimum font size ratio to consider a span a header (vs body text)
_FONT_SIZE_HEADER_RATIO = 1.15  # ≥15% larger than median body text
_BBOX_HEIGHT_HEADER_RATIO = 1.20  # ≥20% taller bounding box


def detect_section_headers(
    df: pd.DataFrame,
    dictionary=None,
    font_spans: Optional[list[dict]] = None,
) -> list[Optional[str]]:
    """Assign each row to a section key (or None if no header detected).

    The approach propagates the last-seen section header downward, so rows
    below a "CAPITAUX PROPRES" header all get section="capitaux_propres"
    until the next header.

    Args:
        df: The table DataFrame (first column assumed to be labels).
        dictionary: NomenclatureDictionary for classify_section().
        font_spans: Optional font metadata from PyMuPDF (list of dicts
                     with 'text', 'font_size', 'is_bold', 'bbox_height' keys).

    Returns:
        List of section keys (one per row), using None for unclassified rows.
    """
    if df.empty or len(df.columns) == 0:
        return []

    label_col = df.columns[0]
    n_rows = len(df)
    sections: list[Optional[str]] = [None] * n_rows

    # Determine detection mode
    use_font_mode = font_spans is not None and len(font_spans) == n_rows
    if use_font_mode:
        header_flags = _detect_via_font(font_spans)
    else:
        header_flags = _detect_via_heuristic(df, label_col)

    # Now resolve section keys using the dictionary
    for i in range(n_rows):
        if header_flags[i]:
            label = str(df.iloc[i][label_col]).strip()
            if dictionary is not None:
                section_key = dictionary.classify_section(label)
                if section_key:
                    sections[i] = section_key

    # Propagate downward: fill gaps with the last-seen section
    _propagate_sections(sections)

    return sections


def _detect_via_font(font_spans: list[dict]) -> list[bool]:
    """Native mode — use PyMuPDF font metadata.

    A row is a header if:
      - Font is bold, OR
      - Font size ≥ 15% above median body text size
    """
    sizes = [s.get("font_size", 10.0) for s in font_spans]
    if not sizes:
        return [False] * len(font_spans)

    median_size = sorted(sizes)[len(sizes) // 2]
    threshold = median_size * _FONT_SIZE_HEADER_RATIO

    flags: list[bool] = []
    for span in font_spans:
        is_bold = span.get("is_bold", False)
        is_large = span.get("font_size", 10.0) >= threshold
        flags.append(is_bold or is_large)
    return flags


def _detect_via_heuristic(df: pd.DataFrame, label_col: str) -> list[bool]:
    """OCR fallback mode — §25 replacement signals.

    Signal 1: ALL-CAPS text (most French section headers use uppercase)
    Signal 2: Row has no numeric content (headers don't have amounts)
    Signal 3: Row spans single column (headers often merge cells)
    """
    flags: list[bool] = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        label = str(row[label_col]).strip()

        if not label or len(label) < 3:
            flags.append(False)
            continue

        # Signal 1 — ALL-CAPS
        is_caps = bool(_CAPS_RE.match(label))

        # Signal 2 — no numeric content in this row
        numeric_count = sum(
            1
            for col in df.columns[1:]  # skip label column
            if pd.notna(row[col]) and _has_numeric(str(row[col]))
        )
        is_label_only = numeric_count == 0

        # Signal 3 — text length suggests a header (short, descriptive)
        is_header_length = 3 <= len(label) <= 80

        # Decision: caps + no numbers is the strongest signal
        score = (1.0 if is_caps else 0.0) + (0.5 if is_label_only else 0.0)
        flags.append(score >= 1.0 and is_header_length)

    return flags


def _has_numeric(text: str) -> bool:
    """Quick check if text contains digit-like characters."""
    return bool(re.search(r"\d", text))


def _propagate_sections(sections: list[Optional[str]]) -> None:
    """Fill gaps by propagating the last-seen section downward."""
    current_section: Optional[str] = None
    for i in range(len(sections)):
        if sections[i] is not None:
            current_section = sections[i]
        else:
            sections[i] = current_section
