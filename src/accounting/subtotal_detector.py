"""
Subtotal Detection with Footing Validation — §16.

Identifies subtotal and total rows in a financial table, then validates
them against their children using arithmetic footing:
  ``subtotal_value == sum(child_values)``

Subtotals are detected by:
  1. Nomenclature entries with ``is_subtotal == True``
  2. Label prefix matching ("Total des …", "Sous-total …")
  3. Bold/separator font hints (when available)

Public API:

    detect_subtotals(df, column_layout, dictionary=None) → list[SubtotalResult]
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Subtotal label patterns (French)
_SUBTOTAL_PREFIXES = [
    "total des ",
    "total ",
    "sous-total ",
    "sous total ",
    "totaux ",
]

_SUBTOTAL_RE = re.compile(
    r"^(total\s+des?|sous[\s-]total|totaux)\b",
    re.IGNORECASE,
)


@dataclass
class SubtotalResult:
    """Result of subtotal detection for a single row."""

    row_index: int
    label: str
    is_subtotal: bool
    canonical_term: Optional[str] = None
    # Footing validation
    expected_value: Optional[Decimal] = None  # sum of children
    actual_value: Optional[Decimal] = None  # value in the cell
    footing_valid: Optional[bool] = None  # True if match, None if not checkable
    child_indices: list[int] = field(default_factory=list)
    tolerance: Decimal = field(default_factory=lambda: Decimal("1"))


def detect_subtotals(
    df: pd.DataFrame,
    column_layout=None,
    dictionary=None,
    sections: Optional[list[Optional[str]]] = None,
    parsed_numbers: Optional[dict[tuple[int, int], Decimal]] = None,
) -> list[SubtotalResult]:
    """Detect subtotal rows and validate footing.

    Args:
        df: Table DataFrame.
        column_layout: ColumnLayout from column_identifier.
        dictionary: NomenclatureDictionary for is_subtotal flag.
        sections: Section assignments per row (from header_detector).
        parsed_numbers: Pre-parsed Decimal values keyed by (row, col).

    Returns:
        List of SubtotalResult for every detected subtotal row.
    """
    if df.empty or len(df.columns) < 2:
        return []

    label_col_idx = 0
    if column_layout and hasattr(column_layout, "label_col_index"):
        label_col_idx = column_layout.label_col_index

    results: list[SubtotalResult] = []

    for row_idx in range(len(df)):
        label = str(df.iloc[row_idx, label_col_idx]).strip()

        # Check if this is a subtotal row
        is_sub = _is_subtotal_row(label, row_idx, dictionary)
        if not is_sub:
            continue

        # Find children (rows between previous subtotal and this one)
        child_indices = _find_children(
            row_idx, results, sections, len(df)
        )

        result = SubtotalResult(
            row_index=row_idx,
            label=label,
            is_subtotal=True,
            child_indices=child_indices,
        )

        # Footing validation (if parsed numbers available)
        if parsed_numbers and column_layout:
            _validate_footing(result, df, column_layout, parsed_numbers)

        results.append(result)

    logger.info(f"Subtotal detection: found {len(results)} subtotal rows")
    return results


def _is_subtotal_row(
    label: str,
    row_idx: int,
    dictionary=None,
) -> bool:
    """Determine if a row is a subtotal/total by label or dictionary flag."""
    label_lower = label.lower().strip()

    if not label_lower:
        return False

    # Check 1 — Dictionary entry has is_subtotal flag
    if dictionary is not None:
        mr = dictionary.fuzzy_match(label)
        if mr.entry and getattr(mr.entry, "is_subtotal", False):
            return True

    # Check 2 — Label prefix matching
    if _SUBTOTAL_RE.match(label_lower):
        return True

    # Check 3 — Exact "total" (standalone)
    if label_lower == "total":
        return True

    return False


def _find_children(
    subtotal_idx: int,
    previous_subtotals: list[SubtotalResult],
    sections: Optional[list[Optional[str]]],
    n_rows: int,
) -> list[int]:
    """Find child row indices for a subtotal.

    Children are the rows between the previous subtotal (or table start)
    and this subtotal, within the same section.
    """
    # Start from after the previous subtotal (or row 0)
    if previous_subtotals:
        start = previous_subtotals[-1].row_index + 1
    else:
        start = 0

    # If we have section info, only include rows in the same section
    if sections and subtotal_idx < len(sections):
        my_section = sections[subtotal_idx]
        children = [
            i
            for i in range(start, subtotal_idx)
            if sections[i] == my_section
        ]
    else:
        children = list(range(start, subtotal_idx))

    # Exclude other subtotals from the children list
    subtotal_indices = {s.row_index for s in previous_subtotals}
    children = [i for i in children if i not in subtotal_indices]

    return children


def _validate_footing(
    result: SubtotalResult,
    df: pd.DataFrame,
    column_layout,
    parsed_numbers: dict[tuple[int, int], Decimal],
) -> None:
    """Validate that subtotal == sum(children) for the first numeric column."""
    if not result.child_indices:
        return

    # Pick the first numeric column for validation
    numeric_cols = getattr(column_layout, "numeric_col_indices", [])
    if not numeric_cols:
        return

    check_col = numeric_cols[0]

    # Get subtotal value
    subtotal_key = (result.row_index, check_col)
    subtotal_val = parsed_numbers.get(subtotal_key)
    if subtotal_val is None:
        return

    result.actual_value = subtotal_val

    # Sum children
    child_sum = Decimal("0")
    for child_idx in result.child_indices:
        child_key = (child_idx, check_col)
        child_val = parsed_numbers.get(child_key)
        if child_val is not None:
            child_sum += child_val

    result.expected_value = child_sum

    # Compare with tolerance (±1 for rounding)
    diff = abs(subtotal_val - child_sum)
    result.footing_valid = diff <= result.tolerance

    if not result.footing_valid:
        logger.warning(
            f"Footing mismatch at row {result.row_index} '{result.label}': "
            f"actual={subtotal_val}, expected={child_sum}, diff={diff}"
        )
