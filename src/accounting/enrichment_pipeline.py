"""
Enrichment Pipeline — §23 DAG Orchestrator.

Converts raw DataFrames into progressively enriched ``CanonicalTable``
objects by running Stages 1–10 in correct dependency order:

    Stage 1: Multi-page table merge         (table_merger)
    Stage 2: Section header detection       (header_detector)
    Stage 3: Column identification          (column_identifier)
    Stage 4: FFFD resolution                (nomenclature.resolve_fffd)
    Stage 5: Number parsing                 (french_number)
    Stage 6: Fuzzy matching                 (nomenclature.fuzzy_match)
    Stage 7: Grouped entry detection        (grouped_entry)
    Stage 8: Subtotal detection + footing   (subtotal_detector)
    Stage 9: Document classification        (statement_segmenter)
    Stage 10: Flag assignment               (cell-level green/yellow/red)

Public API:

    enrich_tables(tables, dictionary, tier, ...) → list[CanonicalTable]
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Optional

import pandas as pd

from accounting.canonical_model import CanonicalCell, CanonicalRow, CanonicalTable

logger = logging.getLogger(__name__)


def enrich_tables(
    raw_tables: list[pd.DataFrame],
    dictionary=None,
    tier: int = 0,
    font_spans_per_table: Optional[list[list[dict]]] = None,
) -> list[CanonicalTable]:
    """Run the full 10-stage enrichment pipeline on raw DataFrames.

    Args:
        raw_tables: List of DataFrames extracted by a tier.
        dictionary: NomenclatureDictionary (loaded lazily if None).
        tier: Extraction tier (0, 1, 2) for provenance tagging.
        font_spans_per_table: Optional font metadata per table (for header detection).

    Returns:
        List of fully enriched CanonicalTable objects.
    """
    if not raw_tables:
        return []

    # Load dictionary lazily
    if dictionary is None:
        try:
            from accounting.nomenclature import load_default_dictionary
            dictionary = load_default_dictionary()
        except ImportError:
            logger.warning("NomenclatureDictionary not available — returning raw output")
            return _raw_fallback(raw_tables, tier)

    provenance = f"tier{tier}"
    canonical_tables: list[CanonicalTable] = []

    for table_idx, df in enumerate(raw_tables):
        if df.empty:
            continue

        font_spans = None
        if font_spans_per_table and table_idx < len(font_spans_per_table):
            font_spans = font_spans_per_table[table_idx]

        ct = _enrich_single_table(df, dictionary, provenance, font_spans)
        canonical_tables.append(ct)

    logger.info(f"Enrichment pipeline: produced {len(canonical_tables)} canonical tables")
    return canonical_tables


def _enrich_single_table(
    df: pd.DataFrame,
    dictionary,
    provenance: str,
    font_spans: Optional[list[dict]] = None,
) -> CanonicalTable:
    """Run stages 2–10 on a single table."""

    ct = CanonicalTable()

    # ---------------------------------------------------------------
    # Stage 2 — Section header detection
    # ---------------------------------------------------------------
    sections: list[Optional[str]] = []
    try:
        from accounting.header_detector import detect_section_headers
        sections = detect_section_headers(df, dictionary, font_spans)
    except Exception as exc:
        logger.debug(f"Stage 2 (headers) skipped: {exc}")
        sections = [None] * len(df)

    # ---------------------------------------------------------------
    # Stage 3 — Column identification
    # ---------------------------------------------------------------
    column_layout = None
    try:
        from accounting.column_identifier import identify_columns
        column_layout = identify_columns(df, dictionary)
        ct.column_model = column_layout.model
    except Exception as exc:
        logger.debug(f"Stage 3 (columns) skipped: {exc}")

    # ---------------------------------------------------------------
    # Stage 4 + 5 + 6 — FFFD → Numbers → Fuzzy Match (per row)
    # ---------------------------------------------------------------
    label_col_idx = 0
    if column_layout and hasattr(column_layout, "label_col_index"):
        label_col_idx = column_layout.label_col_index

    numeric_cols = []
    if column_layout and hasattr(column_layout, "numeric_col_indices"):
        numeric_cols = column_layout.numeric_col_indices

    match_results = []
    parsed_numbers: dict[tuple[int, int], Decimal] = {}

    for row_idx in range(len(df)):
        raw_label = str(df.iloc[row_idx, label_col_idx]).strip() if label_col_idx < len(df.columns) else ""
        section = sections[row_idx] if row_idx < len(sections) else None

        # Stage 4 — FFFD resolution
        resolved_label = raw_label
        try:
            resolved_label, _, _ = dictionary.resolve_fffd(raw_label)
        except Exception:
            pass

        # Stage 6 — Fuzzy matching (uses resolved label + section scope)
        mr = dictionary.fuzzy_match(resolved_label, section=section)
        match_results.append(mr)

        # Build the canonical row
        canonical_row = CanonicalRow(
            raw_text=raw_label,
            section=section,
        )

        if mr.entry:
            canonical_row.canonical_term = mr.entry.canonical_term
            canonical_row.match_type = mr.match_type
            canonical_row.match_confidence = mr.confidence
            canonical_row.account_code = mr.entry.account_code
            canonical_row.validation_field = mr.entry.validation_field
            canonical_row.is_subtotal = getattr(mr.entry, "is_subtotal", False)
        else:
            canonical_row.match_type = mr.match_type
            canonical_row.match_confidence = mr.confidence

        # Stage 5 — Parse numeric cells
        for col_idx in range(len(df.columns)):
            col_name = str(df.columns[col_idx])
            raw_value = str(df.iloc[row_idx, col_idx]) if pd.notna(df.iloc[row_idx, col_idx]) else ""

            cell = CanonicalCell(
                raw_value=raw_value,
                provenance=provenance,
            )

            if col_idx in numeric_cols:
                try:
                    from accounting.french_number import parse_french_number
                    parsed = parse_french_number(raw_value)
                    cell.parsed_value = parsed
                    if parsed is not None:
                        parsed_numbers[(row_idx, col_idx)] = parsed
                    elif raw_value.strip():
                        cell.flag = "yellow"  # parse failure
                except Exception:
                    cell.flag = "yellow"

            # Map column to role name if available
            role_name = _col_idx_to_role(col_idx, column_layout)
            canonical_row.cells[role_name or col_name] = cell

        ct.rows.append(canonical_row)

    # ---------------------------------------------------------------
    # Stage 7 — Grouped entry detection (unmatched rows only)
    # ---------------------------------------------------------------
    try:
        from accounting.grouped_entry import classify_unmatched_rows
        grouped_results = classify_unmatched_rows(df, match_results, dictionary, sections)
        for gr in grouped_results:
            if gr.row_index < len(ct.rows):
                row = ct.rows[gr.row_index]
                if gr.classification == "grouped":
                    row.match_type = "grouped"
                    row.match_confidence = gr.confidence
                    if gr.parent_canonical:
                        row.grouped_from = [gr.parent_canonical]
    except Exception as exc:
        logger.debug(f"Stage 7 (grouped) skipped: {exc}")

    # ---------------------------------------------------------------
    # Stage 8 — Subtotal detection + footing
    # ---------------------------------------------------------------
    try:
        from accounting.subtotal_detector import detect_subtotals
        subtotal_results = detect_subtotals(
            df, column_layout, dictionary, sections, parsed_numbers
        )
        for sr in subtotal_results:
            if sr.row_index < len(ct.rows):
                ct.rows[sr.row_index].is_subtotal = True
                if sr.footing_valid is False:
                    # Footing mismatch — flag the subtotal row
                    for cell in ct.rows[sr.row_index].cells.values():
                        if cell.parsed_value is not None:
                            cell.flag = "yellow"
    except Exception as exc:
        logger.debug(f"Stage 8 (subtotals) skipped: {exc}")

    # ---------------------------------------------------------------
    # Stage 9 — Statement classification
    # ---------------------------------------------------------------
    try:
        labels = [r.raw_text for r in ct.rows]
        ct.statement_type = dictionary.classify_statement(labels)
    except Exception as exc:
        logger.debug(f"Stage 9 (classification) skipped: {exc}")

    # ---------------------------------------------------------------
    # Stage 10 — Flag assignment (already done per-cell above; aggregate)
    # ---------------------------------------------------------------
    ct.recompute_aggregates()

    return ct


def _col_idx_to_role(col_idx: int, column_layout) -> Optional[str]:
    """Map a column index to its canonical role name."""
    if column_layout is None:
        return None
    role_map = getattr(column_layout, "role_map", {})
    for role, idx in role_map.items():
        if idx == col_idx:
            return role
    return None


def _raw_fallback(tables: list[pd.DataFrame], tier: int) -> list[CanonicalTable]:
    """Produce minimal CanonicalTables when no dictionary is available."""
    provenance = f"tier{tier}"
    result: list[CanonicalTable] = []
    for df in tables:
        ct = CanonicalTable()
        for row_idx in range(len(df)):
            row = CanonicalRow(
                raw_text=str(df.iloc[row_idx, 0]) if len(df.columns) > 0 else "",
                match_type="unrecognized",
            )
            for col_idx in range(len(df.columns)):
                raw = str(df.iloc[row_idx, col_idx]) if pd.notna(df.iloc[row_idx, col_idx]) else ""
                row.cells[str(df.columns[col_idx])] = CanonicalCell(
                    raw_value=raw, provenance=provenance
                )
            ct.rows.append(row)
        ct.recompute_aggregates()
        result.append(ct)
    return result
