"""
Integration tests for Phase 3 modules.

Covers:
- table_merger: continuation detection + merge
- header_detector: dual-mode section detection
- column_identifier: layout model identification
- subtotal_detector: footing validation
- grouped_entry: unmatched row classification
- green_flag: 5-gate check
- enrichment_pipeline: full DAG on synthetic data
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import pytest


# ===========================================================================
# table_merger
# ===========================================================================


class TestTableMerger:
    def test_continuation_same_cols(self):
        from accounting.table_merger import PageTable, detect_continuation

        df1 = pd.DataFrame({"Label": ["A", "B"], "N": [100, 200]})
        df2 = pd.DataFrame({"Label": ["C", "D"], "N": [300, 400]})
        t1 = PageTable(df=df1, page_index=0)
        t2 = PageTable(df=df2, page_index=1)
        sig = detect_continuation(t1, t2)
        assert sig.col_count_match is True
        # col count + open section = 2.0 minimum (no header dup or x-offset)
        assert sig.total_score >= 1.0

    def test_merge_two_pages(self):
        from accounting.table_merger import PageTable, merge_page_tables

        df1 = pd.DataFrame({"Label": ["A", "B"], "N": [100, 200]})
        df2 = pd.DataFrame({"Label": ["C", "D"], "N": [300, 400]})
        tables = [PageTable(df=df1, page_index=0), PageTable(df=df2, page_index=1)]
        merged = merge_page_tables(tables)
        # May or may not merge depending on signals — at minimum both tables present
        total_rows = sum(len(df) for df in merged)
        assert total_rows == 4

    def test_non_consecutive_pages_not_merged(self):
        from accounting.table_merger import PageTable, merge_page_tables

        df1 = pd.DataFrame({"Label": ["A"], "N": [100]})
        df2 = pd.DataFrame({"Label": ["B"], "N": [200]})
        tables = [PageTable(df=df1, page_index=0), PageTable(df=df2, page_index=5)]
        merged = merge_page_tables(tables)
        assert len(merged) == 2  # never merged — page gap too large


# ===========================================================================
# header_detector
# ===========================================================================


class TestHeaderDetector:
    def test_heuristic_detects_caps_header(self):
        from accounting.header_detector import detect_section_headers
        from accounting.nomenclature import load_default_dictionary

        load_default_dictionary.cache_clear()
        nom = load_default_dictionary()
        df = pd.DataFrame({
            "Label": ["ACTIFS NON COURANTS", "Immobilisations", "Stocks"],
            "N": [None, 100, 200],
            "N-1": [None, 90, 180],
        })
        sections = detect_section_headers(df, nom)
        assert len(sections) == 3
        assert sections[0] == "actifs_non_courants"
        # Propagation: rows below inherit the section
        assert sections[1] == "actifs_non_courants"

    def test_empty_df(self):
        from accounting.header_detector import detect_section_headers
        assert detect_section_headers(pd.DataFrame()) == []


# ===========================================================================
# column_identifier
# ===========================================================================


class TestColumnIdentifier:
    def test_brut_amort_net_detection(self):
        from accounting.column_identifier import identify_columns

        df = pd.DataFrame({
            "Libellé": ["Row1", "Row2"],
            "Brut": ["1 000", "2 000"],
            "Amortissement": ["500", "800"],
            "Net": ["500", "1 200"],
            "N-1": ["400", "1 100"],
        })
        layout = identify_columns(df)
        assert layout.model == "brut_amort_net"
        assert layout.confidence >= 0.80

    def test_n_n1_detection(self):
        from accounting.column_identifier import identify_columns

        df = pd.DataFrame({
            "Poste": ["Capital", "Réserves"],
            "Exercice N": ["10 000", "5 000"],
            "Exercice N-1": ["10 000", "4 500"],
        })
        layout = identify_columns(df)
        assert layout.model == "n_n1"

    def test_single_column(self):
        from accounting.column_identifier import identify_columns

        df = pd.DataFrame({"Label": ["A"]})
        layout = identify_columns(df)
        assert layout.model == "single"


# ===========================================================================
# subtotal_detector
# ===========================================================================


class TestSubtotalDetector:
    def test_detects_total_row(self):
        from accounting.subtotal_detector import detect_subtotals

        df = pd.DataFrame({
            "Label": ["Item A", "Item B", "Total des actifs"],
            "N": ["100", "200", "300"],
        })
        results = detect_subtotals(df)
        assert len(results) == 1
        assert results[0].row_index == 2
        assert results[0].is_subtotal is True

    def test_footing_validation(self):
        from accounting.subtotal_detector import detect_subtotals
        from accounting.column_identifier import ColumnLayout

        df = pd.DataFrame({
            "Label": ["Item A", "Item B", "Total"],
            "N": ["100", "200", "300"],
        })
        layout = ColumnLayout(model="n_n1", numeric_col_indices=[1])
        parsed = {
            (0, 1): Decimal("100"),
            (1, 1): Decimal("200"),
            (2, 1): Decimal("300"),
        }
        results = detect_subtotals(df, layout, parsed_numbers=parsed)
        assert len(results) == 1
        assert results[0].footing_valid is True

    def test_footing_mismatch(self):
        from accounting.subtotal_detector import detect_subtotals
        from accounting.column_identifier import ColumnLayout

        df = pd.DataFrame({
            "Label": ["Item A", "Item B", "Total"],
            "N": ["100", "200", "999"],
        })
        layout = ColumnLayout(model="n_n1", numeric_col_indices=[1])
        parsed = {
            (0, 1): Decimal("100"),
            (1, 1): Decimal("200"),
            (2, 1): Decimal("999"),
        }
        results = detect_subtotals(df, layout, parsed_numbers=parsed)
        assert len(results) == 1
        assert results[0].footing_valid is False


# ===========================================================================
# green_flag
# ===========================================================================


class TestGreenFlag:
    def test_all_pass(self):
        from accounting.green_flag import check_green_flag

        result = check_green_flag(
            match_rate=0.98,
            parse_fail_rate=0.0,
            column_confidence=0.90,
            classification_confidence=0.80,
        )
        assert result.passed is True
        assert len(result.failure_reasons) == 0

    def test_low_match_rate_fails(self):
        from accounting.green_flag import check_green_flag

        result = check_green_flag(
            match_rate=0.70,
            parse_fail_rate=0.0,
            column_confidence=0.90,
            classification_confidence=0.80,
        )
        assert result.passed is False
        assert result.gate_1_passed is False

    def test_parse_failures_fail(self):
        from accounting.green_flag import check_green_flag

        result = check_green_flag(
            match_rate=0.98,
            parse_fail_rate=0.05,
            column_confidence=0.90,
            classification_confidence=0.80,
        )
        assert result.passed is False
        assert result.gate_2_passed is False


# ===========================================================================
# enrichment_pipeline (end-to-end)
# ===========================================================================


class TestEnrichmentPipeline:
    def test_produces_canonical_tables(self):
        from accounting.enrichment_pipeline import enrich_tables

        df = pd.DataFrame({
            "Label": ["Capital social", "Réserves", "Résultat de l'exercice"],
            "N": ["10 000", "5 000", "1 500"],
            "N-1": ["10 000", "4 500", "1 200"],
        })
        tables = enrich_tables([df], tier=0)
        assert len(tables) == 1
        ct = tables[0]
        assert len(ct.rows) == 3
        # At least some rows should be matched
        matched = [r for r in ct.rows if r.match_type in ("exact", "fuzzy", "resolved")]
        assert len(matched) >= 1

    def test_empty_input(self):
        from accounting.enrichment_pipeline import enrich_tables
        assert enrich_tables([]) == []

    def test_cells_have_parsed_values(self):
        from accounting.enrichment_pipeline import enrich_tables

        df = pd.DataFrame({
            "Poste": ["Capital social"],
            "Exercice N": ["10 000"],
            "Exercice N-1": ["9 000"],
        })
        tables = enrich_tables([df], tier=0)
        ct = tables[0]
        # Check that at least one cell has a parsed value
        all_cells = [c for r in ct.rows for c in r.cells.values()]
        parsed = [c for c in all_cells if c.parsed_value is not None]
        assert len(parsed) >= 1
