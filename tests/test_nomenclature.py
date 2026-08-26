"""
Unit tests for Phase 1 Nomenclature foundation.

Covers:
- ``parse_french_number``: 3-layer parser (OCR fixes, locale norm, Babel)
- ``NomenclatureDictionary.resolve_fffd``: FFFD wildcard resolution (§9)
- ``NomenclatureDictionary.fuzzy_match``: 3-stage cascade (§13)
- ``NomenclatureDictionary.classify_statement``: term-distribution classifier (§24)
- ``NomenclatureDictionary.map_to_field``: validation_field bridge
- ``NomenclatureDictionary.get_prompt_vocab``: VLM prompt grounding
"""

from __future__ import annotations

import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from accounting.french_number import is_numeric_token, parse_french_number
from accounting.nomenclature import (
    NomenclatureDictionary,
    load_default_dictionary,
)


# ===========================================================================
# parse_french_number
# ===========================================================================


class TestFrenchNumber:
    def test_plain_integer(self):
        assert parse_french_number("1500") == Decimal("1500")

    def test_nbsp_thousands(self):
        assert parse_french_number("1\xa0500,00") == Decimal("1500.00")

    def test_regular_space_thousands(self):
        assert parse_french_number("12 450 000") == Decimal("12450000")

    def test_dot_thousands(self):
        assert parse_french_number("1.500,00") == Decimal("1500.00")

    def test_parenthesis_negative(self):
        assert parse_french_number("(1.500,00)") == Decimal("-1500.00")

    def test_dash_is_zero(self):
        assert parse_french_number("-") == Decimal("0")

    def test_em_dash_is_zero(self):
        assert parse_french_number("—") == Decimal("0")

    def test_currency_suffix_stripped(self):
        assert parse_french_number("150,00 DT") == Decimal("150.00")

    def test_currency_prefix_stripped(self):
        assert parse_french_number("DT 1.234,56") == Decimal("1234.56")

    def test_ocr_zero_letter(self):
        assert parse_french_number("l5O,00") == Decimal("150.00")

    def test_simple_decimal_comma(self):
        assert parse_french_number("1,5") == Decimal("1.5")

    def test_empty_string(self):
        assert parse_french_number("") is None

    def test_garbage(self):
        assert parse_french_number("abc") is None

    def test_none(self):
        assert parse_french_number(None) is None

    def test_negative_sign(self):
        assert parse_french_number("-1.500,00") == Decimal("-1500.00")

    def test_int_passthrough(self):
        assert parse_french_number(42) == Decimal("42")

    def test_float_passthrough(self):
        assert parse_french_number(1.5) == Decimal("1.5")

    def test_is_numeric_token_accepts_valid(self):
        assert is_numeric_token("1 500,00")
        assert is_numeric_token("(1.234)")
        assert is_numeric_token("-")

    def test_is_numeric_token_rejects_text(self):
        assert not is_numeric_token("Immobilisations")
        assert not is_numeric_token("")


# ===========================================================================
# NomenclatureDictionary — loading & indices
# ===========================================================================


@pytest.fixture(scope="module")
def nom() -> NomenclatureDictionary:
    load_default_dictionary.cache_clear()
    return load_default_dictionary()


class TestDictionaryLoading:
    def test_loads_entries(self, nom):
        assert len(nom.entries) >= 50

    def test_covers_all_three_statements(self, nom):
        stmts = set(e.statement_type for e in nom.entries)
        assert stmts == {"bilan", "compte_resultat", "flux_tresorerie"}

    def test_column_models_registered(self, nom):
        assert "brut_amort_net" in nom.column_models
        assert "n_n1" in nom.column_models

    def test_sections_registered(self, nom):
        assert "actifs_non_courants" in nom.sections
        assert "capitaux_propres" in nom.sections
        assert "flux_exploitation" in nom.sections

    def test_every_entry_has_normalized(self, nom):
        assert all(e.normalized for e in nom.entries)

    def test_every_entry_has_statement_type(self, nom):
        for e in nom.entries:
            assert e.statement_type in {"bilan", "compte_resultat", "flux_tresorerie"}


# ===========================================================================
# resolve_fffd (§9)
# ===========================================================================


class TestFFFDResolution:
    def test_single_wildcard_unambiguous(self, nom):
        resolved, conf, n = nom.resolve_fffd("R�sultat de l'exercice")
        assert resolved == "Résultat de l'exercice"
        assert conf >= 0.99
        assert n == 1

    def test_two_wildcards(self, nom):
        resolved, conf, _ = nom.resolve_fffd("Immobilisations financi�res")
        assert resolved == "Immobilisations financières"
        assert conf >= 0.80

    def test_multiple_wildcards_same_word(self, nom):
        resolved, conf, _ = nom.resolve_fffd("Charges financi�res")
        assert resolved == "Charges financières"
        assert conf >= 0.80

    def test_resolves_via_variation(self, nom):
        resolved, conf, _ = nom.resolve_fffd("Tr�sorerie de cl�ture")
        assert resolved == "Trésorerie en fin d'exercice"
        assert conf >= 0.80

    def test_question_mark_wildcard(self, nom):
        resolved, conf, _ = nom.resolve_fffd("R?sultat de l'exercice")
        assert resolved == "Résultat de l'exercice"
        assert conf >= 0.99

    def test_no_wildcards_returns_identity(self, nom):
        text = "Produits d'exploitation"
        resolved, conf, n = nom.resolve_fffd(text)
        assert resolved == text
        assert conf == 1.0
        assert n == 0

    def test_empty_input(self, nom):
        resolved, conf, n = nom.resolve_fffd("")
        assert resolved == ""
        assert n == 0


# ===========================================================================
# fuzzy_match (§13)
# ===========================================================================


class TestFuzzyMatch:
    def test_exact_match(self, nom):
        mr = nom.fuzzy_match("Capital social")
        assert mr.match_type == "exact"
        assert mr.confidence == 1.0
        assert mr.entry.validation_field == "capital_social"

    def test_accent_insensitive_exact(self, nom):
        mr = nom.fuzzy_match("capital social")
        assert mr.match_type == "exact"

    def test_variation_match(self, nom):
        mr = nom.fuzzy_match("chiffre d'affaires")
        assert mr.match_type in ("exact", "resolved")
        assert mr.entry is not None
        assert mr.entry.canonical_term == "Revenus"

    def test_fuzzy_abbreviation(self, nom):
        mr = nom.fuzzy_match("Immobilisat. incorpor.")
        assert mr.entry is not None
        assert mr.entry.canonical_term == "Immobilisations incorporelles"

    def test_section_scoping_narrows_candidates(self, nom):
        mr = nom.fuzzy_match("autres actifs non courants", section="actifs_non_courants")
        assert mr.entry is not None
        assert mr.entry.section == "actifs_non_courants"

    def test_unknown_term_is_custom(self, nom):
        mr = nom.fuzzy_match("Contribution sociale de solidarité")
        assert mr.match_type in ("custom", "unrecognized")
        assert mr.entry is None

    def test_empty_input(self, nom):
        mr = nom.fuzzy_match("")
        assert mr.match_type == "unrecognized"
        assert mr.entry is None


# ===========================================================================
# classify_statement (§24)
# ===========================================================================


class TestStatementClassification:
    def test_classifies_bilan(self, nom):
        rows = [
            "Immobilisations corporelles",
            "Stocks",
            "Clients et comptes rattachés",
            "Capital social",
        ]
        assert nom.classify_statement(rows) == "bilan"

    def test_classifies_compte_resultat(self, nom):
        rows = [
            "Revenus",
            "Charges de personnel",
            "Résultat financier",
            "Résultat d'exploitation",
        ]
        assert nom.classify_statement(rows) == "compte_resultat"

    def test_classifies_flux_tresorerie(self, nom):
        rows = [
            "Flux net d'exploitation",
            "Augmentation de capital",
            "Dividendes versés",
            "Trésorerie en fin d'exercice",
        ]
        assert nom.classify_statement(rows) == "flux_tresorerie"

    def test_empty_returns_unknown(self, nom):
        assert nom.classify_statement([]) == "unknown"

    def test_brut_amort_net_booster_favors_bilan(self, nom):
        rows = ["Immobilisations corporelles", "Brut", "Amortissement", "Net"]
        assert nom.classify_statement(rows) == "bilan"


# ===========================================================================
# map_to_field
# ===========================================================================


class TestMapToField:
    def test_canonical_term(self, nom):
        assert nom.map_to_field("Capital social") == "capital_social"

    def test_via_variation(self, nom):
        assert nom.map_to_field("chiffre d'affaires") == "chiffre_affaires"

    def test_accent_insensitive(self, nom):
        assert nom.map_to_field("resultat de l exercice") == "resultat_exercice"

    def test_unknown_returns_none(self, nom):
        assert nom.map_to_field("Totally Unknown Term") is None


# ===========================================================================
# get_prompt_vocab
# ===========================================================================


class TestPromptVocab:
    def test_produces_non_empty_vocab_per_statement(self, nom):
        for stmt in ("bilan", "compte_resultat", "flux_tresorerie"):
            vocab = nom.get_prompt_vocab(stmt)
            assert vocab.strip()
            assert all(line.startswith("- ") for line in vocab.splitlines() if line)

    def test_bilan_vocab_includes_core_items(self, nom):
        vocab = nom.get_prompt_vocab("bilan")
        assert "Capital social" in vocab
        assert "Stocks" in vocab
        assert "Total des actifs" in vocab

    def test_cr_vocab_includes_core_items(self, nom):
        vocab = nom.get_prompt_vocab("compte_resultat")
        assert "Revenus" in vocab
        assert "Charges de personnel" in vocab
        assert "Résultat d'exploitation" in vocab

    def test_tft_vocab_includes_core_items(self, nom):
        vocab = nom.get_prompt_vocab("flux_tresorerie")
        assert "Dividendes versés" in vocab
        assert "Variation nette de trésorerie" in vocab


# ===========================================================================
# classify_section
# ===========================================================================


class TestSectionClassifier:
    def test_actifs_non_courants(self, nom):
        assert nom.classify_section("ACTIFS NON COURANTS") == "actifs_non_courants"

    def test_capitaux_propres(self, nom):
        assert nom.classify_section("Capitaux propres") == "capitaux_propres"

    def test_passifs_courants(self, nom):
        assert nom.classify_section("PASSIFS COURANTS") == "passifs_courants"

    def test_unknown_header_returns_none(self, nom):
        assert nom.classify_section("This is a random sentence with no header keyword") is None
