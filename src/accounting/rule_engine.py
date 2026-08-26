"""
Configurable Validation Rule Engine

Loads validation rules from JSON, executes them against segmented financial
statements, and produces a unified ValidationReport compatible with QCS.

Supports:
- V1: Intra-statement arithmetic (balance equation, VNC, CR equation, subtotals)
- V2: Cross-statement coherence (Bilan <-> CR <-> TFT)
- V3: Anomaly detection (negatives, nulls, magnitude mismatches)
- V4: Ratio computation (informational, non-blocking)

Rules are defined in rules/validation_rules.json and can be toggled per-standard.
"""

import json
import logging
import math
import re
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import pandas as pd
import numpy as np

from accounting.validator import (
    ValidationResult,
    ValidationReport,
    extract_numeric_value,
    find_numeric_columns,
    identify_row_type,
    validate_balance_sheet,
    validate_row_sums,
    validate_variation_percentages,
    validate_field_types,
)
from accounting.statement_segmenter import (
    StatementSegment,
    segment_tables,
    group_by_statement,
)

logger = logging.getLogger(__name__)


# ===========================================================================
# Rule Loader
# ===========================================================================

RULES_DIR = Path(__file__).parent / "rules"


def load_rules(standard: str = "NCT") -> List[Dict]:
    """
    Load active validation rules for a given accounting standard.

    Filters rules by standard and active flag.
    """
    rules_path = RULES_DIR / "validation_rules.json"
    if not rules_path.exists():
        logger.warning(f"Rules file not found: {rules_path}")
        return []

    with open(rules_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_rules = data.get("rules", [])

    # Filter by standard and active flag
    filtered = [
        r for r in all_rules
        if r.get("active", True) and standard in r.get("standards", [])
    ]

    logger.info(f"Loaded {len(filtered)}/{len(all_rules)} rules for standard={standard}")
    return filtered


def get_rules_by_tier(rules: List[Dict], tier: str) -> List[Dict]:
    """Get rules for a specific tier (V1, V2, V3, V4)."""
    return [r for r in rules if r.get("tier") == tier]


# ===========================================================================
# NCT Account Reference Loader
# ===========================================================================

def load_nct_accounts() -> Dict:
    """Load NCT account structure reference."""
    nct_path = RULES_DIR / "nct_accounts.json"
    if not nct_path.exists():
        return {}
    with open(nct_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ===========================================================================
# Value Extraction Helpers
# ===========================================================================

def _strip_accents(text: str) -> str:
    """Strip common French accents for fuzzy keyword matching."""
    for src, dst in [
        ("é", "e"), ("è", "e"), ("ê", "e"), ("ë", "e"),
        ("à", "a"), ("â", "a"),
        ("ô", "o"), ("û", "u"), ("ù", "u"),
        ("î", "i"), ("ï", "i"), ("ç", "c"),
    ]:
        text = text.replace(src, dst)
    return text


def _find_value_by_keywords(
    df: pd.DataFrame,
    keywords: List[str],
    value_col: str,
    label_col: str,
) -> Optional[float]:
    """
    Find a numeric value in a DataFrame by matching row labels against keywords.

    Iterates keywords first (priority order), then rows — so more specific
    keywords listed earlier take precedence over generic ones that might
    match a subtotal higher up in the table.
    """
    # Pre-normalize all row labels once
    labels = []
    for idx, row in df.iterrows():
        raw = str(row[label_col]) if pd.notna(row[label_col]) else ""
        norm = _strip_accents(re.sub(r'\s+', ' ', raw.lower().strip()))
        labels.append((idx, norm))

    for kw in keywords:
        kw_clean = _strip_accents(kw)
        for idx, label_clean in labels:
            if kw_clean in label_clean:
                val = extract_numeric_value(df.at[idx, value_col])
                if val is not None:
                    return val
    return None


def _get_label_col_and_value_col(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Get the label column (first) and best numeric column.

    Prefers a column named 'Net' (for Brut/Amort/Net bilan presentation),
    then falls back to the last numeric column (most likely current-year values).
    """
    if len(df.columns) < 2:
        return None, None
    label_col = df.columns[0]
    numeric_cols = find_numeric_columns(df)
    if not numeric_cols:
        return label_col, None

    # Prefer "Net" column (NCT/SYSCOHADA Brut/Amort/Net presentation)
    for col in numeric_cols:
        if str(col).strip().lower() == "net":
            return label_col, col

    # Prefer last numeric column (often the most complete / current year)
    # But for bilan without Brut/Amort/Net, first numeric is fine
    return label_col, numeric_cols[0]


# ===========================================================================
# V1: Intra-Statement Arithmetic
# ===========================================================================

def validate_vnc_consistency(
    df: pd.DataFrame,
    tolerance: float = 0.01,
    tolerance_min: float = 1.0,
) -> List[ValidationResult]:
    """
    V1-002: Validate VNC (Valeur Nette Comptable) consistency.

    For NCT/SYSCOHADA bilan with Brut/Amort/Net presentation:
    Net = Brut - Amortissements (- Provisions)

    Looks for column headers containing "brut", "amort", "net".
    """
    results = []

    # Find brut/amort/net columns
    brut_col = None
    amort_col = None
    net_col = None

    for col in df.columns:
        col_lower = str(col).lower().strip()
        if "brut" in col_lower:
            brut_col = col
        elif "amort" in col_lower or "deprec" in col_lower:
            amort_col = col
        elif col_lower == "net" or col_lower.endswith(" net"):
            net_col = col

    if not all([brut_col, amort_col, net_col]):
        # No Brut/Amort/Net presentation found — skip
        return results

    label_col = df.columns[0]
    checked = 0
    errors = 0

    for idx, row in df.iterrows():
        label = str(row[label_col]) if pd.notna(row[label_col]) else ""

        brut_val = extract_numeric_value(row[brut_col])
        amort_val = extract_numeric_value(row[amort_col])
        net_val = extract_numeric_value(row[net_col])

        # Skip rows where not all three values are present
        if brut_val is None or net_val is None:
            continue
        if amort_val is None:
            amort_val = 0.0

        checked += 1
        expected_net = brut_val - abs(amort_val)  # Amort is often shown as positive
        difference = abs(net_val - expected_net)
        threshold = max(abs(brut_val) * tolerance, tolerance_min)
        passed = difference <= threshold

        if not passed:
            errors += 1
            if errors <= 3:  # Limit reported errors
                results.append(ValidationResult(
                    check_name=f"VNC: {label[:25]}",
                    passed=False,
                    severity="ERROR",
                    message=f"Net={net_val:,.0f}, expected Brut({brut_val:,.0f}) - Amort({amort_val:,.0f}) = {expected_net:,.0f}",
                    expected=expected_net,
                    actual=net_val,
                    difference=difference,
                ))

    if checked > 0 and errors == 0:
        results.append(ValidationResult(
            check_name="VNC Consistency",
            passed=True,
            severity="ERROR",
            message=f"All {checked} VNC rows verified (Net = Brut - Amort)",
        ))

    return results


def validate_cr_equation(
    df: pd.DataFrame,
    tolerance: float = 0.01,
    tolerance_min: float = 1.0,
) -> List[ValidationResult]:
    """
    V1-003: Validate CR (Compte de Resultat) equation.

    Resultat Net = Total Produits - Total Charges
    or Resultat Net = Resultat d'Exploitation + Resultat Financier - Impots

    Tries multiple strategies to find the components.
    """
    results = []
    label_col, value_col = _get_label_col_and_value_col(df)
    if not label_col or not value_col:
        return results

    resultat_net = _find_value_by_keywords(
        df,
        ["resultat net", "resultat de l'exercice", "resultat net de l'exercice",
         "benefice net", "perte nette"],
        value_col, label_col,
    )

    # Strategy 1: Resultat d'Exploitation + Resultat Financier - Impots
    # This is more reliable than Total Produits - Total Charges because
    # most CRs have exploitation subtotals, not overall totals.
    res_exploit = _find_value_by_keywords(
        df, ["resultat d'exploitation"], value_col, label_col
    )
    res_financier = _find_value_by_keywords(
        df, ["resultat financier", "resultat des activites financieres"], value_col, label_col
    )
    impots = _find_value_by_keywords(
        df, ["impots sur les benefices", "impot sur les societes", "charge fiscale"], value_col, label_col
    )

    if resultat_net is not None and res_exploit is not None:
        res_fin = res_financier if res_financier is not None else 0.0
        tax = impots if impots is not None else 0.0
        expected = res_exploit + res_fin - abs(tax)
        difference = abs(resultat_net - expected)
        threshold = max(abs(res_exploit) * tolerance, tolerance_min)
        passed = difference <= threshold

        results.append(ValidationResult(
            check_name="CR Equation (Exploit+Fin-Impots)",
            passed=passed,
            severity="CRITICAL",
            message=(
                f"Resultat Net={resultat_net:,.0f}, "
                f"Exploit({res_exploit:,.0f}) + Fin({res_fin:,.0f}) - Impots({tax:,.0f}) = {expected:,.0f}"
                + ("" if passed else f" (Diff: {difference:,.0f})")
            ),
            expected=expected,
            actual=resultat_net,
            difference=difference,
        ))

    return results


# ===========================================================================
# V2: Cross-Statement Coherence
# ===========================================================================

def validate_cross_statement(
    grouped: Dict[str, List[StatementSegment]],
    tolerance: float = 0.01,
    tolerance_min: float = 1.0,
) -> List[ValidationResult]:
    """
    V2: Cross-statement validation.

    Checks:
    - V2-001: Bilan Resultat de l'exercice == CR Resultat Net
    - V2-002: Bilan Liquidites == TFT Tresorerie de cloture
    """
    results = []

    bilan_tables = grouped.get("bilan", [])
    cr_tables = grouped.get("compte_resultat", [])
    tft_tables = grouped.get("flux_tresorerie", [])

    # V2-001: Resultat Net match (Bilan <-> CR)
    # Prefer bilan_passif sub-table (resultat is on the equity/passif side)
    bilan_passif = grouped.get("bilan_passif", [])
    if bilan_tables and cr_tables:
        # Use passif sub-table if available, otherwise full bilan
        bilan_for_resultat = bilan_passif[0].df if bilan_passif else bilan_tables[0].df
        cr_df = cr_tables[0].df

        b_label, b_val_col = _get_label_col_and_value_col(bilan_for_resultat)
        c_label, c_val_col = _get_label_col_and_value_col(cr_df)

        if b_label and b_val_col and c_label and c_val_col:
            bilan_resultat = _find_value_by_keywords(
                bilan_for_resultat,
                ["resultat de l'exercice", "resultat net"],
                b_val_col, b_label,
            )
            cr_resultat = _find_value_by_keywords(
                cr_df,
                ["resultat net", "resultat de l'exercice", "benefice net", "perte nette"],
                c_val_col, c_label,
            )

            if bilan_resultat is not None and cr_resultat is not None:
                difference = abs(bilan_resultat - cr_resultat)
                threshold = max(abs(bilan_resultat) * tolerance, tolerance_min)
                passed = difference <= threshold

                results.append(ValidationResult(
                    check_name="V2: Resultat Net Match (Bilan-CR)",
                    passed=passed,
                    severity="CRITICAL",
                    message=(
                        f"Bilan Resultat={bilan_resultat:,.0f}, CR Resultat={cr_resultat:,.0f}"
                        + ("" if passed else f" (Diff: {difference:,.0f})")
                    ),
                    expected=cr_resultat,
                    actual=bilan_resultat,
                    difference=difference,
                ))
            else:
                results.append(ValidationResult(
                    check_name="V2: Resultat Net Match (Bilan-CR)",
                    passed=False,
                    severity="CRITICAL",
                    message=(
                        f"Could not find Resultat in "
                        f"{'Bilan' if bilan_resultat is None else 'CR'}"
                    ),
                ))

    # V2-002: Tresorerie match (Bilan <-> TFT)
    # Prefer bilan_actif sub-table (liquidités are on the actif side)
    bilan_actif = grouped.get("bilan_actif", [])
    if bilan_tables and tft_tables:
        bilan_for_treso = bilan_actif[0].df if bilan_actif else bilan_tables[0].df
        tft_df = tft_tables[0].df

        b_label, b_val_col = _get_label_col_and_value_col(bilan_for_treso)
        t_label, t_val_col = _get_label_col_and_value_col(tft_df)

        if b_label and b_val_col and t_label and t_val_col:
            bilan_treso = _find_value_by_keywords(
                bilan_for_treso,
                ["liquidites et equivalents", "tresorerie", "disponibilites",
                 "liquidites"],
                b_val_col, b_label,
            )
            tft_cloture = _find_value_by_keywords(
                tft_df,
                ["tresorerie de cloture", "tresorerie fin", "solde de cloture"],
                t_val_col, t_label,
            )

            if bilan_treso is not None and tft_cloture is not None:
                difference = abs(bilan_treso - tft_cloture)
                threshold = max(abs(bilan_treso) * tolerance, tolerance_min)
                passed = difference <= threshold

                results.append(ValidationResult(
                    check_name="V2: Tresorerie Match (Bilan-TFT)",
                    passed=passed,
                    severity="CRITICAL",
                    message=(
                        f"Bilan Liquidites={bilan_treso:,.0f}, TFT Cloture={tft_cloture:,.0f}"
                        + ("" if passed else f" (Diff: {difference:,.0f})")
                    ),
                    expected=tft_cloture,
                    actual=bilan_treso,
                    difference=difference,
                ))

    return results


# ===========================================================================
# V3: Anomaly Detection
# ===========================================================================

def detect_anomalies(
    segments: List[StatementSegment],
) -> List[ValidationResult]:
    """
    V3: Anomaly detection for financial statements.

    Checks:
    - V3-001: Negative values where unexpected (capital social, immo brutes, stocks)
    - V3-002: Null operational values (CA = 0)
    - V3-003: Magnitude mismatch (assets in millions, liabilities in thousands)
    """
    results = []

    for seg in segments:
        df = seg.df
        label_col, value_col = _get_label_col_and_value_col(df)
        if not label_col or not value_col:
            continue

        # V3-001: Negative where unexpected
        if seg.statement_type == "bilan":
            negative_fields = {
                "capital social": "Capital social",
                "immobilisations": "Immobilisations brutes",
                "stocks": "Stocks",
            }
            for kw, field_name in negative_fields.items():
                val = _find_value_by_keywords(df, [kw], value_col, label_col)
                if val is not None and val < 0:
                    results.append(ValidationResult(
                        check_name=f"V3: Negative {field_name}",
                        passed=False,
                        severity="WARNING",
                        message=f"{field_name} = {val:,.0f} (negative — accounting impossibility)",
                        actual=val,
                    ))

        # V3-002: Null operational values
        if seg.statement_type == "compte_resultat":
            ca = _find_value_by_keywords(
                df,
                ["chiffre d'affaires", "revenus", "ventes"],
                value_col, label_col,
            )
            if ca is not None and ca == 0:
                results.append(ValidationResult(
                    check_name="V3: Null CA",
                    passed=False,
                    severity="WARNING",
                    message="Chiffre d'affaires = 0 (likely extraction failure for operating company)",
                    actual=0.0,
                ))

        # V3-003: Magnitude mismatch within statement
        magnitudes = []
        for idx, row in df.iterrows():
            val = extract_numeric_value(row[value_col])
            if val is not None and val != 0:
                magnitudes.append(math.floor(math.log10(abs(val))) if abs(val) >= 1 else 0)

        if len(magnitudes) >= 3:
            mag_range = max(magnitudes) - min(magnitudes)
            if mag_range > 6:  # More than 6 orders of magnitude
                results.append(ValidationResult(
                    check_name=f"V3: Magnitude Mismatch (Table {seg.table_index})",
                    passed=False,
                    severity="WARNING",
                    message=(
                        f"Values span {mag_range} orders of magnitude "
                        f"(10^{min(magnitudes)} to 10^{max(magnitudes)}) — possible unit/scale OCR error"
                    ),
                ))

    return results


# ===========================================================================
# V4: Ratio Computation (informational)
# ===========================================================================

def compute_ratios(
    grouped: Dict[str, List[StatementSegment]],
) -> List[ValidationResult]:
    """
    V4: Compute financial ratios (informational, non-blocking).

    - Fonds de Roulement (FR)
    - Tresorerie Nette cross-check
    """
    results = []

    bilan_tables = grouped.get("bilan", [])
    bilan_actif = grouped.get("bilan_actif", [])
    bilan_passif = grouped.get("bilan_passif", [])
    if not bilan_tables:
        results.append(ValidationResult(
            check_name="V4: Fonds de Roulement (FR)",
            passed=True,
            severity="WARNING",
            message="Skipped — no bilan table found",
        ))
        return results

    # Use sub-tables if available for targeted lookups
    actif_df = bilan_actif[0].df if bilan_actif else bilan_tables[0].df
    passif_df = bilan_passif[0].df if bilan_passif else bilan_tables[0].df

    a_label, a_val = _get_label_col_and_value_col(actif_df)
    p_label, p_val = _get_label_col_and_value_col(passif_df)
    if not a_label or not a_val or not p_label or not p_val:
        results.append(ValidationResult(
            check_name="V4: Fonds de Roulement (FR)",
            passed=True,
            severity="WARNING",
            message=(
                f"Skipped — could not detect columns "
                f"(actif: label={a_label}, val={a_val}; passif: label={p_label}, val={p_val})"
            ),
        ))
        return results

    # Try to find key values (actif side for NC assets, passif side for CP/passif NC)
    # Keywords ordered longest-first so more specific phrases match before generic ones
    total_actif_nc = _find_value_by_keywords(
        actif_df,
        [
            "total des actifs non courants",
            "total actifs non courants",
            "total actifs immobilises",
            "total des actifs immobilises",
        ],
        a_val, a_label,
    )

    # For total CP, prefer the most complete total (avant affectation > avant résultat)
    total_cp = _find_value_by_keywords(
        passif_df,
        [
            "total des capitaux propres avant affectation",
            "total capitaux propres avant affectation",
            "total des capitaux propres et passifs",
            "total des capitaux propres",
            "total capitaux propres",
        ],
        p_val, p_label,
    )

    total_passif_nc = _find_value_by_keywords(
        passif_df,
        [
            "total des passifs non courants",
            "total passifs non courants",
        ],
        p_val, p_label,
    )

    # Diagnostic: always produce a check so the user sees what happened
    missing = []
    if total_actif_nc is None:
        missing.append("Total Actifs NC")
    if total_cp is None:
        missing.append("Total Capitaux Propres")
    if total_passif_nc is None:
        missing.append("Total Passifs NC")

    if missing:
        results.append(ValidationResult(
            check_name="V4: Fonds de Roulement (FR)",
            passed=True,
            severity="WARNING",
            message=(
                f"Skipped — could not locate: {', '.join(missing)}. "
                f"Found: Actif NC={total_actif_nc}, CP={total_cp}, Passif NC={total_passif_nc}. "
                f"Columns used: actif val='{a_val}', passif val='{p_val}'"
            ),
        ))
        return results

    # V4-001: Fonds de Roulement
    capitaux_permanents = total_cp + total_passif_nc
    fr = capitaux_permanents - total_actif_nc

    results.append(ValidationResult(
        check_name="V4: Fonds de Roulement (FR)",
        passed=fr >= 0,  # Expected positive
        severity="WARNING",
        message=(
            f"FR = Capitaux permanents({capitaux_permanents:,.0f}) "
            f"- Actif NC({total_actif_nc:,.0f}) = {fr:,.0f}"
            + (" (positive)" if fr >= 0 else " (NEGATIVE — structural risk)")
        ),
        actual=fr,
    ))

    return results


# ===========================================================================
# Main Entry Point: Full Validation Pipeline
# ===========================================================================

def validate_with_rules(
    tables: List[pd.DataFrame],
    standard: str = "NCT",
    run_v2: bool = True,
    run_v3: bool = True,
    run_v4: bool = True,
) -> ValidationReport:
    """
    Run the full configurable validation pipeline on extracted tables.

    Flow:
    1. Segment tables by statement type
    2. Load rules for the given standard
    3. Run V1 intra-statement checks (existing + new VNC, CR equation)
    4. Run V2 cross-statement checks (if multiple statement types found)
    5. Run V3 anomaly detection
    6. Run V4 ratio computation (informational)
    7. Compile unified ValidationReport

    Args:
        tables: List of DataFrames from extraction
        standard: Accounting standard (IFRS, NCT, SYSCOHADA)
        run_v2: Enable cross-statement validation
        run_v3: Enable anomaly detection
        run_v4: Enable ratio computation

    Returns:
        ValidationReport with all check results and severity-weighted score
    """
    report = ValidationReport()

    if not tables:
        report.add_check(ValidationResult(
            check_name="Tables Found",
            passed=False,
            severity="CRITICAL",
            message="No tables extracted from document",
        ))
        report.compute_severity_weighted_score()
        return report

    # Basic check
    report.add_check(ValidationResult(
        check_name="Tables Found",
        passed=True,
        severity="CRITICAL",
        message=f"Found {len(tables)} table(s)",
    ))

    # Load rules (for logging / future dynamic rule execution)
    rules = load_rules(standard)
    v1_rules = get_rules_by_tier(rules, "V1")
    v2_rules = get_rules_by_tier(rules, "V2")
    logger.info(f"Active rules: V1={len(v1_rules)}, V2={len(v2_rules)}")

    # -----------------------------------------------------------------------
    # Step 1: Segment tables
    # -----------------------------------------------------------------------
    segments = segment_tables(tables)
    grouped = group_by_statement(segments)

    segment_summary = {k: len(v) for k, v in grouped.items() if v}
    report.add_check(ValidationResult(
        check_name="Statement Segmentation",
        passed=bool(segment_summary),
        severity="WARNING",
        message=f"Segments: {segment_summary}",
    ))

    # -----------------------------------------------------------------------
    # Step 2: V1 — Intra-statement validation
    # -----------------------------------------------------------------------
    for seg in segments:
        df = seg.df
        prefix = f"[{seg.statement_type}] " if seg.statement_type != "unknown" else ""

        # V1-001: Balance sheet equation (existing)
        if seg.statement_type in ("bilan", "unknown"):
            balance_result = validate_balance_sheet(df)
            if balance_result.expected is not None:
                balance_result.check_name = f"{prefix}{balance_result.check_name}"
                report.add_check(balance_result)

        # V1-002: VNC consistency (new)
        if seg.statement_type in ("bilan", "unknown"):
            vnc_results = validate_vnc_consistency(df)
            for r in vnc_results:
                r.check_name = f"{prefix}{r.check_name}"
                report.add_check(r)

        # V1-003: CR equation (new)
        if seg.statement_type in ("compte_resultat", "unknown"):
            cr_results = validate_cr_equation(df)
            for r in cr_results:
                r.check_name = f"{prefix}{r.check_name}"
                report.add_check(r)

        # V1-004: Subtotal coherence (existing)
        row_results = validate_row_sums(df)
        for r in row_results[:3]:
            r.check_name = f"{prefix}{r.check_name}"
            report.add_check(r)

        # V1-005: Variation % (existing)
        var_results = validate_variation_percentages(df)
        for r in var_results[:3]:
            r.check_name = f"{prefix}{r.check_name}"
            report.add_check(r)

        # V1-006: Field type integrity (existing)
        type_results = validate_field_types(df)
        for r in type_results:
            r.check_name = f"{prefix}{r.check_name}"
            report.add_check(r)

    # -----------------------------------------------------------------------
    # Step 3: V2 — Cross-statement coherence
    # -----------------------------------------------------------------------
    if run_v2:
        cross_results = validate_cross_statement(grouped)
        for r in cross_results:
            report.add_check(r)

    # -----------------------------------------------------------------------
    # Step 4: V3 — Anomaly detection
    # -----------------------------------------------------------------------
    if run_v3:
        anomaly_results = detect_anomalies(segments)
        for r in anomaly_results:
            report.add_check(r)

    # -----------------------------------------------------------------------
    # Step 5: V4 — Ratio computation
    # -----------------------------------------------------------------------
    if run_v4:
        ratio_results = compute_ratios(grouped)
        for r in ratio_results:
            report.add_check(r)

    # -----------------------------------------------------------------------
    # Compute severity-weighted score for QCS
    # -----------------------------------------------------------------------
    report.compute_severity_weighted_score()

    logger.info(
        f"Validation complete: {report.passed_checks}/{report.total_checks} passed, "
        f"critical_failures={report.critical_failures}, "
        f"severity_weighted={report.severity_weighted_score}"
    )

    return report
