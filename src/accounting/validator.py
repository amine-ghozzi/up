"""
Automatic Financial Statement Validation

Validates extracted financial data with severity-weighted scoring:
- Balance sheet balancing (Assets = Liabilities + Equity) — CRITICAL
- Subtotal/row-sum verification — ERROR
- Variation percentage consistency — WARNING
- Field type validation — WARNING

Severity levels (used for QCS semantic layer):
- CRITICAL (weight=3): Balance equation, fundamental arithmetic
- ERROR (weight=2): Subtotal mismatches, structural issues
- WARNING (weight=1): Variation %, field types, minor inconsistencies

See: Scoring-Approach-Analysis.md §4.4 (severity-weighted validation)
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# Severity weights for QCS semantic layer calculation
SEVERITY_WEIGHTS = {"CRITICAL": 3, "ERROR": 2, "WARNING": 1}


@dataclass
class ValidationResult:
    """Result of a single validation check."""
    check_name: str
    passed: bool
    message: str
    severity: str = "WARNING"  # CRITICAL, ERROR, or WARNING
    expected: Optional[float] = None
    actual: Optional[float] = None
    difference: Optional[float] = None


@dataclass
class ValidationReport:
    """Complete validation report for a document with severity-weighted scoring."""
    checks: List[ValidationResult] = field(default_factory=list)
    overall_passed: bool = True
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: int = 0
    critical_failures: int = 0
    severity_weighted_score: Optional[float] = None

    def add_check(self, check: ValidationResult):
        """Add a validation check result."""
        self.checks.append(check)
        self.total_checks += 1
        if check.passed:
            self.passed_checks += 1
        else:
            self.failed_checks += 1
            self.overall_passed = False
            if check.severity == "CRITICAL":
                self.critical_failures += 1

    def compute_severity_weighted_score(self):
        """
        Compute severity-weighted semantic score.

        Formula: sum(severity_weight * pass) / sum(severity_weight)
        where CRITICAL=3, ERROR=2, WARNING=1

        This gives CRITICAL checks 3x the impact of WARNING checks.
        """
        if self.total_checks == 0:
            self.severity_weighted_score = None
            return

        total_weight = 0
        passed_weight = 0
        for check in self.checks:
            w = SEVERITY_WEIGHTS.get(check.severity, 1)
            total_weight += w
            if check.passed:
                passed_weight += w

        self.severity_weighted_score = passed_weight / total_weight if total_weight > 0 else 0.5

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "overall_passed": self.overall_passed,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "critical_failures": self.critical_failures,
            "severity_weighted_score": (
                round(self.severity_weighted_score, 3)
                if self.severity_weighted_score is not None else None
            ),
            "checks": [
                {
                    "name": c.check_name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "message": c.message,
                    "expected": c.expected,
                    "actual": c.actual,
                    "difference": c.difference,
                }
                for c in self.checks
            ]
        }


# Keywords for identifying financial statement components
ASSET_KEYWORDS = [
    "actif", "assets", "immobilisation", "stock", "créance", "disponibilité",
    "trésorerie actif", "total actif", "actifs courants", "actifs non courants"
]

LIABILITY_KEYWORDS = [
    "passif", "liabilities", "dette", "fournisseur", "emprunt", "provision",
    "passifs courants", "passifs non courants"
]

EQUITY_KEYWORDS = [
    "capitaux propres", "equity", "capital social", "réserve", "résultat",
    "report à nouveau"
]

TOTAL_KEYWORDS = [
    "total", "sous-total", "subtotal", "somme"
]


def extract_numeric_value(value) -> Optional[float]:
    """
    Extract numeric value from a cell.
    
    Handles:
    - Direct numbers
    - Numbers with spaces (French format: 1 234 567)
    - Numbers with parentheses for negatives
    - Empty cells
    
    Rejects:
    - Concatenated multi-number cells from garbled extraction
    """
    if pd.isna(value):
        return None
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        val_str = value.strip()
        if not val_str:
            return None

        # Reject concatenated numbers: multiple comma-formatted groups
        # e.g. "95,276,113 41,694,461" = two numbers jammed together
        import re
        multi_number_re = re.compile(
            r'\d{1,3}(?:,\d{3}){1,}\s+\d{1,3}(?:,\d{3}){1,}'
        )
        if multi_number_re.search(val_str):
            return None  # garbled — multiple values in one cell

        # Remove whitespace used as thousands separator
        cleaned = val_str.replace(" ", "").replace("\u00a0", "")
        
        # Handle parentheses for negatives
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        
        # Remove currency symbols and other non-numeric chars
        cleaned = re.sub(r'[^\d\-.,]', '', cleaned)
        
        # Handle French decimal format (comma as decimal separator)
        if "," in cleaned and "." not in cleaned:
            cleaned = cleaned.replace(",", ".")
        elif "," in cleaned and "." in cleaned:
            # Mixed format, assume comma is thousands separator
            cleaned = cleaned.replace(",", "")
        
        try:
            return float(cleaned) if cleaned else None
        except ValueError:
            return None
    
    return None


def find_numeric_columns(df: pd.DataFrame) -> List[str]:
    """Find columns that contain primarily numeric data."""
    numeric_cols = []
    
    for col in df.columns:
        numeric_count = 0
        total_count = 0
        
        for val in df[col]:
            if not pd.isna(val) and str(val).strip():
                total_count += 1
                if extract_numeric_value(val) is not None:
                    numeric_count += 1
        
        # Consider column numeric if >50% of values are numbers
        if total_count > 0 and numeric_count / total_count > 0.5:
            numeric_cols.append(col)
    
    return numeric_cols


def identify_row_type(label: str) -> str:
    """
    Identify the type of a row based on its label.
    
    Returns: 'asset', 'liability', 'equity', 'total', or 'other'
    """
    if not label:
        return 'other'
    
    label_lower = label.lower()
    
    # Check for total rows first
    for kw in TOTAL_KEYWORDS:
        if kw in label_lower:
            has_equity = any(eq_kw in label_lower for eq_kw in EQUITY_KEYWORDS)
            has_liability = any(liab_kw in label_lower for liab_kw in LIABILITY_KEYWORDS)
            has_asset = any(asset_kw in label_lower for asset_kw in ASSET_KEYWORDS)

            # "Total Capitaux Propres et Passifs" = grand total of right side
            if has_equity and has_liability:
                return 'total_equity_and_liabilities'
            if has_asset:
                return 'total_asset'
            if has_liability:
                return 'total_liability'
            if has_equity:
                return 'total_equity'
            return 'total'
    
    # Check category keywords
    for kw in ASSET_KEYWORDS:
        if kw in label_lower:
            return 'asset'
    
    for kw in LIABILITY_KEYWORDS:
        if kw in label_lower:
            return 'liability'
    
    for kw in EQUITY_KEYWORDS:
        if kw in label_lower:
            return 'equity'
    
    return 'other'


def validate_balance_sheet(df: pd.DataFrame, tolerance: float = 0.01) -> ValidationResult:
    """
    Validate that a balance sheet balances.
    
    Assets = Liabilities + Equity
    """
    # Find the label column (usually first column)
    if len(df.columns) < 2:
        return ValidationResult(
            check_name="Balance Sheet Balance",
            passed=False,
            message="Not enough columns to validate"
        )
    
    label_col = df.columns[0]
    numeric_cols = find_numeric_columns(df)
    
    if not numeric_cols:
        return ValidationResult(
            check_name="Balance Sheet Balance",
            passed=False,
            message="No numeric columns found"
        )
    
    # Prefer "Net" column for Brut/Amort/Net bilan presentation
    value_col = numeric_cols[0]
    for col in numeric_cols:
        if str(col).strip().lower() == "net":
            value_col = col
            break
    
    total_assets = 0.0
    total_liabilities = 0.0
    total_equity = 0.0
    total_equity_and_liabilities = None  # "Total Capitaux Propres et Passifs"

    for idx, row in df.iterrows():
        label = str(row[label_col]) if pd.notna(row[label_col]) else ""
        value = extract_numeric_value(row[value_col])

        if value is None:
            continue

        row_type = identify_row_type(label)

        if row_type == 'total_asset':
            total_assets = value
        elif row_type == 'total_liability':
            total_liabilities = value
        elif row_type == 'total_equity':
            total_equity = value
        elif row_type == 'total_equity_and_liabilities':
            total_equity_and_liabilities = value

    # Check if we found the totals
    if total_assets == 0:
        return ValidationResult(
            check_name="Balance Sheet Balance",
            passed=False,
            severity="CRITICAL",
            message="Could not identify Total Assets row"
        )

    # Prefer direct comparison if "Total Capitaux Propres et Passifs" exists
    if total_equity_and_liabilities is not None:
        expected = total_equity_and_liabilities
    else:
        expected = total_liabilities + total_equity

    difference = abs(total_assets - expected)

    # Allow for rounding tolerance (percentage based)
    threshold = max(abs(total_assets) * tolerance, 1.0)
    passed = difference <= threshold

    return ValidationResult(
        check_name="Balance Sheet Balance",
        passed=passed,
        severity="CRITICAL",  # Balance equation is fundamental
        message=f"Assets = {total_assets:,.2f}, Liabilities + Equity = {expected:,.2f}"
                + (" ✓" if passed else f" (Difference: {difference:,.2f})"),
        expected=expected,
        actual=total_assets,
        difference=difference
    )


def validate_row_sums(df: pd.DataFrame, tolerance: float = 0.01) -> List[ValidationResult]:
    """
    Validate that subtotals match the sum of their component rows.
    
    This is a heuristic validation that looks for 'total' rows and
    checks if nearby numeric values sum correctly.
    """
    results = []
    
    if len(df.columns) < 2:
        return results
    
    label_col = df.columns[0]
    numeric_cols = find_numeric_columns(df)
    
    if not numeric_cols:
        return results
    
    value_col = numeric_cols[0]
    
    # Find all total rows and their values
    for idx, row in df.iterrows():
        label = str(row[label_col]) if pd.notna(row[label_col]) else ""
        
        if any(kw in label.lower() for kw in TOTAL_KEYWORDS):
            total_value = extract_numeric_value(row[value_col])
            
            if total_value is not None and total_value != 0:
                # Sum rows above this total until we hit another total or header
                component_sum = 0.0
                component_count = 0
                
                for prev_idx in range(int(idx) - 1, max(0, int(idx) - 20), -1):
                    if prev_idx not in df.index:
                        continue
                    
                    prev_row = df.loc[prev_idx]
                    prev_label = str(prev_row[label_col]) if pd.notna(prev_row[label_col]) else ""
                    
                    # Stop if we hit another total
                    if any(kw in prev_label.lower() for kw in TOTAL_KEYWORDS):
                        break
                    
                    prev_value = extract_numeric_value(prev_row[value_col])
                    if prev_value is not None:
                        component_sum += prev_value
                        component_count += 1
                
                if component_count >= 2:  # Need at least 2 components
                    difference = abs(total_value - component_sum)
                    threshold = max(abs(total_value) * tolerance, 1.0)
                    passed = difference <= threshold

                    results.append(ValidationResult(
                        check_name=f"Row Sum: {label[:30]}...",
                        passed=passed,
                        severity="ERROR",  # Subtotal mismatches are structural
                        message=f"Expected {total_value:,.2f}, Sum of {component_count} rows = {component_sum:,.2f}"
                                + (" ✓" if passed else f" (Diff: {difference:,.2f})"),
                        expected=total_value,
                        actual=component_sum,
                        difference=difference
                    ))
    
    return results


def validate_variation_percentages(df: pd.DataFrame, tolerance: float = 0.5) -> List[ValidationResult]:
    """
    Validate that variation percentage columns match the actual calculated variance.
    
    Common column headers: "Variation %", "Var %", "Δ%", "Evolution %"
    Formula: (Current - Prior) / Prior * 100
    
    Args:
        df: DataFrame with period columns and variation column
        tolerance: Allowed percentage point difference (e.g., 0.5 = ±0.5%)
        
    Returns:
        List of validation results for rows with variance discrepancies
    """
    results = []
    
    if len(df.columns) < 3:  # Need at least: label, value1, value2 (or variation)
        return results
    
    # Find variation column
    variation_col = None
    variation_keywords = ["variation", "var %", "δ%", "evolution", "écart"]
    
    for col in df.columns:
        col_lower = str(col).lower()
        if any(kw in col_lower for kw in variation_keywords) or "%" in str(col):
            variation_col = col
            break
    
    if variation_col is None:
        return results
    
    # Find period columns (e.g., 2023, 2022, N, N-1)
    period_cols = []
    for col in df.columns:
        col_str = str(col)
        # Match year patterns or N/N-1 patterns
        if re.match(r'^\d{4}$', col_str) or re.match(r'^N(-\d)?$', col_str):
            period_cols.append(col)
    
    if len(period_cols) < 2:
        return results
    
    # Assume first period column is current, second is prior
    current_col = period_cols[0]
    prior_col = period_cols[1]
    label_col = df.columns[0]
    
    for idx, row in df.iterrows():
        label = str(row[label_col]) if pd.notna(row[label_col]) else f"Row {idx}"
        
        current_val = extract_numeric_value(row[current_col])
        prior_val = extract_numeric_value(row[prior_col])
        reported_var = extract_numeric_value(row[variation_col])
        
        # Skip if any value is missing or prior is zero
        if current_val is None or prior_val is None or reported_var is None:
            continue
        
        if prior_val == 0:
            continue  # Can't calculate percentage from zero base
        
        # Calculate expected variation
        expected_var = ((current_val - prior_val) / abs(prior_val)) * 100
        difference = abs(reported_var - expected_var)
        
        # Only report if there's a significant discrepancy
        if difference > tolerance:
            passed = False
            results.append(ValidationResult(
                check_name=f"Var%: {label[:25]}",
                passed=passed,
                severity="WARNING",  # Variation % is informational
                message=f"Reported {reported_var:.1f}%, Expected {expected_var:.1f}%",
                expected=expected_var,
                actual=reported_var,
                difference=difference
            ))
    
    # If all checked rows passed, add a summary success
    if not results:
        # Check if we validated any rows
        validated_count = 0
        for idx, row in df.iterrows():
            current_val = extract_numeric_value(row[current_col])
            prior_val = extract_numeric_value(row[prior_col])
            reported_var = extract_numeric_value(row[variation_col])
            if all(v is not None for v in [current_val, prior_val, reported_var]) and prior_val != 0:
                validated_count += 1
        
        if validated_count > 0:
            results.append(ValidationResult(
                check_name="Variation % Check",
                passed=True,
                severity="WARNING",
                message=f"All {validated_count} variation percentages verified ✓"
            ))
    
    return results


def auto_correct_variation_percentages(df: pd.DataFrame, tolerance: float = 5.0) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Auto-correct variation percentage columns by recalculating from numeric values.
    
    When OCR mangles values like "-6,9%" → "%5'9", this function:
    1. Identifies the % variation column (often follows a "Variation" column)
    2. Finds prior/current period columns
    3. Recalculates the % from the numeric values
    4. Replaces obviously wrong values
    
    Args:
        df: DataFrame with period columns and variation column
        tolerance: Max allowed difference before triggering correction (percentage points)
        
    Returns:
        Tuple of (corrected DataFrame, list of corrections made)
    """
    corrections = []
    df_corrected = df.copy()
    
    if len(df.columns) < 3:
        logger.debug(f"auto_correct: Skipping - only {len(df.columns)} columns")
        return df_corrected, corrections
    
    # Find the percentage variation column - try multiple strategies
    variation_pct_col = None
    columns_list = list(df.columns)
    
    # Strategy 1: Look for "%" column that follows a "Variation" column
    for i, col in enumerate(columns_list):
        col_str = str(col).strip().lower()
        if col_str in ["%", "var %", "var%"]:
            variation_pct_col = col
            logger.debug(f"auto_correct: Found % column by exact match: {col}")
            break
        # Check if this column is "Variation" and next one is "%"
        if "variation" in col_str and i + 1 < len(columns_list):
            next_col = str(columns_list[i + 1]).strip()
            if next_col == "%" or "%" in next_col:
                variation_pct_col = columns_list[i + 1]
                logger.debug(f"auto_correct: Found % column after Variation: {variation_pct_col}")
                break
    
    # Strategy 2: Look for column containing variation keywords + %
    if variation_pct_col is None:
        for col in columns_list:
            col_lower = str(col).lower()
            if "%" in col_lower and any(kw in col_lower for kw in ["var", "évol", "ecart"]):
                variation_pct_col = col
                logger.debug(f"auto_correct: Found variation % column by keywords: {col}")
                break
    
    # Strategy 3: Check last column for % symbols in values
    if variation_pct_col is None and len(df.columns) >= 4:
        last_col = columns_list[-1]
        sample_values = df[last_col].dropna().astype(str).head(5)
        if any("%" in str(v) for v in sample_values):
            variation_pct_col = last_col
            logger.debug(f"auto_correct: Found % column by value inspection (last col): {last_col}")
    
    if variation_pct_col is None:
        logger.debug(f"auto_correct: No % column found. Columns: {columns_list}")
        return df_corrected, corrections
    
    # Find numeric period columns (large values = financial amounts)
    period_cols = []
    for col in columns_list:
        if col == variation_pct_col:
            continue
        col_lower = str(col).lower()
        # Skip columns that look like variation/difference columns
        if "variation" in col_lower or "ecart" in col_lower or "diff" in col_lower:
            continue
        # Check if column has large numeric values - sample multiple rows
        non_null = df[col].dropna()
        if len(non_null) == 0:
            continue
        # Check up to 10 values to find a large one
        found_large = False
        for i, val in enumerate(non_null):
            if i >= 10:
                break
            parsed = extract_numeric_value(val)
            if parsed is not None and abs(parsed) > 100:
                found_large = True
                logger.debug(f"auto_correct: Found period column: {col} (sample={parsed})")
                break
        if found_large:
            period_cols.append(col)
    
    if len(period_cols) < 2:
        logger.debug(f"auto_correct: Only {len(period_cols)} period columns found, need 2")
        return df_corrected, corrections
    
    # Use first two large-value columns as current and prior period
    current_col = period_cols[0]
    prior_col = period_cols[1]
    label_col = columns_list[0]
    
    logger.info(f"auto_correct: Processing - label={label_col}, current={current_col}, prior={prior_col}, variation_pct={variation_pct_col}")
    
    # Pre-compute column dot consistency analysis ONCE (cache for performance)
    # If most values don't have dots, but some do, those dots are OCR artifacts
    column_has_dot_artifacts = {}
    for col in [current_col, prior_col]:
        col_values = df[col].dropna().astype(str).tolist()
        dot_count = sum(1 for v in col_values if '.' in v and ',' not in v)
        no_dot_count = sum(1 for v in col_values if '.' not in v and len(re.sub(r'[^\d]', '', v)) > 3)
        # If majority of large values don't have dots, treat any dots as OCR artifacts
        column_has_dot_artifacts[col] = no_dot_count > dot_count
        if column_has_dot_artifacts[col]:
            logger.debug(f"auto_correct: Column {col} has dot artifacts (no_dot={no_dot_count} > dot={dot_count})")
    
    def normalize_column_value(val_str: str, col_name: str) -> Optional[float]:
        """Parse a numeric value, using cached column analysis for dot handling."""
        if pd.isna(val_str):
            return None
        val_str = str(val_str).strip()
        
        # If this column has dot artifacts and this value has a dot (no comma), remove it
        if column_has_dot_artifacts.get(col_name, False) and '.' in val_str and ',' not in val_str:
            cleaned = val_str.replace(" ", "").replace("\xa0", "")
            cleaned = re.sub(r'[^\d\-.]', '', cleaned)
            cleaned = cleaned.replace(".", "")  # Remove OCR artifact dot
            try:
                return float(cleaned) if cleaned else None
            except ValueError:
                return None
        
        return extract_numeric_value(val_str)

    
    for idx, row in df.iterrows():
        label = str(row[label_col]) if pd.notna(row[label_col]) else f"Row {idx}"
        
        # Use column-aware normalization for period values (handles OCR dot artifacts)
        current_val = normalize_column_value(row[current_col], current_col)
        prior_val = normalize_column_value(row[prior_col], prior_col)
        reported_raw = row[variation_pct_col]
        reported_var = extract_numeric_value(reported_raw)
        
        # Skip if missing values or zero base
        if current_val is None or prior_val is None:
            continue
        if prior_val == 0:
            continue
        
        # Calculate expected variation (standard accounting formula, preserves sign)
        expected_var = ((current_val - prior_val) / prior_val) * 100
        
        # Check if reported value is clearly wrong
        needs_correction = False
        
        if reported_var is None:
            # Couldn't even parse a number - definitely needs correction
            needs_correction = True
            logger.debug(f"auto_correct: Row {idx} '{label}': couldn't parse '{reported_raw}'")
        elif abs(reported_var - expected_var) > tolerance:
            # Large discrepancy - likely OCR error
            needs_correction = True
            logger.debug(f"auto_correct: Row {idx} '{label}': mismatch - reported={reported_var}, expected={expected_var:.1f}")
        
        if needs_correction:
            # Format the corrected value (French format with comma)
            if expected_var >= 0:
                corrected_str = f"+{expected_var:.1f}%".replace(".", ",")
            else:
                corrected_str = f"{expected_var:.1f}%".replace(".", ",")
            
            corrections.append({
                "row": idx,
                "label": label[:30],
                "original": str(reported_raw),
                "corrected": corrected_str,
                "expected": round(expected_var, 1),
                "parsed_as": reported_var
            })
            
            df_corrected.at[idx, variation_pct_col] = corrected_str
    
    logger.info(f"auto_correct: Made {len(corrections)} corrections")
    return df_corrected, corrections


def validate_field_types(df: pd.DataFrame) -> List[ValidationResult]:
    """
    Validate that each column contains the expected data type.
    
    - First column: labels (strings expected)
    - Other columns: numeric values expected
    
    Returns:
        List of validation results for type issues
    """
    results = []
    
    if len(df.columns) < 2:
        return results
    
    label_col = df.columns[0]
    numeric_cols = df.columns[1:]
    
    # Track issues per column
    empty_count = 0
    invalid_count = 0
    invalid_cells = []
    
    for col in numeric_cols:
        for idx, val in df[col].items():
            label = str(df.loc[idx, label_col]) if label_col in df.columns else f"Row {idx}"
            
            # Skip empty cells (they might be intentional)
            if pd.isna(val) or str(val).strip() == "":
                empty_count += 1
                continue
            
            # Check if value should be numeric but isn't
            val_str = str(val).strip()
            
            # Skip if it's clearly a header or section title
            if any(kw in val_str.lower() for kw in ["total", "actif", "passif", "note"]):
                continue
            
            # Try to extract numeric value
            numeric = extract_numeric_value(val)
            
            if numeric is None and val_str:
                # Value looks like it should be numeric but isn't
                # Check if it contains digits (suggesting it's a malformed number)
                if any(c.isdigit() for c in val_str):
                    invalid_count += 1
                    if len(invalid_cells) < 3:  # Limit examples
                        invalid_cells.append(f"'{val_str}' at {label[:20]}")
    
    # Report findings
    if invalid_count > 0:
        examples = ", ".join(invalid_cells)
        results.append(ValidationResult(
            check_name="Field Type Check",
            passed=False,
            severity="WARNING",
            message=f"{invalid_count} invalid numeric values (e.g., {examples})"
        ))
    else:
        results.append(ValidationResult(
            check_name="Field Type Check",
            passed=True,
            severity="WARNING",
            message="All numeric fields contain valid values ✓"
        ))
    
    return results


def validate_extraction(tables: List[pd.DataFrame]) -> ValidationReport:
    """
    Run all validation checks on extracted tables.
    
    Args:
        tables: List of DataFrames from extraction
        
    Returns:
        ValidationReport with all check results
    """
    report = ValidationReport()
    
    if not tables:
        report.add_check(ValidationResult(
            check_name="Tables Found",
            passed=False,
            severity="CRITICAL",
            message="No tables extracted from document"
        ))
        report.compute_severity_weighted_score()
        return report

    # Add basic check
    report.add_check(ValidationResult(
        check_name="Tables Found",
        passed=True,
        severity="CRITICAL",
        message=f"Found {len(tables)} table(s)"
    ))
    
    for i, df in enumerate(tables):
        table_prefix = f"Table {i+1}" if len(tables) > 1 else ""
        
        # 1. Balance sheet validation
        balance_result = validate_balance_sheet(df)
        if balance_result.expected is not None:
            balance_result.check_name = f"{table_prefix}: {balance_result.check_name}".strip(": ")
            report.add_check(balance_result)
        
        # 2. Row sum validation
        row_results = validate_row_sums(df)
        for result in row_results[:3]:  # Limit to first 3 to avoid noise
            result.check_name = f"{table_prefix}: {result.check_name}".strip(": ")
            report.add_check(result)
        
        # 3. Variation % validation (NEW)
        var_results = validate_variation_percentages(df)
        for result in var_results[:3]:  # Limit to first 3 issues
            result.check_name = f"{table_prefix}: {result.check_name}".strip(": ")
            report.add_check(result)
        
        # 4. Field type validation (NEW)
        type_results = validate_field_types(df)
        for result in type_results:
            result.check_name = f"{table_prefix}: {result.check_name}".strip(": ")
            report.add_check(result)

    # Compute severity-weighted score for QCS semantic layer
    report.compute_severity_weighted_score()

    return report


if __name__ == "__main__":
    # Test with sample data including variation column
    sample_data = {
        "Label": [
            "ACTIFS NON COURANTS",
            "Immobilisations incorporelles",
            "Immobilisations corporelles",
            "TOTAL ACTIFS NON COURANTS",
            "ACTIFS COURANTS",
            "Stocks",
            "Créances",
            "TOTAL ACTIFS COURANTS",
            "TOTAL ACTIF",
        ],
        "2023": [
            "",
            120000,
            220000,
            340000,
            "",
            55000,
            165000,
            220000,
            560000,
        ],
        "2022": [
            "",
            100000,
            200000,
            300000,
            "",
            50000,
            150000,
            200000,
            500000,
        ],
        "Variation %": [
            "",
            20.0,    # Correct: (120000-100000)/100000 = 20%
            10.0,    # Correct: (220000-200000)/200000 = 10%
            13.3,    # Correct: (340000-300000)/300000 = 13.33%
            "",
            10.0,    # Correct: (55000-50000)/50000 = 10%
            10.0,    # Correct: (165000-150000)/150000 = 10%
            10.0,    # Correct
            12.0,    # Correct: (560000-500000)/500000 = 12%
        ]
    }
    
    df = pd.DataFrame(sample_data)
    report = validate_extraction([df])
    
    print("\nValidation Report:")
    print(f"  Overall: {'PASSED' if report.overall_passed else 'FAILED'}")
    print(f"  Checks: {report.passed_checks}/{report.total_checks} passed")
    
    for check in report.checks:
        status = "✓" if check.passed else "✗"
        print(f"  {status} {check.check_name}: {check.message}")

