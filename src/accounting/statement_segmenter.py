"""
Financial Statement Segmenter

Classifies extracted tables as Bilan / Compte de Résultat / Flux de Trésorerie
using the :class:`NomenclatureDictionary` (§24) as the single source of truth —
no more hardcoded keyword lists duplicated across files.

Public API is unchanged (``StatementSegment``, ``classify_table``,
``segment_tables``, ``group_by_statement``) so the pipeline and HITL code do
not need to be touched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

from accounting.nomenclature import (
    NomenclatureDictionary,
    load_default_dictionary,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bilan side disambiguation — still needs a small keyword set because the
# Nomenclature entries themselves don't carry an "actif-only / passif-only"
# flag at the line-item level (sub_type == actif | passif). We derive these
# from the dictionary at import time.
# ---------------------------------------------------------------------------


def _actif_passif_keywords(
    dictionary: NomenclatureDictionary,
) -> tuple[list[str], list[str]]:
    """Extract normalized label keyword sets from dictionary entries.

    Anything in the Bilan with ``sub_type == "actif"`` contributes to the
    Actif keyword set; likewise for Passif. Section labels are added as
    anchor terms too.
    """
    actif: list[str] = []
    passif: list[str] = []
    for entry in dictionary.entries:
        if entry.statement_type != "bilan":
            continue
        # Pick up both canonical terms and variations — gives resilience against
        # label reorderings and OCR variants.
        forms = [entry.normalized] + entry.normalized_variations()
        if entry.sub_type == "actif":
            actif.extend(forms)
        elif entry.sub_type == "passif":
            passif.extend(forms)
    # Section labels
    for section_key, cfg in dictionary.sections.items():
        if cfg.get("statement_type") != "bilan":
            continue
        label = str(cfg.get("label_canonical", "")).lower()
        if not label:
            continue
        if cfg.get("sub_type") == "actif":
            actif.append(label)
        elif cfg.get("sub_type") == "passif":
            passif.append(label)
    # Deduplicate, preserve order
    seen_a: set[str] = set()
    seen_p: set[str] = set()
    uniq_actif = [t for t in actif if t and not (t in seen_a or seen_a.add(t))]
    uniq_passif = [t for t in passif if t and not (t in seen_p or seen_p.add(t))]
    return uniq_actif, uniq_passif


@dataclass
class StatementSegment:
    """A classified table segment."""

    table_index: int
    statement_type: str            # bilan | compte_resultat | flux_tresorerie | unknown
    confidence: float              # 0.0 – 1.0
    keyword_hits: int
    df: pd.DataFrame = field(repr=False)
    sub_type: Optional[str] = None  # actif | passif | None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_table_labels(df: pd.DataFrame) -> list[str]:
    """Gather label-ish text from a DataFrame (columns + first 1-2 cols)."""
    parts: list[str] = []

    for col in df.columns:
        parts.append(str(col))

    if len(df.columns) > 0:
        label_col = df.columns[0]
        for val in df[label_col]:
            if pd.notna(val):
                parts.append(str(val))

    # Scan second column too if it looks textual (some Bilan formats put labels there)
    if len(df.columns) > 1:
        second_col = df.columns[1]
        for val in df[second_col].head(5):
            if pd.notna(val) and isinstance(val, str) and not any(c.isdigit() for c in str(val)):
                parts.append(str(val))

    return [p for p in parts if p and p.strip()]


def _score_bilan_side(
    labels_joined_norm: str,
    actif_keywords: list[str],
    passif_keywords: list[str],
) -> Optional[str]:
    actif_hits = sum(1 for kw in actif_keywords if kw and kw in labels_joined_norm)
    passif_hits = sum(1 for kw in passif_keywords if kw and kw in labels_joined_norm)

    if actif_hits >= 3 and passif_hits >= 3:
        return None  # combined bilan
    if actif_hits >= 2 and passif_hits <= 1:
        return "actif"
    if passif_hits >= 2 and actif_hits <= 1:
        return "passif"
    return None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_table(
    df: pd.DataFrame,
    table_index: int = 0,
    dictionary: Optional[NomenclatureDictionary] = None,
) -> StatementSegment:
    """Classify a single DataFrame as bilan / compte_resultat / flux_tresorerie.

    Delegates the heavy lifting to ``NomenclatureDictionary.classify_statement``,
    which aggregates term-distribution scores plus structural boosters (§24).
    """
    if dictionary is None:
        dictionary = load_default_dictionary()

    labels = _collect_table_labels(df)
    if not labels:
        return StatementSegment(
            table_index=table_index,
            statement_type="unknown",
            confidence=0.0,
            keyword_hits=0,
            df=df,
        )

    statement_type = dictionary.classify_statement(labels)
    keyword_hits = sum(
        1
        for lbl in labels
        if (mr := dictionary.fuzzy_match(lbl)).entry is not None
        and mr.match_type in ("exact", "fuzzy", "resolved")
    )

    if statement_type == "unknown":
        return StatementSegment(
            table_index=table_index,
            statement_type="unknown",
            confidence=0.0,
            keyword_hits=keyword_hits,
            df=df,
        )

    # Confidence: hits over row count, capped at 1.0
    confidence = min(1.0, keyword_hits / max(len(labels), 1) + 0.2)

    sub_type: Optional[str] = None
    if statement_type == "bilan":
        # Normalized join for substring checking
        from accounting.nomenclature import _normalize  # re-export of module helper
        labels_norm = _normalize(" ".join(labels))
        actif_kw, passif_kw = _actif_passif_keywords(dictionary)
        sub_type = _score_bilan_side(labels_norm, actif_kw, passif_kw)

    return StatementSegment(
        table_index=table_index,
        statement_type=statement_type,
        confidence=confidence,
        keyword_hits=keyword_hits,
        df=df,
        sub_type=sub_type,
    )


def segment_tables(
    tables: List[pd.DataFrame],
    dictionary: Optional[NomenclatureDictionary] = None,
) -> List[StatementSegment]:
    """Classify every table. Shares one dictionary load across the batch."""
    if dictionary is None:
        dictionary = load_default_dictionary()

    segments: List[StatementSegment] = []
    for i, df in enumerate(tables):
        segment = classify_table(df, table_index=i, dictionary=dictionary)
        segments.append(segment)
        sub_info = f", sub_type={segment.sub_type}" if segment.sub_type else ""
        logger.info(
            f"Table {i}: classified as '{segment.statement_type}' "
            f"(confidence={segment.confidence:.2f}, hits={segment.keyword_hits}{sub_info})"
        )
    return segments


def group_by_statement(segments: List[StatementSegment]) -> dict:
    """Group segments by statement type, with ``bilan_actif`` / ``bilan_passif``
    buckets derived from ``sub_type`` for targeted downstream lookups.
    """
    groups: dict[str, list[StatementSegment]] = {
        "bilan": [],
        "bilan_actif": [],
        "bilan_passif": [],
        "compte_resultat": [],
        "flux_tresorerie": [],
        "unknown": [],
    }
    for seg in segments:
        groups.setdefault(seg.statement_type, []).append(seg)
        if seg.statement_type == "bilan" and seg.sub_type == "actif":
            groups["bilan_actif"].append(seg)
        elif seg.statement_type == "bilan" and seg.sub_type == "passif":
            groups["bilan_passif"].append(seg)
    return groups
