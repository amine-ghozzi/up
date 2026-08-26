"""
Variation Capture & Bootstrapping — §28 of the nomenclature revamp plan.

Every pipeline run logs fuzzy and low-confidence matches to a JSONL staging
file (``data/variation_candidates.jsonl``).  Human HITL corrections are logged
with ``status="approved"`` for immediate use; after 3+ consistent sightings a
candidate is auto-promotable to the YAML dictionary.

Public surface:

    log_variation_candidate(…)    — called by fuzzy_match / HITL hooks
    log_hitl_correction(…)        — called when a human corrects a label
    log_confirmed_custom(…)       — called when a human marks a row "genuinely custom"
    get_approved_variations()     — returns approved variations for hot-reload
    promote_variations(…)         — periodic batch: candidate → promoted into YAML
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default log location — can be overridden via constructor or env var
_DEFAULT_LOG_PATH = Path(__file__).resolve().parents[2] / "data" / "variation_candidates.jsonl"


class VariationLogger:
    """Manages the variation candidate lifecycle: candidate → approved → promoted."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = Path(log_path) if log_path else _DEFAULT_LOG_PATH
        # Ensure parent directory exists
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def log_variation_candidate(
        self,
        raw_text: str,
        canonical_term: Optional[str],
        match_type: str,
        confidence: float,
        section: Optional[str] = None,
        document_id: str = "",
    ) -> None:
        """Log a potential new variation for future review.

        Called automatically when ``fuzzy_match()`` returns ``confidence < 0.95``
        and ``match_type`` is ``fuzzy`` or ``resolved``.
        """
        if match_type not in ("fuzzy", "resolved") or confidence >= 0.95:
            return  # Only log uncertain matches

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_text": raw_text,
            "matched_to": canonical_term,
            "confidence": round(confidence, 4),
            "match_type": match_type,
            "section": section,
            "document_id": document_id,
            "status": "candidate",
        }
        self._append(entry)

    def log_hitl_correction(
        self,
        raw_text: str,
        canonical_term: str,
        document_id: str = "",
    ) -> None:
        """Log a human-validated correction — immediately usable.

        Called when a HITL reviewer selects the correct canonical term for
        a label.  Status is ``approved`` — no candidate-stage wait.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_text": raw_text,
            "matched_to": canonical_term,
            "confidence": 1.0,
            "match_type": "hitl_correction",
            "section": None,
            "document_id": document_id,
            "status": "approved",
        }
        self._append(entry)

    def log_confirmed_custom(
        self,
        raw_text: str,
        document_id: str = "",
    ) -> None:
        """Mark a label as genuinely company-specific.

        Prevents future pipeline runs from re-suggesting this text as a
        fuzzy match candidate.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "raw_text": raw_text,
            "matched_to": None,
            "confidence": 0.0,
            "match_type": "confirmed_custom",
            "section": None,
            "document_id": document_id,
            "status": "confirmed_custom",
        }
        self._append(entry)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_approved_variations(self) -> dict[str, str]:
        """Return all approved (human-validated) variations as {raw_text: canonical}.

        Used for hot-reload: the fuzzy matcher can check these before
        running RapidFuzz, providing O(1) lookups for known corrections.
        """
        approved: dict[str, str] = {}
        for record in self._read_all():
            if record.get("status") == "approved" and record.get("matched_to"):
                approved[record["raw_text"]] = record["matched_to"]
        return approved

    def get_confirmed_customs(self) -> set[str]:
        """Return all confirmed-custom labels (should never be fuzzy-matched)."""
        customs: set[str] = set()
        for record in self._read_all():
            if record.get("status") == "confirmed_custom":
                customs.add(record["raw_text"])
        return customs

    def get_promotable_candidates(self, min_count: int = 3) -> list[dict]:
        """Find candidates seen ≥ ``min_count`` times with consistent mapping.

        Returns a list of ``{raw_text, canonical_term, count}`` dicts
        ready for promotion to ``nomenclature_data.yaml``.
        """
        # Group candidates by (raw_text, matched_to)
        pair_counts: Counter[tuple[str, str]] = Counter()
        for record in self._read_all():
            if record.get("status") == "candidate" and record.get("matched_to"):
                key = (record["raw_text"], record["matched_to"])
                pair_counts[key] += 1

        # Check for consistency: same raw_text always maps to same canonical
        raw_to_canonicals: dict[str, set[str]] = {}
        for (raw, canon), count in pair_counts.items():
            raw_to_canonicals.setdefault(raw, set()).add(canon)

        promotable = []
        for (raw, canon), count in pair_counts.items():
            if count >= min_count:
                is_consistent = len(raw_to_canonicals.get(raw, set())) == 1
                promotable.append({
                    "raw_text": raw,
                    "canonical_term": canon,
                    "count": count,
                    "consistent": is_consistent,
                })
        return promotable

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _append(self, entry: dict) -> None:
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            logger.warning("Failed to write variation log entry to %s", self.log_path)

    def _read_all(self) -> list[dict]:
        if not self.log_path.exists():
            return []
        records: list[dict] = []
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except OSError:
            logger.warning("Failed to read variation log from %s", self.log_path)
        return records
