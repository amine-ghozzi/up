"""
Nomenclature Dictionary — the pipeline's single source of truth for
French/NCT accounting terms.

Loads ``rules/nomenclature_data.yaml`` and exposes the public surface
described in §3 of the nomenclature revamp plan:

    resolve_fffd(text)              — replace U+FFFD (and ``?``) via wildcard search
    classify_statement(rows)        — infer bilan / compte_resultat / flux_tresorerie
    map_to_field(canonical_term)    — bridge to validation_rules.json
    fuzzy_match(text, section=None) — 3-stage matcher (exact → variation → rapidfuzz)
    get_terms_for(statement_type)   — canonical vocabulary per statement
    get_prompt_vocab(statement_type)— newline-joined vocab for VLM prompt
    classify_section(text)          — section heuristic for header detection (§14)

The schema mirrors §19 ``NomenclatureEntry`` and includes the column_model /
section registries from §15/§27.
"""

from __future__ import annotations

import itertools
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import yaml
from rapidfuzz import fuzz, process, utils

logger = logging.getLogger(__name__)

# Replacement characters: U+FFFD is what Python produces for undecodable bytes,
# and "?" is the ASCII fallback some extractors emit instead.
_WILDCARD_CHARS = ("�", "?")

# Cartesian-expansion cap: "b�n�fice" with ~40 unicode candidates per slot
# would explode. Cap at 144 per §9.
# For 1-slot cases 40 candidates is trivial. For 2-slot cases we need
# 40*40 = 1600 combinations to cover all accent pairs. Cap at 3000 which
# still takes microseconds to enumerate and prevents 3+-slot blow-up.
_MAX_FFFD_CANDIDATES = 3000

# Fuzzy thresholds (§13)
_FUZZ_HIGH = 85      # auto-accept as green
_FUZZ_MED_CUTOFF = 75  # below this, treat as custom/unrecognized

# Candidate character set used to fill U+FFFD. We bias toward accented French
# letters + common ligatures. The 27-char French alphabet is plenty.
_FFFD_FILL_CHARS = "éèêëàâäîïôöùûüçñÉÈÊËÀÂÄÎÏÔÖÙÛÜÇÑeaiouycnEAIOUYCN"


def _normalize(text: str) -> str:
    """Accent-strip + lowercase + collapse whitespace — used for matching keys."""
    if not text:
        return ""
    # NFKD then drop combining marks → strip accents
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    lowered = stripped.lower()
    # Apostrophes / ampersands -> space, so "l'exercice" == "l exercice"
    # and the ampersand forms ("stocks & en-cours") still match the
    # "et"-joined canonical form after token normalization.
    for sep in ("'", "’", "‘", "`", "&"):
        lowered = lowered.replace(sep, " ")
    # Collapse any whitespace (incl. NBSP) to single space
    collapsed = re.sub(r"[\s  ]+", " ", lowered).strip()
    return collapsed


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NomenclatureEntry:
    """One accounting term — see §19."""

    canonical_term: str
    normalized: str
    account_code: Optional[str] = None
    class_num: Optional[int] = None
    statement_type: str = "bilan"        # bilan | compte_resultat | flux_tresorerie
    section: Optional[str] = None
    sub_type: Optional[str] = None        # actif | passif | null
    validation_field: Optional[str] = None
    is_subtotal: bool = False
    decomposable: bool = False
    parent_subtotal: Optional[str] = None
    column_model: Optional[str] = None
    variations: list[str] = field(default_factory=list)

    def normalized_variations(self) -> list[str]:
        """All variation forms pre-normalized for fast lookup."""
        return [_normalize(v) for v in self.variations]


@dataclass
class MatchResult:
    """Outcome of ``fuzzy_match`` — mirrors the canonical output schema (§5)."""

    entry: Optional[NomenclatureEntry]
    match_type: str                       # exact | resolved | fuzzy | custom | unrecognized
    confidence: float
    score: float = 0.0                     # raw fuzz score 0..100 (0 if non-fuzzy)


# ---------------------------------------------------------------------------
# Dictionary
# ---------------------------------------------------------------------------


class NomenclatureDictionary:
    """Central dictionary — loads YAML once, serves all consumers."""

    def __init__(
        self,
        entries: list[NomenclatureEntry],
        column_models: dict,
        sections: dict,
        meta: Optional[dict] = None,
    ):
        self.entries: list[NomenclatureEntry] = entries
        self.column_models: dict = column_models
        self.sections: dict = sections
        self.meta: dict = meta or {}

        # Indices --------------------------------------------------------
        # Exact normalized → entry (first canonical wins; variations do not
        # override canonicals with the same normalized form).
        self._by_normalized: dict[str, NomenclatureEntry] = {}
        # Variation normalized → entry (§13 Stage 2)
        self._by_variation: dict[str, NomenclatureEntry] = {}
        # statement_type → [entries]
        self._by_statement: dict[str, list[NomenclatureEntry]] = {}
        # section → [entries]
        self._by_section: dict[str, list[NomenclatureEntry]] = {}
        # validation_field → entry
        self._by_validation_field: dict[str, NomenclatureEntry] = {}
        # account_code → entry (first registered wins)
        self._by_account_code: dict[str, NomenclatureEntry] = {}

        # Variation logger (§28) — optional, loaded lazily
        self._variation_logger = None
        self._approved_variations: dict[str, str] = {}  # raw → canonical
        self._confirmed_customs: set[str] = set()
        self._load_variation_overrides()

        self._build_indices()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: Path | str) -> "NomenclatureDictionary":
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        raw_entries = data.get("entries", []) or []
        entries: list[NomenclatureEntry] = []
        for raw in raw_entries:
            canonical = raw.get("canonical_term", "").strip()
            if not canonical:
                continue
            normalized = raw.get("normalized") or _normalize(canonical)
            entries.append(
                NomenclatureEntry(
                    canonical_term=canonical,
                    normalized=normalized,
                    account_code=raw.get("account_code"),
                    class_num=raw.get("class_num"),
                    statement_type=raw.get("statement_type", "bilan"),
                    section=raw.get("section"),
                    sub_type=raw.get("sub_type"),
                    validation_field=raw.get("validation_field"),
                    is_subtotal=bool(raw.get("is_subtotal", False)),
                    decomposable=bool(raw.get("decomposable", False)),
                    parent_subtotal=raw.get("parent_subtotal"),
                    column_model=raw.get("column_model"),
                    variations=list(raw.get("variations") or []),
                )
            )

        return cls(
            entries=entries,
            column_models=data.get("column_models", {}) or {},
            sections=data.get("sections", {}) or {},
            meta=data.get("meta", {}) or {},
        )

    def _load_variation_overrides(self) -> None:
        """Load approved variations + confirmed customs from the log (§28)."""
        try:
            from accounting.variation_logger import VariationLogger
            self._variation_logger = VariationLogger()
            self._approved_variations = self._variation_logger.get_approved_variations()
            self._confirmed_customs = self._variation_logger.get_confirmed_customs()
            if self._approved_variations:
                logger.info(
                    "Loaded %d approved variation overrides from log",
                    len(self._approved_variations),
                )
        except Exception:  # noqa: BLE001 — variation logger is optional
            self._variation_logger = None
            self._approved_variations = {}
            self._confirmed_customs = set()

    def _build_indices(self) -> None:
        for entry in self.entries:
            # Canonical index — skip collisions so the first registration wins.
            # (The YAML intentionally lists "Résultat net de l'exercice" under
            # both compte_resultat and flux_tresorerie; the first is the
            # authoritative mapping for exact lookups.)
            self._by_normalized.setdefault(entry.normalized, entry)
            # Variations index
            for v_norm in entry.normalized_variations():
                if v_norm and v_norm != entry.normalized:
                    self._by_variation.setdefault(v_norm, entry)
            # Statement / section buckets
            self._by_statement.setdefault(entry.statement_type, []).append(entry)
            if entry.section:
                self._by_section.setdefault(entry.section, []).append(entry)
            # Validation field index
            if entry.validation_field:
                self._by_validation_field.setdefault(entry.validation_field, entry)
            # Account code index
            if entry.account_code:
                self._by_account_code.setdefault(str(entry.account_code), entry)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def all_entries(self) -> list[NomenclatureEntry]:
        return self.entries

    def get_terms_for(self, statement_type: str) -> list[NomenclatureEntry]:
        """All canonical entries scoped to one statement type."""
        return list(self._by_statement.get(statement_type, []))

    def get_prompt_vocab(self, statement_type: str) -> str:
        """Newline-joined canonical terms — for grounding the VLM prompt (§6)."""
        terms = [e.canonical_term for e in self.get_terms_for(statement_type)]
        # Deduplicate while preserving order
        seen = set()
        unique = []
        for t in terms:
            if t not in seen:
                seen.add(t)
                unique.append(t)
        return "\n".join(f"- {t}" for t in unique)

    def map_to_field(self, canonical_term: str) -> Optional[str]:
        """Canonical term or known variation → validation_field.

        Bridge to ``validation_rules.json``. Falls through to the variations
        index so aliases (e.g. "chiffre d'affaires" → ``Revenus``) resolve
        without the caller having to run ``fuzzy_match`` first.
        """
        norm = _normalize(canonical_term)
        entry = self._by_normalized.get(norm) or self._by_variation.get(norm)
        return entry.validation_field if entry else None

    def lookup(self, canonical_term: str) -> Optional[NomenclatureEntry]:
        """Direct lookup by canonical term or known variation (normalized)."""
        norm = _normalize(canonical_term)
        return self._by_normalized.get(norm) or self._by_variation.get(norm)

    def lookup_by_account(self, code: str) -> Optional[NomenclatureEntry]:
        return self._by_account_code.get(str(code))

    def get_column_model(self, name: str) -> Optional[dict]:
        return self.column_models.get(name)

    def get_section(self, name: str) -> Optional[dict]:
        return self.sections.get(name)

    # ------------------------------------------------------------------
    # FFFD resolution (§9)
    # ------------------------------------------------------------------

    def resolve_fffd(self, text: str) -> tuple[str, float, int]:
        """Replace ``\\uFFFD`` / ``?`` placeholders with accented candidates.

        Returns ``(resolved_text, confidence, candidate_count)``:
            - ``resolved_text``: the best canonical form (or original if no
              dictionary hit is found).
            - ``confidence``: 0.99 for unambiguous single match, 0.80 for 2-3
              candidates, 0.50 for >3 (§18 confidence gate), 1.0 if nothing
              to resolve.
            - ``candidate_count``: how many dictionary entries matched the
              wildcard pattern (1 = unambiguous).
        """
        if not text:
            return text, 1.0, 0
        if not any(ch in text for ch in _WILDCARD_CHARS):
            return text, 1.0, 0

        # Normalize both wildcards to a single "?" for uniform handling.
        probe = text
        for wc in _WILDCARD_CHARS:
            probe = probe.replace(wc, "?")

        slot_positions = [i for i, ch in enumerate(probe) if ch == "?"]
        # Bound the search: if too many wildcards, skip expensive Cartesian.
        if len(_FFFD_FILL_CHARS) ** len(slot_positions) > _MAX_FFFD_CANDIDATES * 10:
            # Too many slots — fall back to full-phrase fuzzy match.
            mr = self.fuzzy_match(probe.replace("?", ""))
            if mr.entry:
                return mr.entry.canonical_term, min(0.80, mr.confidence), 1
            return text, 0.50, 0

        # Enumerate candidate fills, but cap at _MAX_FFFD_CANDIDATES.
        fills = itertools.islice(
            itertools.product(_FFFD_FILL_CHARS, repeat=len(slot_positions)),
            _MAX_FFFD_CANDIDATES,
        )

        probe_norm = _normalize(probe.replace("?", ""))

        hits: list[NomenclatureEntry] = []
        seen_ids: set[int] = set()

        for fill in fills:
            chars = list(probe)
            for pos, ch in zip(slot_positions, fill):
                chars[pos] = ch
            candidate_norm = _normalize("".join(chars))
            entry = self._by_normalized.get(candidate_norm) or self._by_variation.get(
                candidate_norm
            )
            if entry and id(entry) not in seen_ids:
                seen_ids.add(id(entry))
                hits.append(entry)

        if not hits:
            # No exact dictionary hit — try fuzzy against the wildcard-stripped form.
            mr = self.fuzzy_match(probe.replace("?", ""))
            if mr.entry and mr.confidence >= 0.80:
                return mr.entry.canonical_term, min(0.80, mr.confidence), 1
            return text, 0.50, 0

        if len(hits) == 1:
            return hits[0].canonical_term, 0.99, 1
        if len(hits) <= 3:
            # Ambiguous — prefer the one whose normalized form is closest
            best = max(
                hits,
                key=lambda e: fuzz.ratio(e.normalized, probe_norm, processor=utils.default_process),
            )
            return best.canonical_term, 0.80, len(hits)
        return hits[0].canonical_term, 0.50, len(hits)

    # ------------------------------------------------------------------
    # Fuzzy matching (§13)
    # ------------------------------------------------------------------

    def fuzzy_match(
        self,
        raw_label: str,
        section: Optional[str] = None,
        statement_type: Optional[str] = None,
        document_id: str = "",
    ) -> MatchResult:
        """3-stage cascade: exact → variation → rapidfuzz.

        ``section`` narrows the search to one section's entries when known
        (much faster and more accurate — see §13 section scoping).
        ``statement_type`` narrows similarly (used when section is unknown).
        ``document_id`` is forwarded to the variation logger (§28) so
        promotable candidates can be traced back to source documents.
        """
        result = self._fuzzy_match_inner(raw_label, section, statement_type)
        # §28 — log fuzzy/resolved matches with confidence < 0.95 for the
        # candidate→promoted lifecycle. ``log_variation_candidate`` itself
        # filters by match_type/confidence, so this is just a forwarding hook.
        if (
            self._variation_logger is not None
            and result.entry is not None
            and result.match_type in ("fuzzy", "resolved")
        ):
            try:
                self._variation_logger.log_variation_candidate(
                    raw_text=raw_label,
                    canonical_term=result.entry.canonical_term,
                    match_type=result.match_type,
                    confidence=result.confidence,
                    section=section,
                    document_id=document_id,
                )
            except Exception:  # noqa: BLE001 — logging is best-effort
                pass
        return result

    def _fuzzy_match_inner(
        self,
        raw_label: str,
        section: Optional[str] = None,
        statement_type: Optional[str] = None,
    ) -> MatchResult:
        """Core matcher — pure function with no logging side-effects."""
        if not raw_label:
            return MatchResult(entry=None, match_type="unrecognized", confidence=0.0)

        normalized = _normalize(raw_label)
        if not normalized:
            return MatchResult(entry=None, match_type="unrecognized", confidence=0.0)

        # Stage 0 — confirmed custom bypass (§28)
        if raw_label in self._confirmed_customs:
            return MatchResult(entry=None, match_type="custom", confidence=0.0)

        scope = self._resolve_scope(section, statement_type)

        # Stage 1 — exact normalized lookup
        if scope is self.entries:
            entry = self._by_normalized.get(normalized) or self._by_variation.get(
                normalized
            )
            if entry:
                match_type = "exact" if entry.normalized == normalized else "resolved"
                confidence = 1.0 if match_type == "exact" else 0.98
                return MatchResult(entry=entry, match_type=match_type, confidence=confidence, score=100.0)
        else:
            for entry in scope:
                if entry.normalized == normalized:
                    return MatchResult(entry=entry, match_type="exact", confidence=1.0, score=100.0)
            for entry in scope:
                if normalized in entry.normalized_variations():
                    return MatchResult(entry=entry, match_type="resolved", confidence=0.98, score=98.0)

        # Stage 2.5 — approved variation override (§28 hot-reload)
        if raw_label in self._approved_variations:
            canonical = self._approved_variations[raw_label]
            entry = self._by_normalized.get(_normalize(canonical))
            if entry:
                return MatchResult(entry=entry, match_type="resolved", confidence=0.98, score=98.0)

        # Stage 3 — rapidfuzz (token_sort_ratio, then partial_ratio)
        candidates = {e.normalized: e for e in scope}
        if not candidates:
            return MatchResult(entry=None, match_type="custom", confidence=0.0)

        best_result = process.extractOne(
            normalized,
            list(candidates.keys()),
            scorer=fuzz.token_sort_ratio,
            processor=utils.default_process,
            score_cutoff=_FUZZ_MED_CUTOFF,
        )
        if best_result is not None:
            matched_key, score, _ = best_result
            if score >= _FUZZ_HIGH:
                return MatchResult(
                    entry=candidates[matched_key],
                    match_type="fuzzy",
                    confidence=score / 100.0,
                    score=float(score),
                )

        partial_result = process.extractOne(
            normalized,
            list(candidates.keys()),
            scorer=fuzz.partial_ratio,
            processor=utils.default_process,
            score_cutoff=80,
        )
        if partial_result is not None:
            matched_key, score, _ = partial_result
            if score >= _FUZZ_HIGH:
                return MatchResult(
                    entry=candidates[matched_key],
                    match_type="fuzzy",
                    confidence=score / 100.0,
                    score=float(score),
                )

        # Best available below threshold — return as custom with score info
        best_score = 0.0
        best_entry = None
        if best_result is not None:
            matched_key, best_score, _ = best_result
            best_entry = candidates[matched_key]
        return MatchResult(
            entry=None,
            match_type="custom",
            confidence=best_score / 100.0 if best_score else 0.0,
            score=float(best_score),
        )

    def _resolve_scope(
        self, section: Optional[str], statement_type: Optional[str]
    ) -> list[NomenclatureEntry]:
        if section and section in self._by_section:
            return self._by_section[section]
        if statement_type and statement_type in self._by_statement:
            return self._by_statement[statement_type]
        return self.entries

    # ------------------------------------------------------------------
    # Statement classification (§24)
    # ------------------------------------------------------------------

    def classify_statement(self, rows: Iterable[str]) -> str:
        """Classify a table by term distribution — Nomenclature-powered (§24)."""
        scores = {"bilan": 0, "compte_resultat": 0, "flux_tresorerie": 0}
        texts = [r for r in rows if r]
        if not texts:
            return "unknown"

        for raw_text in texts:
            mr = self.fuzzy_match(raw_text)
            if mr.entry and mr.match_type in ("exact", "fuzzy", "resolved"):
                scores[mr.entry.statement_type] = scores.get(mr.entry.statement_type, 0) + 1

        # Structural boosters
        joined = " ".join(texts).lower()
        if any(kw in joined for kw in ("brut", "amort", "amortissement", "dépréc", "deprec")):
            # Brut/Amort/Net column layout → Bilan Actif
            scores["bilan"] += 3
        if any(kw in joined for kw in (
            "actifs non courants", "actifs courants", "actifs immobilis",
            "total des actifs", "total de l'actif", "total actif",
            "immobilisations incorporelles", "immobilisations corporelles",
        )):
            # Bilan Actif structural terms
            scores["bilan"] += 3
        if any(kw in joined for kw in (
            "capitaux propres", "passifs non courants", "passifs courants",
            "total des passifs", "total du passif", "total passif",
            "total des capitaux propres", "capital social",
            "capitaux propres et passifs",
        )):
            # Bilan Passif structural terms
            scores["bilan"] += 3
        if any(kw in joined for kw in ("résultat d'exploitation", "resultat d'exploitation",
                                        "produits d'exploitation", "charges d'exploitation",
                                        "charges de personnel", "achats consomm")):
            scores["compte_resultat"] += 3
        if any(kw in joined for kw in ("flux de tr", "flux net", "trésorerie de clôture",
                                        "tresorerie de cloture", "activités d'exploitation",
                                        "activités d'investissement", "activités de financement")):
            scores["flux_tresorerie"] += 3

        winner = max(scores, key=scores.get)
        if scores[winner] >= 2:
            return winner
        return "unknown"

    def classify_statement_with_confidence(self, rows: Iterable[str]) -> tuple[str, float]:
        """Classify statement rows and return (label, confidence).

        Label is one of: 'bilan', 'compte_resultat', 'flux_tresorerie', or 'unknown'.
        Confidence is a float between 0.0 and 1.0 based on score concentration.
        """
        scores = {"bilan": 0.0, "compte_resultat": 0.0, "flux_tresorerie": 0.0}
        texts = [r for r in rows if r]
        if not texts:
            return "unknown", 0.0

        for raw_text in texts:
            mr = self.fuzzy_match(raw_text)
            if mr.entry and mr.match_type in ("exact", "fuzzy", "resolved"):
                scores[mr.entry.statement_type] = scores.get(mr.entry.statement_type, 0.0) + 1.0

        # Structural boosters (same as classify_statement)
        joined = " ".join(texts).lower()
        if any(kw in joined for kw in ("brut", "amort", "amortissement", "dépréc", "deprec")):
            scores["bilan"] += 3.0
        if any(kw in joined for kw in (
            "actifs non courants", "actifs courants", "actifs immobilis",
            "total des actifs", "total de l'actif", "total actif",
            "immobilisations incorporelles", "immobilisations corporelles",
        )):
            scores["bilan"] += 3.0
        if any(kw in joined for kw in (
            "capitaux propres", "passifs non courants", "passifs courants",
            "total des passifs", "total du passif", "total passif",
            "total des capitaux propres", "capital social",
            "capitaux propres et passifs",
        )):
            scores["bilan"] += 3.0
        if any(kw in joined for kw in ("résultat d'exploitation", "resultat d'exploitation",
                                        "produits d'exploitation", "charges d'exploitation",
                                        "charges de personnel", "achats consomm")):
            scores["compte_resultat"] += 3.0
        if any(kw in joined for kw in ("flux de tr", "flux net", "trésorerie de clôture",
                                        "tresorerie de cloture", "activités d'exploitation",
                                        "activités d'investissement", "activités de financement")):
            scores["flux_tresorerie"] += 3.0

        # Determine winner and a conservative confidence metric based on concentration
        total = sum(scores.values())
        if total <= 0:
            return "unknown", 0.0
        winner = max(scores, key=scores.get)
        # Confidence: proportion of winner score to total, scaled and clipped
        raw_conf = scores[winner] / total
        # Apply a small sigmoid-ish scaling to stretch mid-range
        confidence = max(0.0, min(1.0, (raw_conf * 1.2)))
        return winner, float(confidence)

    # ------------------------------------------------------------------
    # Section classifier (for header detection — §14 Signal 3)
    # ------------------------------------------------------------------

    def classify_section(self, text: str) -> Optional[str]:
        """Return a section key (e.g. 'actifs_non_courants') if ``text``
        matches a section label.

        Used by the font-based header detector (§14 Signal 3) and the OCR
        bbox-based fallback (§25).
        """
        if not text:
            return None
        norm = _normalize(text)
        if not norm:
            return None
        # Direct section-label match
        for section_key, cfg in self.sections.items():
            label_norm = _normalize(cfg.get("label_canonical", ""))
            if label_norm and (label_norm in norm or norm in label_norm):
                return section_key
        # Looser keyword matches for Bilan sides
        if any(kw in norm for kw in ("actifs non courants", "actif non courant",
                                      "actifs immobilises", "actif immobilise")):
            return "actifs_non_courants"
        if "actifs courants" in norm or "actif courant" in norm:
            return "actifs_courants"
        if "capitaux propres" in norm:
            return "capitaux_propres"
        if "passifs non courants" in norm or "passif non courant" in norm:
            return "passifs_non_courants"
        if "passifs courants" in norm or "passif courant" in norm:
            return "passifs_courants"
        return None


# ---------------------------------------------------------------------------
# Default loader (cached)
# ---------------------------------------------------------------------------


_DEFAULT_YAML_PATH = (
    Path(__file__).resolve().parent / "rules" / "nomenclature_data.yaml"
)


@lru_cache(maxsize=1)
def load_default_dictionary() -> NomenclatureDictionary:
    """Load ``src/accounting/rules/nomenclature_data.yaml`` (cached)."""
    return NomenclatureDictionary.from_yaml(_DEFAULT_YAML_PATH)
