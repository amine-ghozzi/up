"""
Smart OCR Text Corrector

Replaces the naive dictionary-lookup approach with:
1. Vocabulary extraction from YAML correction files
2. Word-break DP to split concatenated ALL-CAPS strings
3. Fuzzy matching (Levenshtein) to correct misspelled words

The vocabulary is built once from the YAML corrections and cached.
This handles both known and unknown concatenations/misspellings.
"""

import re
import logging
from pathlib import Path
from functools import lru_cache
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# Accent Normalization
# =============================================================================

_ACCENT_MAP = str.maketrans(
    "àâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇ",
    "aaaeeeeiioouucAAAEEEEIIOOUUC",
)


def _normalize(text: str) -> str:
    """Normalize to uppercase ASCII for matching."""
    return text.upper().translate(_ACCENT_MAP)


# =============================================================================
# Vocabulary Builder
# =============================================================================

_vocab_cache: Dict[str, "Vocabulary"] = {}


class Vocabulary:
    """
    Accounting vocabulary extracted from YAML correction files.

    Stores words in both accented (display) and normalized (matching) forms.
    """

    def __init__(self, words_with_accents: Set[str]):
        # canonical_form[NORMALIZED] = "Accented" (for display)
        self.canonical: Dict[str, str] = {}
        # All normalized forms for fast lookup
        self.normalized: Set[str] = set()
        # Words grouped by length for fuzzy search pruning
        self.by_length: Dict[int, List[str]] = {}

        for w in words_with_accents:
            norm = _normalize(w)
            self.normalized.add(norm)
            # Keep the version with accents for output
            if norm not in self.canonical or len(w) > len(self.canonical[norm]):
                self.canonical[norm] = w

            length = len(norm)
            if length not in self.by_length:
                self.by_length[length] = []
            self.by_length[length].append(norm)

        # Min/max word lengths for DP bounds
        self.min_len = min(len(w) for w in self.normalized) if self.normalized else 1
        self.max_len = max(len(w) for w in self.normalized) if self.normalized else 20

    def __contains__(self, word: str) -> bool:
        return _normalize(word) in self.normalized

    def get_display(self, normalized_word: str) -> str:
        """Get the accented display form of a normalized word."""
        return self.canonical.get(normalized_word, normalized_word)


def build_vocabulary(standard: str) -> Vocabulary:
    """Build vocabulary from YAML correction file for a given standard."""
    if standard in _vocab_cache:
        return _vocab_cache[standard]

    terms_dir = Path(__file__).parent / "terms"
    yaml_path = terms_dir / f"{standard.lower()}.yaml"

    words: Set[str] = set()

    # Common French function words (always in vocabulary)
    function_words = {
        "DE", "DES", "DU", "ET", "EN", "AU", "AUX", "LE", "LA", "LES",
        "UN", "UNE", "SUR", "PAR", "POUR", "AVEC", "DANS", "MOINS", "PLUS",
        "OU", "NE", "PAS", "À", "A", "D", "L", "LIÉS", "LIÉES", "LIÉ",
        "NON", "NET", "NETS", "NETTE", "NETTES",
    }
    words.update(function_words)

    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            for section, items in data.items():
                if section == "metadata" or not isinstance(items, list):
                    continue
                for item in items:
                    if isinstance(item, dict) and "correction" in item:
                        for w in item["correction"].split():
                            # Strip punctuation, keep only letters
                            clean = re.sub(r"[^a-zA-ZÀ-ÿ']", "", w)
                            if clean and len(clean) >= 1:
                                words.add(clean.upper())
        except Exception as e:
            logger.error(f"Failed to load vocabulary from {yaml_path}: {e}")

    vocab = Vocabulary(words)
    _vocab_cache[standard] = vocab
    logger.info(f"Built vocabulary for {standard}: {len(vocab.normalized)} words")
    return vocab


def clear_vocab_cache():
    """Clear cached vocabularies."""
    global _vocab_cache
    _vocab_cache = {}


# =============================================================================
# Levenshtein Distance
# =============================================================================

def _levenshtein(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)

    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            # Cost is 0 if characters match, 1 otherwise
            cost = 0 if c1 == c2 else 1
            curr_row.append(min(
                curr_row[j] + 1,       # insertion
                prev_row[j + 1] + 1,   # deletion
                prev_row[j] + cost,    # substitution
            ))
        prev_row = curr_row

    return prev_row[-1]


def fuzzy_match(word: str, vocab: Vocabulary, max_distance: int = 2) -> Optional[str]:
    """
    Find the closest vocabulary word by Levenshtein distance.

    Args:
        word: Normalized (uppercase, no accents) word to match
        vocab: Vocabulary to search
        max_distance: Maximum edit distance to accept

    Returns:
        Best matching normalized word, or None if no close match
    """
    if word in vocab.normalized:
        return word

    word_len = len(word)
    best_match = None
    best_dist = max_distance + 1

    # Only check words within ±max_distance length (others can't match)
    for length in range(max(1, word_len - max_distance), word_len + max_distance + 1):
        candidates = vocab.by_length.get(length, [])
        for candidate in candidates:
            dist = _levenshtein(word, candidate)
            if dist < best_dist:
                best_dist = dist
                best_match = candidate
                if dist == 1:
                    # Good enough — don't search further
                    return best_match

    return best_match if best_dist <= max_distance else None


# =============================================================================
# Word-Break DP
# =============================================================================

def word_break(text: str, vocab: Vocabulary) -> Optional[List[str]]:
    """
    Split a concatenated string into vocabulary words using dynamic programming.

    Uses exact vocabulary matching. Returns None if no valid segmentation found.

    Args:
        text: Normalized uppercase string (e.g., "TOTALACTIFSNONCOURANTS")
        vocab: Vocabulary for word lookup

    Returns:
        List of normalized words, or None if unsplittable
    """
    n = len(text)
    if n == 0:
        return []

    # dp[i] = list of words that segment text[:i], or None
    dp: List[Optional[List[str]]] = [None] * (n + 1)
    dp[0] = []

    for i in range(1, n + 1):
        # Try all possible last-word lengths
        for length in range(vocab.min_len, min(vocab.max_len, i) + 1):
            start = i - length
            if dp[start] is None:
                continue
            candidate = text[start:i]
            if candidate in vocab.normalized:
                new_seg = dp[start] + [candidate]
                # Prefer fewer words (longer matches)
                if dp[i] is None or len(new_seg) < len(dp[i]):
                    dp[i] = new_seg

    return dp[n]


def word_break_fuzzy(text: str, vocab: Vocabulary, max_distance: int = 1) -> Optional[List[str]]:
    """
    Split a concatenated string allowing fuzzy word matches.

    First tries exact word-break. If that fails, tries with fuzzy matching
    on each segment.

    Args:
        text: Normalized uppercase string
        vocab: Vocabulary
        max_distance: Max edit distance per word

    Returns:
        List of corrected normalized words, or None
    """
    # Try exact first
    exact = word_break(text, vocab)
    if exact is not None:
        return exact

    n = len(text)
    if n == 0:
        return []

    # dp[i] = (word_list, total_distance) or None
    dp: List[Optional[Tuple[List[str], int]]] = [None] * (n + 1)
    dp[0] = ([], 0)

    for i in range(1, n + 1):
        for length in range(max(2, vocab.min_len), min(vocab.max_len + max_distance, i) + 1):
            start = i - length
            if dp[start] is None:
                continue

            candidate = text[start:i]
            prev_words, prev_dist = dp[start]

            # Try exact match first
            if candidate in vocab.normalized:
                total_dist = prev_dist
                matched = candidate
            else:
                # Try fuzzy match
                matched = fuzzy_match(candidate, vocab, max_distance)
                if matched is None:
                    continue
                total_dist = prev_dist + _levenshtein(candidate, matched)

            new_entry = (prev_words + [matched], total_dist)

            if dp[i] is None or (
                # Prefer lower total distance, then fewer words
                total_dist < dp[i][1]
                or (total_dist == dp[i][1] and len(new_entry[0]) < len(dp[i][0]))
            ):
                dp[i] = new_entry

    if dp[n] is None:
        return None
    return dp[n][0]


# =============================================================================
# Main Correction Function
# =============================================================================

def smart_correct(text: str, vocab: Vocabulary) -> str:
    """
    Correct a single OCR text token (typically a table cell label).

    Strategy:
    1. If text has spaces, correct each word independently
    2. If text is a single concatenated token, try word-break + fuzzy
    3. Fall back to fuzzy match on the whole token

    Args:
        text: Raw OCR text (single cell or label)
        vocab: Vocabulary for the accounting standard

    Returns:
        Corrected text with proper spacing and accents
    """
    if not text or not text.strip():
        return text

    stripped = text.strip()

    # Skip numeric values, short tokens
    if re.match(r'^[\d\s.,%-]+$', stripped):
        return text

    # Split on existing spaces
    raw_words = stripped.split()

    corrected_words = []
    for raw_word in raw_words:
        corrected = _correct_word(raw_word, vocab)
        corrected_words.append(corrected)

    return " ".join(corrected_words)


def _correct_word(word: str, vocab: Vocabulary) -> str:
    """Correct a single word (no spaces). May return multiple words if concatenated."""
    # Strip punctuation from edges but preserve them
    prefix = ""
    suffix = ""
    core = word

    # Extract leading/trailing punctuation
    m = re.match(r'^([^a-zA-ZÀ-ÿ]*)(.*?)([^a-zA-ZÀ-ÿ]*)$', word)
    if m:
        prefix, core, suffix = m.group(1), m.group(2), m.group(3)

    if not core:
        return word

    norm = _normalize(core)

    # 1. Exact vocabulary match — just fix accents
    if norm in vocab.normalized:
        display = vocab.get_display(norm)
        # Preserve original casing style
        return prefix + _match_case(display, core) + suffix

    # 2. Short word — try fuzzy only (word-break on 3 chars is pointless)
    if len(norm) <= 4:
        matched = fuzzy_match(norm, vocab, max_distance=1)
        if matched:
            return prefix + _match_case(vocab.get_display(matched), core) + suffix
        return word

    # 3. Longer token — try word-break (exact then fuzzy)
    segments = word_break_fuzzy(norm, vocab, max_distance=1)
    if segments:
        display_words = [vocab.get_display(s) for s in segments]
        # Apply case matching to each
        result = " ".join(display_words)
        return prefix + _match_case_phrase(result, core) + suffix

    # 4. Last resort — fuzzy on the whole thing (maybe just a typo)
    matched = fuzzy_match(norm, vocab, max_distance=2)
    if matched:
        return prefix + _match_case(vocab.get_display(matched), core) + suffix

    return word


def _match_case(corrected: str, original: str) -> str:
    """Match the casing style of the original to the corrected word."""
    if original.isupper():
        return corrected.upper()
    elif original.islower():
        return corrected.lower()
    elif original and original[0].isupper():
        return corrected[0].upper() + corrected[1:].lower() if len(corrected) > 1 else corrected.upper()
    return corrected


def _match_case_phrase(corrected_phrase: str, original: str) -> str:
    """Match casing for a multi-word correction from a single original token."""
    if original.isupper():
        return corrected_phrase.upper()
    elif original.islower():
        return corrected_phrase.lower()
    elif original and original[0].isupper():
        # Title case the first word, lowercase the rest
        words = corrected_phrase.split()
        if words:
            words[0] = words[0][0].upper() + words[0][1:].lower() if len(words[0]) > 1 else words[0].upper()
            for i in range(1, len(words)):
                words[i] = words[i].lower()
        return " ".join(words)
    return corrected_phrase


# =============================================================================
# Batch Correction (for DataFrames)
# =============================================================================

def smart_correct_dataframe(df, standard: str = "NCT") -> int:
    """
    Apply smart correction to all string columns in a DataFrame (in-place).

    Returns total number of cells corrected.
    """
    import pandas as pd

    vocab = build_vocabulary(standard)
    corrections = 0

    for col in df.columns:
        if df[col].dtype == object:
            for idx in df.index:
                val = df.at[idx, col]
                if isinstance(val, str) and val.strip():
                    corrected = smart_correct(val, vocab)
                    if corrected != val:
                        df.at[idx, col] = corrected
                        corrections += 1

    return corrections
