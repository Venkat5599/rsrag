"""BM25 sparse retrieval.

Spec section 5 names BM25 as the sparse arm of hybrid retrieval. Okapi BM25 is
implemented directly here rather than pulled from ``rank_bm25`` because the whole
algorithm is about forty lines, it removes a dependency, and it lets the index
support the incremental ``add`` and the metadata-filtered ``search`` that the hybrid
pipeline needs. ``rank_bm25`` builds a static corpus at construction time and has no
filtered search, so it would have to be rebuilt on every document upload.

Scoring, for query term q against document D:

    idf(q)   = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))
    score    = sum over q of idf(q) * f(q,D) * (k1 + 1)
                                    / (f(q,D) + k1 * (1 - b + b * |D| / avgdl))

Raw BM25 is unbounded, so :meth:`search` also returns a 0-1 normalised score for
fusion with the dense and cross-encoder signals, which are already bounded.
"""

import math
from typing import Dict, List, Optional, Sequence, Tuple

from rag.semantic import tokenize

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


class BM25Index:
    def __init__(self, k1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        self.k1 = k1
        self.b = b
        self._ids: List[str] = []
        self._term_frequencies: Dict[str, Dict[str, int]] = {}
        self._lengths: Dict[str, int] = {}
        self._document_frequency: Dict[str, int] = {}
        self._total_length = 0

    def __len__(self) -> int:
        return len(self._ids)

    @property
    def average_length(self) -> float:
        if not self._ids:
            return 0.0
        return self._total_length / len(self._ids)

    def add(self, chunk_id: str, text: str) -> int:
        tokens = tokenize(text or "")

        if chunk_id in self._term_frequencies:
            self.remove(chunk_id)

        frequencies: Dict[str, int] = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1

        self._ids.append(chunk_id)
        self._term_frequencies[chunk_id] = frequencies
        self._lengths[chunk_id] = len(tokens)
        self._total_length += len(tokens)

        for term in frequencies:
            self._document_frequency[term] = self._document_frequency.get(term, 0) + 1

        return len(tokens)

    def add_many(self, documents: Sequence[Tuple[str, str]]) -> int:
        return sum(self.add(chunk_id, text) for chunk_id, text in documents)

    def remove(self, chunk_id: str) -> bool:
        frequencies = self._term_frequencies.pop(chunk_id, None)

        if frequencies is None:
            return False

        for term in frequencies:
            remaining = self._document_frequency.get(term, 0) - 1
            if remaining > 0:
                self._document_frequency[term] = remaining
            else:
                self._document_frequency.pop(term, None)

        self._total_length -= self._lengths.pop(chunk_id, 0)
        self._ids = [existing for existing in self._ids if existing != chunk_id]
        return True

    def clear(self) -> None:
        self._ids = []
        self._term_frequencies = {}
        self._lengths = {}
        self._document_frequency = {}
        self._total_length = 0

    def idf(self, term: str) -> float:
        total = len(self._ids)

        if not total:
            return 0.0

        seen = self._document_frequency.get(term, 0)
        return math.log(1.0 + (total - seen + 0.5) / (seen + 0.5))

    def score(self, query_tokens: Sequence[str], chunk_id: str) -> float:
        frequencies = self._term_frequencies.get(chunk_id)

        if not frequencies or not query_tokens:
            return 0.0

        length = self._lengths.get(chunk_id, 0)
        average = self.average_length or 1.0
        total = 0.0

        for term in query_tokens:
            frequency = frequencies.get(term, 0)

            if not frequency:
                continue

            denominator = frequency + self.k1 * (1.0 - self.b + self.b * length / average)
            total += self.idf(term) * frequency * (self.k1 + 1.0) / denominator

        return total

    def search(
        self,
        query: str,
        top_k: int = 25,
        allowed_ids: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float, float]]:
        """Return ``(chunk_id, raw_bm25, normalised_bm25)``, best first.

        The normalised score divides by the best raw score in this result set, so it
        is comparable with the dense and cross-encoder signals during fusion. Chunks
        scoring zero are dropped, because a BM25 zero means no query term occurs.
        """

        query_tokens = tokenize(query or "")

        if not query_tokens or not self._ids or top_k <= 0:
            return []

        allowed = set(allowed_ids) if allowed_ids is not None else None

        if allowed is not None and not allowed:
            return []

        scored = [
            (chunk_id, self.score(query_tokens, chunk_id))
            for chunk_id in self._ids
            if allowed is None or chunk_id in allowed
        ]
        scored = [(chunk_id, value) for chunk_id, value in scored if value > 0.0]

        if not scored:
            return []

        scored.sort(key=lambda item: item[1], reverse=True)
        best = scored[0][1] or 1.0

        return [
            (chunk_id, value, min(value / best, 1.0)) for chunk_id, value in scored[:top_k]
        ]
