"""FAISS vector index.

Spec section 5 names FAISS as the vector database. Vectors are unit-normalised by
``embeddings``, so inner product and cosine similarity are the same measure and
``IndexFlatIP`` is the correct index type.

If ``faiss`` is not importable the same API is served by a pure-Python dot-product
scan. That is O(n) per query rather than FAISS's optimised scan, which is fine for
a per-contract clause count (tens to low hundreds) and keeps the stack runnable
with no native dependency. :attr:`VectorIndex.backend` reports which is in use.
"""

import logging
from typing import List, Optional, Sequence, Tuple

from .embeddings import cosine

logger = logging.getLogger(__name__)


def _load_faiss():
    try:
        import faiss

        return faiss
    except Exception as error:
        logger.info("FAISS unavailable, using the pure-Python vector scan: %s", error)
        return None


class VectorIndex:
    def __init__(self, dimensions: int, use_faiss: bool = True) -> None:
        self._dimensions = dimensions
        self._ids: List[str] = []
        self._vectors: List[List[float]] = []
        self._faiss = _load_faiss() if use_faiss else None
        self._index = None

        if self._faiss is not None:
            self._index = self._faiss.IndexFlatIP(dimensions)

    @property
    def backend(self) -> str:
        return "faiss" if self._index is not None else "python"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> int:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must be the same length")

        accepted: List[List[float]] = []
        accepted_ids: List[str] = []

        for chunk_id, vector in zip(ids, vectors):
            if len(vector) != self._dimensions:
                raise ValueError(
                    f"vector for '{chunk_id}' has {len(vector)} dimensions, "
                    f"index expects {self._dimensions}"
                )
            accepted_ids.append(chunk_id)
            accepted.append([float(value) for value in vector])

        if not accepted:
            return 0

        self._ids.extend(accepted_ids)
        self._vectors.extend(accepted)

        if self._index is not None:
            try:
                import numpy

                self._index.add(numpy.asarray(accepted, dtype="float32"))
            except Exception as error:
                logger.warning(
                    "FAISS add failed, dropping to the Python scan for this index: %s", error
                )
                self._index = None

        return len(accepted)

    def clear(self) -> None:
        self._ids = []
        self._vectors = []

        if self._faiss is not None:
            self._index = self._faiss.IndexFlatIP(self._dimensions)

    def search(
        self,
        vector: Sequence[float],
        top_k: int = 25,
        allowed_ids: Optional[Sequence[str]] = None,
    ) -> List[Tuple[str, float]]:
        """Return ``(chunk_id, similarity)`` pairs, best first.

        ``allowed_ids`` is the metadata pre-filter from spec section 8. FAISS has no
        native id filter on a flat index, so when a filter is supplied the scan runs
        in Python over the filtered subset, which is cheaper than searching the whole
        index and discarding.
        """

        if not self._ids or not vector or top_k <= 0:
            return []

        allowed = set(allowed_ids) if allowed_ids is not None else None

        if allowed is not None and not allowed:
            return []

        if self._index is not None and allowed is None:
            hits = self._faiss_search(vector, top_k)
            if hits is not None:
                return hits

        scored = [
            (chunk_id, cosine(vector, candidate))
            for chunk_id, candidate in zip(self._ids, self._vectors)
            if allowed is None or chunk_id in allowed
        ]

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def _faiss_search(
        self, vector: Sequence[float], top_k: int
    ) -> Optional[List[Tuple[str, float]]]:
        try:
            import numpy

            query = numpy.asarray([list(vector)], dtype="float32")
            scores, indices = self._index.search(query, min(top_k, len(self._ids)))

            return [
                (self._ids[position], float(score))
                for score, position in zip(scores[0], indices[0])
                if position >= 0
            ]
        except Exception as error:
            logger.warning("FAISS search failed, using the Python scan: %s", error)
            return None
