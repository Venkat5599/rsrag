"""Embedding generation.

Spec section 5 names BAAI/bge-large-en-v1.5 as the embedding model. When
``sentence_transformers`` and the weights are present that is exactly what runs.

When they are not, encoding falls back to a deterministic hashed bag-of-words
projection over ``rag.semantic.tokenize``. That is not a semantic embedding and does
not pretend to be. It exists so the index, the retriever and the whole test suite
stay exercisable offline, and because the same-token-same-dimension property means
cosine similarity over these vectors still carries real lexical signal. Callers can
check :attr:`EmbeddingBackend.is_semantic` to know which mode produced a vector.
"""

import hashlib
import logging
import math
from typing import Dict, List, Optional, Sequence

from rag.semantic import tokenize

logger = logging.getLogger(__name__)

FALLBACK_DIMENSIONS = 384
_HASH_PROJECTIONS = 3


def _hash_positions(token: str, dimensions: int) -> List[int]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=12).digest()
    return [
        int.from_bytes(digest[index * 4:(index + 1) * 4], "big") % dimensions
        for index in range(_HASH_PROJECTIONS)
    ]


def normalise(vector: Sequence[float]) -> List[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 0.0:
        return [0.0] * len(vector)
    return [value / norm for value in vector]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """Dot product of two vectors. Both are expected to be unit length already."""

    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def hashed_vector(text: str, dimensions: int = FALLBACK_DIMENSIONS) -> List[float]:
    vector = [0.0] * dimensions
    tokens = tokenize(text or "")

    if not tokens:
        return vector

    counts: Dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1

    for token, count in counts.items():
        weight = 1.0 + math.log(count)
        for position in _hash_positions(token, dimensions):
            vector[position] += weight

    return normalise(vector)


class EmbeddingBackend:
    def __init__(self, model_name: str, dimensions: int = FALLBACK_DIMENSIONS) -> None:
        self.model_name = model_name
        self._dimensions = dimensions
        self._encoder = None
        self._encoder_loaded = False
        self._cache: Dict[str, List[float]] = {}

    @property
    def dimensions(self) -> int:
        """Vector width, resolved against the real model.

        This forces the load. The width is not knowable until the model is
        resolved - bge-large-en-v1.5 is 1024, the hashed fallback is 384 - and
        callers size a fixed-width index from this value at construction time.
        Returning the fallback width before the model has loaded builds a 384-wide
        index that then rejects every 1024-wide vector the model produces.
        """

        self._load()
        return self._dimensions

    @property
    def is_semantic(self) -> bool:
        """True when a real embedding model produced the vectors."""

        self._load()
        return self._encoder is not None

    def backend_name(self) -> str:
        return self.model_name if self.is_semantic else "hashed_bag_of_words"

    def _load(self):
        if self._encoder_loaded:
            return self._encoder

        self._encoder_loaded = True

        try:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)
            self._dimensions = int(
                self._encoder.get_sentence_embedding_dimension() or self._dimensions
            )
        except Exception as error:
            logger.warning(
                "Embedding model '%s' unavailable, using hashed fallback vectors: %s",
                self.model_name,
                error,
            )
            self._encoder = None

        return self._encoder

    def encode(self, texts: Sequence[str], use_cache: bool = True) -> List[List[float]]:
        if not texts:
            return []

        if not use_cache:
            return self._encode_uncached(list(texts))

        pending = [text for text in dict.fromkeys(texts) if text not in self._cache]

        if pending:
            for text, vector in zip(pending, self._encode_uncached(pending)):
                self._cache[text] = vector

        return [self._cache[text] for text in texts]

    def encode_one(self, text: str, use_cache: bool = True) -> List[float]:
        return self.encode([text], use_cache=use_cache)[0]

    def _encode_uncached(self, texts: Sequence[str]) -> List[List[float]]:
        encoder = self._load()

        if encoder is None:
            return [hashed_vector(text, self._dimensions) for text in texts]

        try:
            raw = encoder.encode(
                list(texts),
                convert_to_numpy=False,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            return [normalise([float(value) for value in vector]) for vector in raw]
        except Exception as error:
            logger.warning("Embedding failed, using hashed fallback vectors: %s", error)
            self._encoder = None
            return [hashed_vector(text, self._dimensions) for text in texts]

    def clear_cache(self) -> None:
        self._cache.clear()

    def cache_size(self) -> int:
        return len(self._cache)


_shared: Optional[EmbeddingBackend] = None


def get_backend(model_name: str) -> EmbeddingBackend:
    """Process-wide backend, so the model loads at most once per model name."""

    global _shared

    if _shared is None or _shared.model_name != model_name:
        _shared = EmbeddingBackend(model_name)

    return _shared
