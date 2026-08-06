"""Cross-encoder reranking.

Spec section 10: retrieve the top 25 candidates, rerank with
BAAI/bge-reranker-large, return the top 5. A cross encoder reads the query and the
clause together instead of comparing two independently-produced vectors, which is
why it separates near-duplicate clauses that dense retrieval scores almost equally.

Without ``sentence_transformers`` the fallback scores each candidate on lexical
overlap with the query, heading agreement, and clause-type importance. That is a
weaker signal than a real cross encoder and is not presented as equivalent, but it
is a genuine reordering rather than a pass-through, so the pipeline shape and the
tests stay honest offline. :meth:`CrossEncoderReranker.is_model_backed` reports which
path produced the scores.
"""

import logging
import math
from typing import List, Optional, Sequence

from rag.clause_taxonomy import clause_importance
from rag.schemas import RetrievedChunk
from rag.semantic import lexical_similarity

logger = logging.getLogger(__name__)

MAX_PAIR_CHARS = 2000


def _squash(value: float) -> float:
    """Map a cross-encoder logit into 0-1.

    bge-reranker returns an unbounded relevance logit, not a probability, so a
    logistic squash is applied before the score is fused with the other signals.
    """

    if 0.0 <= value <= 1.0:
        return value

    try:
        return 1.0 / (1.0 + math.exp(-value))
    except OverflowError:
        return 0.0 if value < 0 else 1.0


class CrossEncoderReranker:
    def __init__(self, model_name: str, enabled: bool = True) -> None:
        self.model_name = model_name
        self._enabled = enabled
        self._encoder = None
        self._loaded = not enabled

    @property
    def is_model_backed(self) -> bool:
        self._load()
        return self._encoder is not None

    def backend_name(self) -> str:
        return self.model_name if self.is_model_backed else "lexical_rerank"

    def _load(self):
        if self._loaded:
            return self._encoder

        self._loaded = True

        try:
            from sentence_transformers import CrossEncoder

            self._encoder = CrossEncoder(self.model_name)
        except Exception as error:
            logger.warning(
                "Cross encoder '%s' unavailable, using the lexical reranker: %s",
                self.model_name,
                error,
            )
            self._encoder = None

        return self._encoder

    def score(self, query: str, chunks: Sequence[RetrievedChunk]) -> List[float]:
        if not query or not chunks:
            return []

        encoder = self._load()

        if encoder is None:
            return [self._lexical_score(query, chunk) for chunk in chunks]

        pairs = [(query, chunk.text[:MAX_PAIR_CHARS]) for chunk in chunks]

        try:
            raw = encoder.predict(pairs, show_progress_bar=False)
            return [_squash(float(value)) for value in raw]
        except Exception as error:
            logger.warning("Cross encoder scoring failed, using the lexical reranker: %s", error)
            self._encoder = None
            return [self._lexical_score(query, chunk) for chunk in chunks]

    def _lexical_score(self, query: str, chunk: RetrievedChunk) -> float:
        body = lexical_similarity(query, chunk.text)
        heading = lexical_similarity(query, chunk.clause_type or chunk.section)
        importance = clause_importance(chunk.clause_type)

        return max(0.0, min(1.0, 0.6 * body + 0.25 * heading + 0.15 * importance))

    def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: Optional[int] = None,
    ) -> List[RetrievedChunk]:
        """Score every candidate, write ``rerank_score``, return the best ``top_k``.

        The chunks are copied rather than mutated so a reranked result never
        corrupts the scores held in the index.
        """

        if not chunks:
            return []

        scores = self.score(query, chunks)
        reranked: List[RetrievedChunk] = []

        for chunk, score in zip(chunks, scores):
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    contract_id=chunk.contract_id,
                    text=chunk.text,
                    clause_type=chunk.clause_type,
                    section=chunk.section,
                    page=chunk.page,
                    risk=chunk.risk,
                    risk_score=chunk.risk_score,
                    risk_factors=list(chunk.risk_factors),
                    entities=list(chunk.entities),
                    embedding=list(chunk.embedding),
                    dense_score=chunk.dense_score,
                    sparse_score=chunk.sparse_score,
                    rerank_score=score,
                    retrieval_score=chunk.retrieval_score,
                    metadata=dict(chunk.metadata),
                )
            )

        reranked.sort(key=lambda item: item.rerank_score, reverse=True)

        return reranked[:top_k] if top_k else reranked
