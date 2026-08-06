"""Hybrid retrieval.

Spec sections 8, 9 and 10 in one pipeline:

    query
      -> metadata filter          (MetadataStore.candidates)
      -> semantic search          (VectorIndex, dense)
      -> BM25 search              (BM25Index, sparse)
      -> merged candidate pool    (config.candidate_pool_size, default 25)
      -> cross encoder reranking  (CrossEncoderReranker)
      -> top context              (config.top_k_context, default 5)

Section 9 is the part that is a research contribution rather than plumbing: the
final ranking is not cosine similarity alone but a weighted fusion of six signals.
The weights live in :class:`RetrievalSignals` so an experiment can sweep them and
report the result, and every component score is written back onto the returned
chunk so a ranking can be explained after the fact.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from rag.clause_taxonomy import clause_importance
from rag.schemas import RetrievedChunk
from rag.semantic import lexical_similarity

from .bm25_index import BM25Index
from .embeddings import EmbeddingBackend
from .metadata_store import MetadataStore
from .reranker import CrossEncoderReranker
from .vector_index import VectorIndex


@dataclass(frozen=True)
class RetrievalSignals:
    """Fusion weights for the section 9 ranking function.

    Defaults put the majority of the mass on the cross encoder and the dense score,
    because those are the two signals that actually read the clause. Heading, entity
    and clause-importance are tie-breakers: they are what separate two clauses that
    say similar things but sit in different parts of the contract.
    """

    semantic: float = 0.28
    bm25: float = 0.18
    cross_encoder: float = 0.32
    clause_importance: float = 0.08
    heading_match: float = 0.08
    entity_match: float = 0.06

    def total(self) -> float:
        return (
            self.semantic
            + self.bm25
            + self.cross_encoder
            + self.clause_importance
            + self.heading_match
            + self.entity_match
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "semantic": self.semantic,
            "bm25": self.bm25,
            "cross_encoder": self.cross_encoder,
            "clause_importance": self.clause_importance,
            "heading_match": self.heading_match,
            "entity_match": self.entity_match,
        }


def _entity_match(query: str, chunk: RetrievedChunk) -> float:
    if not chunk.entities:
        return 0.0

    lowered = query.lower()
    hits = sum(1 for entity in chunk.entities if entity.text.lower() in lowered)

    return min(hits / 2.0, 1.0)


def _allocate_across_contracts(
    chunks: Sequence[RetrievedChunk],
    contract_ids: Sequence[str],
    limit: int,
) -> List[RetrievedChunk]:
    """Interleave results so every contract in scope is actually represented.

    Taking the global top-N across two contracts is the failure mode this exists to
    prevent: whichever agreement scores higher fills every slot, and the answer then
    reads as a comparison while being sourced from one side. Round-robin over the
    per-contract rankings guarantees the losing contract still contributes evidence,
    and any unused slots fall back to global order so a contract with few matching
    clauses does not waste the budget.
    """

    if limit <= 0 or not chunks:
        return []

    buckets: Dict[str, List[RetrievedChunk]] = {name: [] for name in contract_ids}

    for chunk in chunks:
        buckets.setdefault(chunk.contract_id, []).append(chunk)

    ordered_names = [name for name in contract_ids if buckets.get(name)]
    ordered_names += [
        name for name in buckets if name not in contract_ids and buckets[name]
    ]

    if not ordered_names:
        return list(chunks)[:limit]

    selected: List[RetrievedChunk] = []
    seen: set = set()
    depth = 0
    deepest = max(len(bucket) for bucket in buckets.values())

    while len(selected) < limit and depth < deepest:
        for name in ordered_names:
            bucket = buckets.get(name) or []
            if depth < len(bucket) and len(selected) < limit:
                chunk = bucket[depth]
                selected.append(chunk)
                seen.add(chunk.chunk_id)
        depth += 1

    for chunk in chunks:
        if len(selected) >= limit:
            break
        if chunk.chunk_id not in seen:
            selected.append(chunk)
            seen.add(chunk.chunk_id)

    return selected


class HybridRetriever:
    """Implements the ``rag.retrieval.ClauseRetriever`` protocol."""

    def __init__(
        self,
        store: MetadataStore,
        vectors: VectorIndex,
        bm25: BM25Index,
        embeddings: EmbeddingBackend,
        reranker: CrossEncoderReranker,
        candidate_pool_size: int = 25,
        top_k_context: int = 5,
        signals: Optional[RetrievalSignals] = None,
    ) -> None:
        self._store = store
        self._vectors = vectors
        self._bm25 = bm25
        self._embeddings = embeddings
        self._reranker = reranker
        self._pool_size = max(candidate_pool_size, 1)
        self._top_k = max(top_k_context, 1)
        self._signals = signals or RetrievalSignals()

    @property
    def signals(self) -> RetrievalSignals:
        return self._signals

    def describe(self) -> Dict[str, object]:
        return {
            "embedding_backend": self._embeddings.backend_name(),
            "vector_backend": self._vectors.backend,
            "reranker_backend": self._reranker.backend_name(),
            "candidate_pool_size": self._pool_size,
            "top_k_context": self._top_k,
            "signal_weights": self._signals.to_dict(),
            "indexed_chunks": len(self._store),
        }

    def candidates(
        self,
        query: str,
        clause_filters: Optional[Sequence[str]] = None,
        contract_id: Optional[str] = None,
        contract_ids: Optional[Sequence[str]] = None,
    ) -> List[RetrievedChunk]:
        """Stages one to three: metadata filter, dense search, BM25 search, merge."""

        scope_names = self._store.scope_names(contract_id, contract_ids)

        allowed = self._store.candidates(
            clause_filters,
            contract_id,
            min_candidates=self._pool_size,
            contract_ids=contract_ids,
        )

        if not allowed:
            return []

        # A comparison spanning N contracts needs a pool deep enough that every
        # contract is represented. Widening the pool per contract is what stops the
        # higher-scoring agreement from filling all 25 slots on its own.
        pool_size = self._pool_size * max(len(scope_names), 1)

        dense_hits = self._vectors.search(
            self._embeddings.encode_one(query), pool_size, allowed_ids=allowed
        )
        sparse_hits = self._bm25.search(query, pool_size, allowed_ids=allowed)

        dense_scores = {chunk_id: score for chunk_id, score in dense_hits}
        sparse_scores = {chunk_id: normalised for chunk_id, _, normalised in sparse_hits}

        ordered_ids = list(dense_scores) + [
            chunk_id for chunk_id in sparse_scores if chunk_id not in dense_scores
        ]

        merged: List[RetrievedChunk] = []

        for chunk_id in ordered_ids:
            source = self._store.chunk(chunk_id)

            if source is None:
                continue

            merged.append(
                RetrievedChunk(
                    chunk_id=source.chunk_id,
                    contract_id=source.contract_id,
                    text=source.text,
                    clause_type=source.clause_type,
                    section=source.section,
                    page=source.page,
                    risk=source.risk,
                    risk_score=source.risk_score,
                    risk_factors=list(source.risk_factors),
                    entities=list(source.entities),
                    embedding=list(source.embedding),
                    dense_score=max(0.0, min(1.0, dense_scores.get(chunk_id, 0.0))),
                    sparse_score=sparse_scores.get(chunk_id, 0.0),
                    metadata=dict(source.metadata),
                )
            )

        merged.sort(
            key=lambda chunk: 0.6 * chunk.dense_score + 0.4 * chunk.sparse_score,
            reverse=True,
        )

        if len(scope_names) > 1:
            return _allocate_across_contracts(merged, scope_names, self._pool_size)

        return merged[: self._pool_size]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        clause_filters: Optional[Sequence[str]] = None,
        contract_id: Optional[str] = None,
        contract_ids: Optional[Sequence[str]] = None,
    ) -> List[RetrievedChunk]:
        query = (query or "").strip()

        if not query or not len(self._store):
            return []

        scope_names = self._store.scope_names(contract_id, contract_ids)
        pool = self.candidates(query, clause_filters, contract_id, contract_ids)

        if not pool:
            return []

        reranked = self._reranker.rerank(query, pool)
        filters = {value.lower() for value in (clause_filters or [])}
        weights = self._signals
        divisor = weights.total() or 1.0

        for chunk in reranked:
            heading = lexical_similarity(query, chunk.clause_type or chunk.section)
            entity = _entity_match(query, chunk)
            importance = clause_importance(chunk.clause_type)

            fused = (
                weights.semantic * chunk.dense_score
                + weights.bm25 * chunk.sparse_score
                + weights.cross_encoder * chunk.rerank_score
                + weights.clause_importance * importance
                + weights.heading_match * heading
                + weights.entity_match * entity
            ) / divisor

            if chunk.clause_type and chunk.clause_type.lower() in filters:
                fused = min(fused + 0.05, 1.0)

            chunk.retrieval_score = max(0.0, min(1.0, fused))
            chunk.metadata = dict(chunk.metadata)
            chunk.metadata["signals"] = {
                "semantic": round(chunk.dense_score, 4),
                "bm25": round(chunk.sparse_score, 4),
                "cross_encoder": round(chunk.rerank_score, 4),
                "clause_importance": round(importance, 4),
                "heading_match": round(heading, 4),
                "entity_match": round(entity, 4),
            }

        reranked.sort(key=lambda chunk: chunk.retrieval_score, reverse=True)

        limit = max(top_k or self._top_k, 1)

        if len(scope_names) > 1:
            return _allocate_across_contracts(reranked, scope_names, limit)

        return reranked[:limit]
