"""Shared core for LegalEase v3.

This package holds only what BOTH owner packages depend on:

    schemas          - the chunk / plan / citation / answer data contracts
    config           - environment-driven runtime configuration
    semantic         - tokenisation, sentence splitting, similarity primitives
    clause_taxonomy  - clause vocabulary, importance weights, entity aliases
    retrieval        - the ClauseRetriever protocol, chunk builders, baseline retriever
    service          - process-wide wiring that selects a retriever and an engine
    evaluation       - end-to-end harness that exercises both halves together

Ownership of everything else lives in the owner packages:

    kiran_retrieval/    knowledge base, embeddings, FAISS, BM25, hybrid retrieval, reranker
    venkata_answering/  query classifier, routing, DeBERTa, LegalBERT, grounded LLM,
                        prompt builder, citations, confidence, faithfulness

Nothing in this package imports from an owner package at module level. ``service``
imports them lazily inside its factory functions, which keeps the dependency arrow
one-way and avoids a partial-import cycle.
"""

from .clause_taxonomy import clause_importance, match_clause_types
from .config import RagConfig, load_config
from .retrieval import (
    ClauseRetriever,
    InMemoryClauseRetriever,
    chunks_from_clause_map,
    chunks_from_text,
)
from .schemas import (
    AnswerResult,
    AnswerStrategy,
    Citation,
    ConfidenceReport,
    Entity,
    FaithfulnessReport,
    QueryIntent,
    QueryPlan,
    RetrievedChunk,
)

__all__ = [
    "AnswerResult",
    "AnswerStrategy",
    "Citation",
    "ClauseRetriever",
    "ConfidenceReport",
    "Entity",
    "FaithfulnessReport",
    "InMemoryClauseRetriever",
    "QueryIntent",
    "QueryPlan",
    "RagConfig",
    "RetrievedChunk",
    "chunks_from_clause_map",
    "chunks_from_text",
    "clause_importance",
    "load_config",
    "match_clause_types",
]
