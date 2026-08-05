"""Kiran's ownership area: the clause-aware legal knowledge base and hybrid retrieval stack.

Scope (LegalEase v3 Hybrid RAG spec, section 15 - Kiran):
    * Clause chunking            -> clause_chunker
    * Metadata extraction        -> metadata_store
    * Embeddings                 -> embeddings
    * FAISS                      -> vector_index
    * BM25                       -> bm25_index
    * Hybrid retrieval           -> hybrid_retriever
    * Cross encoder reranker     -> reranker

The public entry point is :class:`KnowledgeBase`, which satisfies the
``rag.retrieval.ClauseRetriever`` protocol and can therefore be handed straight to
``rag.service.set_retriever`` so Venkata's answering layer consumes it unchanged.
"""

from .bm25_index import BM25Index
from .clause_chunker import ClauseChunker, segment_clauses
from .embeddings import EmbeddingBackend
from .hybrid_retriever import HybridRetriever, RetrievalSignals
from .knowledge_base import KnowledgeBase, build_knowledge_base
from .metadata_store import MetadataStore
from .reranker import CrossEncoderReranker
from .vector_index import VectorIndex

__all__ = [
    "BM25Index",
    "ClauseChunker",
    "CrossEncoderReranker",
    "EmbeddingBackend",
    "HybridRetriever",
    "KnowledgeBase",
    "MetadataStore",
    "RetrievalSignals",
    "VectorIndex",
    "build_knowledge_base",
    "segment_clauses",
]
