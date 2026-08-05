"""The clause-aware legal knowledge base.

This is the left-hand column of the spec section 4 architecture:

    contract text
      -> clause segmentation      (clause_chunker)
      -> metadata extraction      (metadata_store)
      -> embedding generation     (embeddings)
      -> FAISS index + BM25 index (vector_index, bm25_index)

and the entry point for the right-hand column, since :meth:`KnowledgeBase.retrieve`
satisfies the ``rag.retrieval.ClauseRetriever`` protocol. That is the whole contract
between Kiran's half and Venkata's half: one method, already defined in shared code.
"""

import logging
from typing import Any, Dict, List, Optional, Sequence

from rag.config import RagConfig, load_config
from rag.retrieval import chunks_from_clause_map
from rag.schemas import RetrievedChunk

from .bm25_index import BM25Index
from .clause_chunker import ClauseChunker
from .embeddings import EmbeddingBackend
from .hybrid_retriever import HybridRetriever, RetrievalSignals
from .metadata_store import MetadataStore
from .reranker import CrossEncoderReranker
from .vector_index import VectorIndex

logger = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(
        self,
        config: Optional[RagConfig] = None,
        signals: Optional[RetrievalSignals] = None,
    ) -> None:
        self._config = config or load_config()

        self._chunker = ClauseChunker()
        self._store = MetadataStore()
        self._embeddings = EmbeddingBackend(self._config.embedding_model)
        self._vectors = VectorIndex(self._embeddings.dimensions)
        self._bm25 = BM25Index(k1=self._config.bm25_k1, b=self._config.bm25_b)
        self._reranker = CrossEncoderReranker(self._config.reranker_model)

        self._retriever = HybridRetriever(
            store=self._store,
            vectors=self._vectors,
            bm25=self._bm25,
            embeddings=self._embeddings,
            reranker=self._reranker,
            candidate_pool_size=self._config.candidate_pool_size,
            top_k_context=self._config.top_k_context,
            signals=signals,
        )

    @property
    def config(self) -> RagConfig:
        return self._config

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def store(self) -> MetadataStore:
        return self._store

    @property
    def retriever(self) -> HybridRetriever:
        return self._retriever

    def describe(self) -> Dict[str, Any]:
        description = self._retriever.describe()
        description.update(self._store.stats())
        return description

    def index_chunks(self, chunks: Sequence[RetrievedChunk]) -> int:
        """Add pre-built chunks to the metadata store and both indexes."""

        accepted = [chunk for chunk in chunks if chunk.text and chunk.text.strip()]

        if not accepted:
            return 0

        self._store.add_many(accepted)

        vectors = self._embeddings.encode([chunk.text for chunk in accepted])
        self._vectors.add([chunk.chunk_id for chunk in accepted], vectors)

        for chunk in accepted:
            self._bm25.add(chunk.chunk_id, chunk.text)

        return len(accepted)

    # ``add`` keeps the KnowledgeBase interchangeable with InMemoryClauseRetriever,
    # which rag.service calls through hasattr(retriever, "add").
    def add(self, chunks: Sequence[RetrievedChunk]) -> None:
        self.index_chunks(chunks)

    def index_text(
        self,
        text: str,
        contract_id: Optional[str] = None,
        filename: str = "",
    ) -> int:
        return self.index_chunks(self._chunker.chunk(text, contract_id, filename))

    def index_document(self, document: Dict[str, Any]) -> int:
        """Index a stored analysis document.

        A v3 pipeline document already carries an extracted clause map, which is
        better structured than anything re-segmentation could recover, so it is used
        directly. Only when that map is absent does the raw summary text go through
        the clause chunker.
        """

        contract_id = str(document.get("docId") or document.get("id") or "").strip()

        if not contract_id:
            return 0

        filename = str(document.get("filename", ""))
        chunks = chunks_from_clause_map(document.get("clauses") or {}, contract_id, filename)

        if not chunks:
            chunks = self._chunker.chunk(
                str(document.get("summary", "")), contract_id, filename
            )

        return self.index_chunks(chunks)

    def clear(self) -> None:
        self._store.clear()
        self._vectors.clear()
        self._bm25.clear()
        self._embeddings.clear_cache()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        clause_filters: Optional[Sequence[str]] = None,
        contract_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        return self._retriever.retrieve(
            query, top_k=top_k, clause_filters=clause_filters, contract_id=contract_id
        )


def build_knowledge_base(
    config: Optional[RagConfig] = None,
    signals: Optional[RetrievalSignals] = None,
) -> KnowledgeBase:
    return KnowledgeBase(config=config, signals=signals)
