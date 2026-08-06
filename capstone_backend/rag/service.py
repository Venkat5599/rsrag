import logging
import threading
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from .config import RagConfig, load_config
from .retrieval import ClauseRetriever, InMemoryClauseRetriever, chunks_from_clause_map
from .schemas import RetrievedChunk

if TYPE_CHECKING:  # pragma: no cover - typing only; the runtime import stays lazy
    from venkata_answering.answer_engine import AnswerEngine

logger = logging.getLogger(__name__)

_lock = threading.RLock()
_config: Optional[RagConfig] = None
_retriever: Optional[ClauseRetriever] = None
_engine: Optional["AnswerEngine"] = None
_indexed_contracts = set()


def get_config() -> RagConfig:
    global _config

    with _lock:
        if _config is None:
            _config = load_config()
        return _config


def _build_retriever(config: RagConfig) -> ClauseRetriever:
    """Select the retrieval backend.

    ``hybrid`` (default) is Kiran's clause-aware knowledge base: metadata filtering,
    dense plus BM25 candidate generation, and cross-encoder reranking. ``memory`` is
    the original single-pass lexical retriever, kept as the benchmark baseline.
    """

    if config.retriever_backend == "memory":
        return InMemoryClauseRetriever(embedding_model=config.embedding_model)

    try:
        from kiran_retrieval.knowledge_base import build_knowledge_base

        return build_knowledge_base(config)
    except Exception as error:
        logger.error(
            "Hybrid knowledge base unavailable, falling back to the baseline retriever: %s",
            error,
        )
        return InMemoryClauseRetriever(embedding_model=config.embedding_model)


def get_retriever() -> ClauseRetriever:
    global _retriever

    config = get_config()

    with _lock:
        if _retriever is None:
            _retriever = _build_retriever(config)
        return _retriever


def set_retriever(retriever: ClauseRetriever) -> None:
    global _retriever, _engine

    with _lock:
        _retriever = retriever
        _engine = None


def get_engine() -> "AnswerEngine":
    global _engine

    from venkata_answering.answer_engine import AnswerEngine

    retriever = get_retriever()
    config = get_config()

    with _lock:
        if _engine is None:
            _engine = AnswerEngine(retriever, config)
        return _engine


def is_indexed(contract_id: str) -> bool:
    return contract_id in _indexed_contracts


def _annotate_risk(chunks: Sequence[RetrievedChunk]) -> None:
    """Fill the section 7 ``risk`` field before indexing.

    The KnowledgeBase does this itself, but the baseline in-memory retriever does
    not, and the fallback path is exactly where an empty ``risk`` would otherwise
    reappear. Imported locally because rag/ is the shared layer and must not depend
    on kiran_retrieval/ at module import time.
    """

    try:
        from kiran_retrieval import risk_classifier
    except Exception:  # pragma: no cover - retrieval package unavailable
        return

    risk_classifier.annotate_many(chunks)


def index_chunks(contract_id: str, chunks: Sequence[RetrievedChunk]) -> int:
    retriever = get_retriever()

    if not chunks or not hasattr(retriever, "add"):
        return 0

    _annotate_risk(chunks)

    retriever.add(list(chunks))
    _indexed_contracts.add(contract_id)
    return len(chunks)


def _chunks_from_free_text(text: str, contract_id: str, filename: str) -> Sequence[RetrievedChunk]:
    """Clause-aware chunking, with the fixed-window builder as the fallback."""

    try:
        from kiran_retrieval.clause_chunker import segment_clauses

        chunks = segment_clauses(text, contract_id, filename)
        if chunks:
            return chunks
    except Exception as error:
        logger.warning("Clause chunking unavailable, using fixed windows: %s", error)

    from .retrieval import chunks_from_text

    return chunks_from_text(text, contract_id, filename)


def index_document(document: Dict[str, Any]) -> int:
    contract_id = str(document.get("docId") or document.get("id") or "").strip()

    if not contract_id:
        return 0

    filename = str(document.get("filename", ""))
    chunks = chunks_from_clause_map(document.get("clauses") or {}, contract_id, filename)

    if not chunks:
        chunks = _chunks_from_free_text(
            str(document.get("summary", "")), contract_id, filename
        )

    return index_chunks(contract_id, chunks)


def reset() -> None:
    global _config, _retriever, _engine, _indexed_contracts

    with _lock:
        _config = None
        _retriever = None
        _engine = None
        _indexed_contracts = set()
