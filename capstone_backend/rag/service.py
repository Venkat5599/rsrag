import threading
from typing import Any, Dict, Optional, Sequence

from .answer_engine import AnswerEngine
from .config import RagConfig, load_config
from .retrieval import ClauseRetriever, InMemoryClauseRetriever, chunks_from_clause_map, chunks_from_text
from .schemas import RetrievedChunk

_lock = threading.RLock()
_config: Optional[RagConfig] = None
_retriever: Optional[ClauseRetriever] = None
_engine: Optional[AnswerEngine] = None
_indexed_contracts = set()


def get_config() -> RagConfig:
    global _config

    with _lock:
        if _config is None:
            _config = load_config()
        return _config


def get_retriever() -> ClauseRetriever:
    global _retriever

    embedding_model = get_config().embedding_model

    with _lock:
        if _retriever is None:
            _retriever = InMemoryClauseRetriever(embedding_model=embedding_model)
        return _retriever


def set_retriever(retriever: ClauseRetriever) -> None:
    global _retriever, _engine

    with _lock:
        _retriever = retriever
        _engine = None


def get_engine() -> AnswerEngine:
    global _engine

    retriever = get_retriever()
    config = get_config()

    with _lock:
        if _engine is None:
            _engine = AnswerEngine(retriever, config)
        return _engine


def is_indexed(contract_id: str) -> bool:
    return contract_id in _indexed_contracts


def index_chunks(contract_id: str, chunks: Sequence[RetrievedChunk]) -> int:
    retriever = get_retriever()

    if not chunks or not hasattr(retriever, "add"):
        return 0

    retriever.add(list(chunks))
    _indexed_contracts.add(contract_id)
    return len(chunks)


def index_document(document: Dict[str, Any]) -> int:
    contract_id = str(document.get("docId") or document.get("id") or "").strip()

    if not contract_id:
        return 0

    filename = str(document.get("filename", ""))
    chunks = chunks_from_clause_map(document.get("clauses") or {}, contract_id, filename)

    if not chunks:
        chunks = chunks_from_text(str(document.get("summary", "")), contract_id, filename)

    return index_chunks(contract_id, chunks)


def reset() -> None:
    global _config, _retriever, _engine, _indexed_contracts

    with _lock:
        _config = None
        _retriever = None
        _engine = None
        _indexed_contracts = set()
