import os
from dataclasses import dataclass


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(_env_str(name, str(default))))
    except ValueError:
        return default


@dataclass(frozen=True)
class RagConfig:
    retriever_backend: str
    embedding_model: str
    reranker_model: str
    rerank_top_k: int
    bm25_k1: float
    bm25_b: float
    qa_model: str
    ner_model: str
    llm_provider: str
    llm_model: str
    llm_api_key: str
    llm_base_url: str
    llm_timeout: float
    llm_temperature: float
    llm_max_output_tokens: int
    candidate_pool_size: int
    top_k_context: int
    max_context_chars: int
    faithfulness_threshold: float
    low_confidence_threshold: float
    min_answer_confidence: float


DEFAULT_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta",
    "ollama": "http://localhost:11434",
}


RETRIEVER_BACKENDS = {"hybrid", "memory"}


def load_config() -> RagConfig:
    provider = _env_str("LEGALEASE_LLM_PROVIDER", "none").lower()
    base_url = _env_str("LEGALEASE_LLM_BASE_URL", DEFAULT_BASE_URLS.get(provider, ""))
    backend = _env_str("LEGALEASE_RETRIEVER", "hybrid").lower()

    if backend not in RETRIEVER_BACKENDS:
        backend = "hybrid"

    return RagConfig(
        retriever_backend=backend,
        embedding_model=_env_str("LEGALEASE_EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5"),
        reranker_model=_env_str("LEGALEASE_RERANKER_MODEL", "BAAI/bge-reranker-large"),
        rerank_top_k=_env_int("LEGALEASE_RERANK_TOP_K", 5),
        bm25_k1=_env_float("LEGALEASE_BM25_K1", 1.5),
        bm25_b=_env_float("LEGALEASE_BM25_B", 0.75),
        qa_model=_env_str("LEGALEASE_QA_MODEL", "deepset/deberta-v3-base-squad2"),
        ner_model=_env_str("LEGALEASE_NER_MODEL", "nlpaueb/legal-bert-base-uncased"),
        llm_provider=provider,
        llm_model=_env_str("LEGALEASE_LLM_MODEL", "gpt-4.1-mini"),
        llm_api_key=_env_str("LEGALEASE_LLM_API_KEY", ""),
        llm_base_url=base_url,
        llm_timeout=_env_float("LEGALEASE_LLM_TIMEOUT", 60.0),
        llm_temperature=_env_float("LEGALEASE_LLM_TEMPERATURE", 0.0),
        llm_max_output_tokens=_env_int("LEGALEASE_LLM_MAX_TOKENS", 700),
        candidate_pool_size=_env_int("LEGALEASE_CANDIDATE_POOL", 25),
        top_k_context=_env_int("LEGALEASE_TOP_K", 5),
        max_context_chars=_env_int("LEGALEASE_MAX_CONTEXT_CHARS", 9000),
        faithfulness_threshold=_env_float("LEGALEASE_FAITHFULNESS_THRESHOLD", 0.55),
        low_confidence_threshold=_env_float("LEGALEASE_LOW_CONFIDENCE_THRESHOLD", 0.45),
        min_answer_confidence=_env_float("LEGALEASE_MIN_ANSWER_CONFIDENCE", 0.15),
    )
