import logging
import math
import re
from typing import Iterable, List, Optional, Sequence

logger = logging.getLogger(__name__)

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?;])\s+")

_STOPWORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "by", "for", "from", "has",
    "have", "in", "is", "it", "its", "of", "on", "or", "shall", "that", "the",
    "this", "to", "was", "were", "will", "with", "such", "which", "may",
}

_encoder = None
_encoder_failed = False
_encoder_name = ""


_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ied", "ers", "er", "ed", "es", "s")


def _stem(token: str) -> str:
    for suffix in _SUFFIXES:
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> List[str]:
    return [
        _stem(token)
        for token in _TOKEN_PATTERN.findall(text.lower())
        if token not in _STOPWORDS
    ]


def split_sentences(text: str, min_words: int = 3) -> List[str]:
    parts = [p.strip() for p in _SENTENCE_PATTERN.split(text or "") if p.strip()]
    return [p for p in parts if len(p.split()) >= min_words]


def lexical_similarity(left: str, right: str) -> float:
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))

    if not left_tokens or not right_tokens:
        return 0.0

    overlap = len(left_tokens & right_tokens)
    return overlap / math.sqrt(len(left_tokens) * len(right_tokens))


def load_encoder(model_name: str):
    global _encoder, _encoder_failed, _encoder_name

    if _encoder is not None and _encoder_name == model_name:
        return _encoder

    if _encoder_failed and _encoder_name == model_name:
        return None

    try:
        from sentence_transformers import SentenceTransformer

        _encoder = SentenceTransformer(model_name)
        _encoder_name = model_name
        _encoder_failed = False
    except Exception as error:
        logger.warning("Embedding model unavailable, using lexical fallback: %s", error)
        _encoder = None
        _encoder_name = model_name
        _encoder_failed = True

    return _encoder


def encoder_available() -> bool:
    return _encoder is not None


def similarity(left: str, right: str, model_name: Optional[str] = None) -> float:
    if not left or not right:
        return 0.0

    encoder = load_encoder(model_name) if model_name else _encoder

    if encoder is None:
        return lexical_similarity(left, right)

    try:
        from sentence_transformers import util

        embeddings = encoder.encode([left, right], convert_to_tensor=True, show_progress_bar=False)
        score = float(util.cos_sim(embeddings[0], embeddings[1]).item())
        return max(0.0, min(1.0, score))
    except Exception as error:
        logger.warning("Embedding similarity failed, using lexical fallback: %s", error)
        return lexical_similarity(left, right)


def max_similarity(text: str, candidates: Sequence[str], model_name: Optional[str] = None) -> float:
    if not text or not candidates:
        return 0.0

    encoder = load_encoder(model_name) if model_name else _encoder

    if encoder is None:
        return max(lexical_similarity(text, candidate) for candidate in candidates)

    try:
        from sentence_transformers import util

        query_embedding = encoder.encode(text, convert_to_tensor=True, show_progress_bar=False)
        candidate_embeddings = encoder.encode(
            list(candidates), convert_to_tensor=True, show_progress_bar=False
        )
        score = float(util.cos_sim(query_embedding, candidate_embeddings).max().item())
        return max(0.0, min(1.0, score))
    except Exception as error:
        logger.warning("Embedding batch similarity failed, using lexical fallback: %s", error)
        return max(lexical_similarity(text, candidate) for candidate in candidates)


def average(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        return 0.0
    return sum(items) / len(items)
