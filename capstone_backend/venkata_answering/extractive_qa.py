import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

from rag import semantic
from .models import get_qa_pipeline
from rag.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

MAX_ANSWER_LENGTH = 220
MIN_ANSWER_WORDS = 2


@dataclass
class ExtractiveAnswer:
    text: str
    score: float
    chunk: Optional[RetrievedChunk]
    model_used: bool


def _fallback_answer(question: str, chunks: Sequence[RetrievedChunk]) -> ExtractiveAnswer:
    best_text = ""
    best_score = 0.0
    best_chunk: Optional[RetrievedChunk] = None

    for chunk in chunks:
        for sentence in semantic.split_sentences(chunk.text) or [chunk.text]:
            score = semantic.lexical_similarity(question, sentence)
            if score > best_score:
                best_score = score
                best_text = sentence.strip()
                best_chunk = chunk

    if not best_text and chunks:
        top = chunks[0]
        sentences = semantic.split_sentences(top.text) or [top.text]
        best_text = sentences[0].strip()
        best_score = 0.4 * top.retrieval_score
        best_chunk = top

    return ExtractiveAnswer(
        text=best_text,
        score=round(best_score, 4),
        chunk=best_chunk,
        model_used=False,
    )


def answer_from_chunks(
    question: str,
    chunks: Sequence[RetrievedChunk],
    qa_model: str,
) -> ExtractiveAnswer:
    if not question or not chunks:
        return ExtractiveAnswer(text="", score=0.0, chunk=None, model_used=False)

    pipeline = get_qa_pipeline(qa_model)

    if pipeline is None:
        return _fallback_answer(question, chunks)

    candidates: List[ExtractiveAnswer] = []

    for chunk in chunks:
        try:
            result = pipeline(
                question=question,
                context=chunk.text,
                max_answer_len=MAX_ANSWER_LENGTH,
                handle_impossible_answer=True,
            )
        except Exception as error:
            logger.warning("Extractive QA failed on chunk %s: %s", chunk.chunk_id, error)
            continue

        text = str(result.get("answer", "")).strip()
        score = float(result.get("score", 0.0) or 0.0)

        if not text or len(text.split()) < MIN_ANSWER_WORDS:
            continue

        combined = 0.7 * score + 0.3 * chunk.retrieval_score
        candidates.append(
            ExtractiveAnswer(text=text, score=round(combined, 4), chunk=chunk, model_used=True)
        )

    if not candidates:
        return _fallback_answer(question, chunks)

    return max(candidates, key=lambda item: item.score)
