from typing import List, Optional, Sequence

from . import semantic
from .schemas import FaithfulnessReport, RetrievedChunk

REFUSAL_MARKER = "does not answer this question"
MIN_STATEMENT_WORDS = 4


def _statements(answer: str) -> List[str]:
    sentences = semantic.split_sentences(answer, min_words=MIN_STATEMENT_WORDS)
    if sentences:
        return sentences
    cleaned = (answer or "").strip()
    return [cleaned] if cleaned else []


def verify_answer(
    answer: str,
    evidence: Sequence[RetrievedChunk],
    threshold: float,
    embedding_model: Optional[str] = None,
) -> FaithfulnessReport:
    answer = (answer or "").strip()

    if not answer:
        return FaithfulnessReport(
            score=0.0,
            supported=False,
            threshold=threshold,
            unsupported_statements=[],
            statement_scores=[],
        )

    if REFUSAL_MARKER in answer.lower():
        return FaithfulnessReport(
            score=1.0,
            supported=True,
            threshold=threshold,
            unsupported_statements=[],
            statement_scores=[1.0],
        )

    if not evidence:
        return FaithfulnessReport(
            score=0.0,
            supported=False,
            threshold=threshold,
            unsupported_statements=_statements(answer),
            statement_scores=[],
        )

    evidence_texts = [chunk.text for chunk in evidence if chunk.text.strip()]
    statements = _statements(answer)
    scores: List[float] = []
    unsupported: List[str] = []

    for statement in statements:
        score = semantic.max_similarity(statement, evidence_texts, embedding_model)
        scores.append(score)
        if score < threshold:
            unsupported.append(statement)

    overall = semantic.average(scores)
    supported = bool(scores) and overall >= threshold and len(unsupported) <= len(scores) // 2

    return FaithfulnessReport(
        score=overall,
        supported=supported,
        threshold=threshold,
        unsupported_statements=unsupported,
        statement_scores=scores,
    )
