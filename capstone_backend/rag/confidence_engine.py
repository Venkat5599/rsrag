from typing import Dict, List, Optional, Sequence

from . import semantic
from .router import strategy_prior
from .schemas import (
    AnswerStrategy,
    Citation,
    ConfidenceReport,
    FaithfulnessReport,
    QueryPlan,
    RetrievedChunk,
)

SIGNAL_WEIGHTS: Dict[str, float] = {
    "retrieval": 0.24,
    "answer_model": 0.22,
    "faithfulness": 0.26,
    "citation_support": 0.14,
    "intent": 0.08,
    "strategy_prior": 0.06,
}

HIGH_BAND = 0.7
MEDIUM_BAND = 0.45


def _retrieval_signal(evidence: Sequence[RetrievedChunk]) -> float:
    if not evidence:
        return 0.0

    top = evidence[0].retrieval_score
    mean = semantic.average(chunk.retrieval_score for chunk in evidence)
    coverage = min(len(evidence) / 3.0, 1.0)

    return max(0.0, min(1.0, 0.5 * top + 0.3 * mean + 0.2 * coverage))


def _citation_signal(citations: Sequence[Citation]) -> float:
    if not citations:
        return 0.0

    top = citations[0].support_score
    mean = semantic.average(citation.support_score for citation in citations)
    return max(0.0, min(1.0, 0.6 * top + 0.4 * mean))


def band_for(score: float) -> str:
    if score >= HIGH_BAND:
        return "high"
    if score >= MEDIUM_BAND:
        return "medium"
    return "low"


def score_answer(
    plan: QueryPlan,
    evidence: Sequence[RetrievedChunk],
    citations: Sequence[Citation],
    faithfulness: Optional[FaithfulnessReport],
    answer_model_score: float,
    penalties: Optional[List[float]] = None,
) -> ConfidenceReport:
    signals: Dict[str, float] = {
        "retrieval": _retrieval_signal(evidence),
        "answer_model": max(0.0, min(1.0, answer_model_score)),
        "faithfulness": faithfulness.score if faithfulness else 0.0,
        "citation_support": _citation_signal(citations),
        "intent": max(0.0, min(1.0, plan.intent_confidence)),
        "strategy_prior": strategy_prior(plan.strategy),
    }

    score = sum(SIGNAL_WEIGHTS[name] * value for name, value in signals.items())

    if faithfulness is not None and not faithfulness.supported:
        score *= 0.55

    if plan.strategy is AnswerStrategy.GROUNDED_LLM and not citations:
        score *= 0.6

    for penalty in penalties or []:
        score *= max(0.0, min(1.0, penalty))

    score = max(0.0, min(1.0, score))

    return ConfidenceReport(score=score, band=band_for(score), signals=signals)
