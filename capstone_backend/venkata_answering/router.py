from typing import Optional

from rag.clause_taxonomy import match_clause_types
from .query_classifier import classify_query
from rag.schemas import AnswerStrategy, QueryIntent, QueryPlan

INTENT_STRATEGY = {
    QueryIntent.FACT_LOOKUP: AnswerStrategy.EXTRACTIVE_QA,
    QueryIntent.CLAUSE_RETRIEVAL: AnswerStrategy.RETRIEVER_ONLY,
    QueryIntent.ENTITY_LOOKUP: AnswerStrategy.ENTITY_EXTRACTION,
    QueryIntent.CLAUSE_COMPARISON: AnswerStrategy.GROUNDED_LLM,
    QueryIntent.SUMMARIZATION: AnswerStrategy.GROUNDED_LLM,
    QueryIntent.RISK_EXPLANATION: AnswerStrategy.GROUNDED_LLM,
    QueryIntent.LEGAL_REASONING: AnswerStrategy.GROUNDED_LLM,
}

STRATEGY_PRIOR = {
    AnswerStrategy.EXTRACTIVE_QA: 0.85,
    AnswerStrategy.RETRIEVER_ONLY: 0.9,
    AnswerStrategy.ENTITY_EXTRACTION: 0.8,
    AnswerStrategy.GROUNDED_LLM: 0.75,
}

LLM_FALLBACK = {
    QueryIntent.CLAUSE_COMPARISON: AnswerStrategy.RETRIEVER_ONLY,
    QueryIntent.SUMMARIZATION: AnswerStrategy.RETRIEVER_ONLY,
    QueryIntent.RISK_EXPLANATION: AnswerStrategy.RETRIEVER_ONLY,
    QueryIntent.LEGAL_REASONING: AnswerStrategy.EXTRACTIVE_QA,
}

AMBIGUITY_THRESHOLD = 0.25


def build_plan(
    query: str,
    embedding_model: Optional[str] = None,
    llm_available: bool = True,
) -> QueryPlan:
    intent, confidence, scores = classify_query(query, embedding_model)
    strategy = INTENT_STRATEGY[intent]
    reason = f"intent {intent.value} routed to {strategy.value}"

    if confidence < AMBIGUITY_THRESHOLD and strategy is AnswerStrategy.GROUNDED_LLM:
        strategy = AnswerStrategy.EXTRACTIVE_QA
        reason = "low intent confidence, downgraded to extractive answering"

    if strategy is AnswerStrategy.GROUNDED_LLM and not llm_available:
        strategy = LLM_FALLBACK.get(intent, AnswerStrategy.RETRIEVER_ONLY)
        reason = "grounded llm unavailable, degraded to evidence-only answering"

    return QueryPlan(
        query=query,
        intent=intent,
        intent_confidence=confidence,
        strategy=strategy,
        clause_filters=match_clause_types(query),
        intent_scores=scores,
        routing_reason=reason,
    )


def strategy_prior(strategy: AnswerStrategy) -> float:
    return STRATEGY_PRIOR.get(strategy, 0.7)
