from rag.router import build_plan, strategy_prior
from rag.schemas import AnswerStrategy, QueryIntent


def test_fact_lookup_routes_to_extractive_qa():
    plan = build_plan("What is the notice period for termination?")

    assert plan.intent is QueryIntent.FACT_LOOKUP
    assert plan.strategy is AnswerStrategy.EXTRACTIVE_QA


def test_reasoning_routes_to_grounded_llm_when_available():
    plan = build_plan("What happens if the supplier misses the deadline?", llm_available=True)

    assert plan.strategy is AnswerStrategy.GROUNDED_LLM


def test_reasoning_degrades_when_llm_missing():
    plan = build_plan("What happens if the supplier misses the deadline?", llm_available=False)

    assert plan.strategy is not AnswerStrategy.GROUNDED_LLM
    assert "degraded" in plan.routing_reason


def test_clause_filters_are_derived_from_query():
    plan = build_plan("Show me the confidentiality clause")

    assert "Confidentiality" in plan.clause_filters


def test_strategy_prior_is_bounded():
    for strategy in AnswerStrategy:
        assert 0.0 < strategy_prior(strategy) <= 1.0
