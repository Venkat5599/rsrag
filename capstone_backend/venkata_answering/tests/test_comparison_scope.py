"""Tests for honest clause comparison (spec section 11).

The engine will happily answer a comparison question from a single contract - the
retriever has no way to know a second agreement was meant. What it must never do is
answer that way *silently*. These tests hold the warning in place.
"""

import pytest

from rag.retrieval import InMemoryClauseRetriever, chunks_from_clause_map
from rag.schemas import QueryIntent
from venkata_answering.answer_engine import AnswerEngine

COMPARISON_QUERY = "Compare the termination clauses of these two agreements."

SINGLE_SCOPE_WARNING = "only one contract is in scope"
ONE_SIDED_WARNING = "comparison is one-sided"

SECOND_CONTRACT = {
    "Termination for Convenience": {
        "span": (
            "Either party may terminate this Agreement for convenience upon ninety (90) "
            "days prior written notice to the other party."
        ),
        "score": 0.8,
        "page": 4,
        "section": "Clause 3.1",
    },
    "Governing Law": {
        "span": "This Agreement shall be governed by the laws of the State of New York.",
        "score": 0.75,
        "page": 9,
        "section": "Clause 12.2",
    },
}


@pytest.fixture
def two_contract_retriever(sample_chunks):
    second = chunks_from_clause_map(SECOND_CONTRACT, "contract-2", "sow.pdf")
    return InMemoryClauseRetriever(list(sample_chunks) + list(second))


def _has(result, fragment):
    return any(fragment in warning for warning in result.warnings)


def test_comparison_routes_to_the_comparison_intent(retriever, config):
    engine = AnswerEngine(retriever, config)
    result = engine.answer(COMPARISON_QUERY, contract_id="contract-1")

    assert result.plan.intent is QueryIntent.CLAUSE_COMPARISON


def test_comparison_with_one_contract_warns_instead_of_answering_silently(
    retriever, config
):
    engine = AnswerEngine(retriever, config)
    result = engine.answer(COMPARISON_QUERY, contract_id="contract-1")

    assert _has(result, SINGLE_SCOPE_WARNING)


def test_comparison_across_two_contracts_does_not_warn(two_contract_retriever, config):
    engine = AnswerEngine(two_contract_retriever, config)
    result = engine.answer(
        COMPARISON_QUERY, contract_ids=["contract-1", "contract-2"], top_k=4
    )

    assert not _has(result, SINGLE_SCOPE_WARNING)
    assert not _has(result, ONE_SIDED_WARNING)


def test_comparison_evidence_spans_both_contracts(two_contract_retriever, config):
    engine = AnswerEngine(two_contract_retriever, config)
    result = engine.answer(
        COMPARISON_QUERY, contract_ids=["contract-1", "contract-2"], top_k=4
    )

    assert {chunk.contract_id for chunk in result.evidence} == {
        "contract-1",
        "contract-2",
    }


def test_a_requested_contract_with_no_evidence_is_named(two_contract_retriever, config):
    engine = AnswerEngine(two_contract_retriever, config)
    result = engine.answer(
        COMPARISON_QUERY, contract_ids=["contract-1", "contract-404"], top_k=4
    )

    assert _has(result, ONE_SIDED_WARNING)
    assert _has(result, "contract-404")


def test_non_comparison_queries_are_not_warned_about(retriever, config):
    engine = AnswerEngine(retriever, config)
    result = engine.answer(
        "What is the notice period for termination?", contract_id="contract-1"
    )

    assert not _has(result, SINGLE_SCOPE_WARNING)
