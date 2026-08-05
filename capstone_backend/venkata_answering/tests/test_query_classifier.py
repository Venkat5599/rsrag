import pytest

from venkata_answering.query_classifier import classify_query
from rag.schemas import QueryIntent

CASES = [
    ("What is the notice period for termination?", QueryIntent.FACT_LOOKUP),
    ("Show me the confidentiality clause", QueryIntent.CLAUSE_RETRIEVAL),
    ("Who are the parties to this agreement?", QueryIntent.ENTITY_LOOKUP),
    ("Compare the termination clauses of both agreements", QueryIntent.CLAUSE_COMPARISON),
    ("Summarise this agreement", QueryIntent.SUMMARIZATION),
    ("What are the risks in the indemnity clause?", QueryIntent.RISK_EXPLANATION),
    ("What happens if the customer fails to pay?", QueryIntent.LEGAL_REASONING),
]


@pytest.mark.parametrize("query,expected", CASES)
def test_intent_detection(query, expected):
    intent, confidence, scores = classify_query(query)

    assert intent is expected
    assert 0.0 <= confidence <= 1.0
    assert scores


def test_empty_query_defaults_to_fact_lookup():
    intent, confidence, _ = classify_query("")

    assert intent is QueryIntent.FACT_LOOKUP
    assert confidence == 0.0
