from rag.answer_engine import NO_EVIDENCE_ANSWER, AnswerEngine
from rag.config import load_config
from rag.llm_client import LLMClient, LLMResponse, LLMUnavailableError
from rag.retrieval import InMemoryClauseRetriever
from rag.schemas import AnswerStrategy, QueryIntent


class StubLLMClient(LLMClient):
    def __init__(self, config, text="", fail=False):
        super().__init__(config)
        self._text = text
        self._fail = fail
        self.calls = []

    def is_available(self):
        return True

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))

        if self._fail:
            raise LLMUnavailableError("stub failure")

        return LLMResponse(text=self._text, model="stub-model", provider="stub", raw={})


def test_fact_lookup_returns_span_with_citation(retriever, config):
    engine = AnswerEngine(retriever, config)
    result = engine.answer("What is the notice period for termination?", contract_id="contract-1")

    assert result.plan.strategy is AnswerStrategy.EXTRACTIVE_QA
    assert "thirty" in result.answer.lower()
    assert result.citations
    assert result.confidence.score > 0.0


def test_entity_lookup_lists_parties(retriever, config):
    engine = AnswerEngine(retriever, config)
    result = engine.answer("Who are the parties to this agreement?", contract_id="contract-1")

    assert result.plan.intent is QueryIntent.ENTITY_LOOKUP
    assert "Acme Corporation" in result.answer


def test_clause_retrieval_returns_clause_text(retriever, config):
    engine = AnswerEngine(retriever, config)
    result = engine.answer("Retrieve the governing law section", contract_id="contract-1")

    assert result.plan.strategy is AnswerStrategy.RETRIEVER_ONLY
    assert "Delaware" in result.answer


def test_grounded_llm_answer_is_cited(retriever, config):
    client = StubLLMClient(
        config,
        text="Either party may terminate on thirty days written notice [E1].",
    )
    engine = AnswerEngine(retriever, config, client)
    result = engine.answer(
        "What happens if a party wants to exit the agreement early?", contract_id="contract-1"
    )

    assert result.plan.strategy is AnswerStrategy.GROUNDED_LLM
    assert result.generator == "stub:stub-model"
    assert result.citations
    assert client.calls


def test_grounded_llm_failure_falls_back_to_extraction(retriever, config):
    client = StubLLMClient(config, fail=True)
    engine = AnswerEngine(retriever, config, client)
    result = engine.answer(
        "What happens if a party wants to exit the agreement early?", contract_id="contract-1"
    )

    assert any("grounded generation failed" in warning for warning in result.warnings)
    assert result.answer


def test_unknown_contract_returns_grounding_guard(retriever, config):
    engine = AnswerEngine(retriever, config)
    result = engine.answer("What is the payment schedule?", contract_id="missing-contract")

    assert result.answer == NO_EVIDENCE_ANSWER
    assert result.generator == "grounding_guard"
    assert result.evidence == []


def test_empty_query_is_rejected(retriever, config):
    engine = AnswerEngine(retriever, config)
    result = engine.answer("   ")

    assert result.answer == NO_EVIDENCE_ANSWER
    assert "empty query" in result.warnings


def test_result_serialises_to_primitives(retriever, config):
    engine = AnswerEngine(retriever, config)
    payload = engine.answer("What is the notice period?", contract_id="contract-1").to_dict()

    assert payload["plan"]["intent"]
    assert isinstance(payload["citations"], list)
    assert isinstance(payload["confidence"]["signals"], dict)


def test_engine_handles_retriever_errors(config):
    class BrokenRetriever:
        def retrieve(self, query, top_k=5, clause_filters=None, contract_id=None):
            raise RuntimeError("index unavailable")

    engine = AnswerEngine(BrokenRetriever(), config)
    result = engine.answer("What is the notice period?")

    assert result.answer == NO_EVIDENCE_ANSWER
    assert "no evidence retrieved" in result.warnings


def test_engine_accepts_empty_index(config):
    engine = AnswerEngine(InMemoryClauseRetriever(), config)
    result = engine.answer("What is the notice period?")

    assert result.answer == NO_EVIDENCE_ANSWER
