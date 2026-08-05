import json
import os

from venkata_answering.answer_engine import AnswerEngine
from rag.evaluation import EvaluationCase, evaluate, load_cases
from rag.schemas import AnswerStrategy, QueryIntent

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "..", "benchmarks")


def test_evaluation_reports_routing_accuracy(retriever, config):
    engine = AnswerEngine(retriever, config)
    cases = [
        EvaluationCase(
            question="What is the notice period for termination?",
            contract_id="contract-1",
            expected_intent=QueryIntent.FACT_LOOKUP,
            expected_strategy=AnswerStrategy.EXTRACTIVE_QA,
            expected_clause_types=["Termination for Convenience"],
            expected_answer_contains=["thirty"],
        ),
        EvaluationCase(
            question="Who are the parties to this agreement?",
            contract_id="contract-1",
            expected_intent=QueryIntent.ENTITY_LOOKUP,
            expected_answer_contains=["Acme Corporation"],
        ),
    ]

    report = evaluate(engine, cases)

    assert report.metrics["cases"] == 2
    assert report.metrics["intent_accuracy"] == 1.0
    assert report.metrics["routing_accuracy"] == 1.0
    assert report.metrics["answer_accuracy"] == 1.0
    assert report.metrics["citation_coverage"] == 1.0


def test_refusal_case_is_scored(retriever, config):
    engine = AnswerEngine(retriever, config)
    cases = [
        EvaluationCase(
            question="What is the delivery schedule for hardware spares?",
            contract_id="missing-contract",
            expect_refusal=True,
        )
    ]

    report = evaluate(engine, cases)

    assert report.metrics["refusal_accuracy"] == 1.0


def test_report_is_json_serialisable(retriever, config):
    engine = AnswerEngine(retriever, config)
    report = evaluate(engine, [EvaluationCase(question="What law governs this agreement?", contract_id="contract-1")])

    assert json.dumps(report.to_dict())


def test_benchmark_case_file_loads():
    cases = load_cases(os.path.join(BENCHMARK_DIR, "cases.json"))

    assert len(cases) >= 10
    assert any(case.expect_refusal for case in cases)
