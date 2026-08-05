import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from venkata_answering.answer_engine import NO_EVIDENCE_ANSWER, AnswerEngine

from .schemas import AnswerStrategy, QueryIntent


@dataclass
class EvaluationCase:
    question: str
    contract_id: Optional[str] = None
    expected_intent: Optional[QueryIntent] = None
    expected_strategy: Optional[AnswerStrategy] = None
    expected_clause_types: List[str] = field(default_factory=list)
    expected_answer_contains: List[str] = field(default_factory=list)
    expect_refusal: bool = False

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EvaluationCase":
        intent = payload.get("expected_intent")
        strategy = payload.get("expected_strategy")

        return cls(
            question=str(payload["question"]),
            contract_id=payload.get("contract_id"),
            expected_intent=QueryIntent(intent) if intent else None,
            expected_strategy=AnswerStrategy(strategy) if strategy else None,
            expected_clause_types=list(payload.get("expected_clause_types", [])),
            expected_answer_contains=list(payload.get("expected_answer_contains", [])),
            expect_refusal=bool(payload.get("expect_refusal", False)),
        )


@dataclass
class CaseOutcome:
    question: str
    intent: str
    strategy: str
    intent_correct: Optional[bool]
    strategy_correct: Optional[bool]
    citation_correct: Optional[bool]
    answer_correct: Optional[bool]
    refusal_correct: Optional[bool]
    faithfulness: float
    confidence: float
    citation_count: int
    latency_ms: float
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "intent": self.intent,
            "strategy": self.strategy,
            "intent_correct": self.intent_correct,
            "strategy_correct": self.strategy_correct,
            "citation_correct": self.citation_correct,
            "answer_correct": self.answer_correct,
            "refusal_correct": self.refusal_correct,
            "faithfulness": round(self.faithfulness, 4),
            "confidence": round(self.confidence, 4),
            "citation_count": self.citation_count,
            "latency_ms": round(self.latency_ms, 2),
            "warnings": self.warnings,
        }


@dataclass
class EvaluationReport:
    outcomes: List[CaseOutcome]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "cases": [outcome.to_dict() for outcome in self.outcomes],
        }


def _ratio(correct: int, total: int) -> float:
    return correct / total if total else 0.0


def _accuracy(values: Sequence[Optional[bool]]) -> float:
    scored = [value for value in values if value is not None]
    return _ratio(sum(1 for value in scored if value), len(scored))


def evaluate(engine: AnswerEngine, cases: Sequence[EvaluationCase]) -> EvaluationReport:
    outcomes: List[CaseOutcome] = []

    for case in cases:
        started = time.perf_counter()
        result = engine.answer(case.question, contract_id=case.contract_id)
        elapsed = (time.perf_counter() - started) * 1000.0

        answer_lower = result.answer.lower()
        cited_types = {citation.clause_type for citation in result.citations}

        intent_correct = (
            result.plan.intent is case.expected_intent if case.expected_intent else None
        )
        strategy_correct = (
            result.plan.strategy is case.expected_strategy if case.expected_strategy else None
        )
        citation_correct = (
            all(clause in cited_types for clause in case.expected_clause_types)
            if case.expected_clause_types
            else None
        )
        answer_correct = (
            all(token.lower() in answer_lower for token in case.expected_answer_contains)
            if case.expected_answer_contains
            else None
        )
        refusal_correct = (
            (result.answer.strip() == NO_EVIDENCE_ANSWER) == case.expect_refusal
            if case.expect_refusal
            else None
        )

        outcomes.append(
            CaseOutcome(
                question=case.question,
                intent=result.plan.intent.value,
                strategy=result.plan.strategy.value,
                intent_correct=intent_correct,
                strategy_correct=strategy_correct,
                citation_correct=citation_correct,
                answer_correct=answer_correct,
                refusal_correct=refusal_correct,
                faithfulness=result.faithfulness.score if result.faithfulness else 0.0,
                confidence=result.confidence.score if result.confidence else 0.0,
                citation_count=len(result.citations),
                latency_ms=elapsed,
                warnings=list(result.warnings),
            )
        )

    total = len(outcomes)
    grounded = sum(1 for outcome in outcomes if outcome.citation_count > 0)
    latencies = [outcome.latency_ms for outcome in outcomes]

    metrics = {
        "cases": float(total),
        "intent_accuracy": _accuracy([o.intent_correct for o in outcomes]),
        "routing_accuracy": _accuracy([o.strategy_correct for o in outcomes]),
        "citation_accuracy": _accuracy([o.citation_correct for o in outcomes]),
        "answer_accuracy": _accuracy([o.answer_correct for o in outcomes]),
        "refusal_accuracy": _accuracy([o.refusal_correct for o in outcomes]),
        "citation_coverage": _ratio(grounded, total),
        "mean_faithfulness": _ratio(sum(o.faithfulness for o in outcomes), total),
        "mean_confidence": _ratio(sum(o.confidence for o in outcomes), total),
        "mean_latency_ms": _ratio(sum(latencies), total),
        "max_latency_ms": max(latencies) if latencies else 0.0,
    }

    return EvaluationReport(outcomes=outcomes, metrics=metrics)


def load_cases(path: str) -> List[EvaluationCase]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    return [EvaluationCase.from_dict(item) for item in payload]
