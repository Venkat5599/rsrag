"""Adaptive routing benchmark.

Spec section 11 is the claim that the system does not always invoke an LLM. The
integration note asks for that claim to be benchmarked. This measures the router in
isolation, without retrieval or generation, so a routing regression cannot hide
behind a good answer.

Reported:
    intent accuracy        - did the classifier pick the right intent
    strategy accuracy      - did the router pick the right answering strategy
    per-intent precision   - of the queries routed to intent X, how many belonged
    per-intent recall      - of the queries that were X, how many were caught
    llm avoidance          - share of queries answered without the LLM, which is the
                             cost and latency claim in section 11
    confusions             - the actual (expected, predicted) pairs, so a failure is
                             diagnosable rather than just a number going down

Run it with ``python -m venkata_answering.routing_benchmark``.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rag.schemas import AnswerStrategy, QueryIntent

from .router import INTENT_STRATEGY, build_plan

# Labelled set. Every intent in the taxonomy is represented, with the phrasings a
# user actually types rather than the prototype sentences the classifier is fitted
# against, so this measures generalisation rather than memorisation.
LABELLED_QUERIES: List[Tuple[str, QueryIntent]] = [
    ("What is the notice period for termination?", QueryIntent.FACT_LOOKUP),
    ("How many days notice must we give?", QueryIntent.FACT_LOOKUP),
    ("What law governs this agreement?", QueryIntent.FACT_LOOKUP),
    ("When does the initial term end?", QueryIntent.FACT_LOOKUP),
    ("Show me the confidentiality clause.", QueryIntent.CLAUSE_RETRIEVAL),
    ("Retrieve the section covering dispute resolution.", QueryIntent.CLAUSE_RETRIEVAL),
    ("Quote the full text of the indemnity provision.", QueryIntent.CLAUSE_RETRIEVAL),
    ("Find the clause about assignment.", QueryIntent.CLAUSE_RETRIEVAL),
    ("Who are the parties to this agreement?", QueryIntent.ENTITY_LOOKUP),
    ("Which company is responsible for indemnification?", QueryIntent.ENTITY_LOOKUP),
    ("Name the signatories of this contract.", QueryIntent.ENTITY_LOOKUP),
    ("Compare the termination clauses in both agreements.", QueryIntent.CLAUSE_COMPARISON),
    (
        "What is the difference between the indemnity and liability clauses?",
        QueryIntent.CLAUSE_COMPARISON,
    ),
    ("Which one is stricter, the NDA or the MSA?", QueryIntent.CLAUSE_COMPARISON),
    ("Give me a summary of this agreement.", QueryIntent.SUMMARIZATION),
    ("Provide an overview of the key contract terms.", QueryIntent.SUMMARIZATION),
    ("What is this contract about?", QueryIntent.SUMMARIZATION),
    ("What are the main risks in this contract for the supplier?", QueryIntent.RISK_EXPLANATION),
    ("Are there any red flags in the liability terms?", QueryIntent.RISK_EXPLANATION),
    ("Explain our exposure under the indemnity clause.", QueryIntent.RISK_EXPLANATION),
    ("What happens if the customer fails to pay on time?", QueryIntent.LEGAL_REASONING),
    ("Are we permitted to subcontract our obligations?", QueryIntent.LEGAL_REASONING),
    ("Is this non-compete enforceable?", QueryIntent.LEGAL_REASONING),
    ("Would we breach the agreement by terminating early?", QueryIntent.LEGAL_REASONING),
]

NON_LLM_STRATEGY_VALUES = {
    AnswerStrategy.EXTRACTIVE_QA.value,
    AnswerStrategy.RETRIEVER_ONLY.value,
    AnswerStrategy.ENTITY_EXTRACTION.value,
}


@dataclass
class RoutingOutcome:
    query: str
    expected_intent: str
    predicted_intent: str
    expected_strategy: str
    predicted_strategy: str
    confidence: float

    @property
    def intent_correct(self) -> bool:
        return self.expected_intent == self.predicted_intent

    @property
    def strategy_correct(self) -> bool:
        return self.expected_strategy == self.predicted_strategy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "expected_intent": self.expected_intent,
            "predicted_intent": self.predicted_intent,
            "expected_strategy": self.expected_strategy,
            "predicted_strategy": self.predicted_strategy,
            "confidence": round(self.confidence, 4),
            "intent_correct": self.intent_correct,
            "strategy_correct": self.strategy_correct,
        }


@dataclass
class RoutingReport:
    outcomes: List[RoutingOutcome] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    per_intent: Dict[str, Dict[str, float]] = field(default_factory=dict)
    confusions: List[Tuple[str, str, int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": {name: round(value, 4) for name, value in self.metrics.items()},
            "per_intent": {
                intent: {name: round(value, 4) for name, value in scores.items()}
                for intent, scores in self.per_intent.items()
            },
            "confusions": [
                {"expected": expected, "predicted": predicted, "count": count}
                for expected, predicted, count in self.confusions
            ],
            "cases": [outcome.to_dict() for outcome in self.outcomes],
        }

    def render(self) -> str:
        lines = ["Adaptive routing benchmark", "=" * 40]

        for name, value in self.metrics.items():
            lines.append(f"{name:<24} {value:.4f}")

        lines.append("")
        lines.append(f"{'intent':<24}{'precision':>10}{'recall':>10}{'support':>9}")

        for intent, scores in sorted(self.per_intent.items()):
            lines.append(
                f"{intent:<24}{scores['precision']:>10.3f}"
                f"{scores['recall']:>10.3f}{int(scores['support']):>9}"
            )

        if self.confusions:
            lines.append("")
            lines.append("Confusions (expected -> predicted):")
            for expected, predicted, count in self.confusions:
                lines.append(f"  {expected} -> {predicted} ({count})")

        return "\n".join(lines)


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def run_routing_benchmark(
    queries: Optional[Sequence[Tuple[str, QueryIntent]]] = None,
    embedding_model: Optional[str] = None,
    llm_available: bool = True,
) -> RoutingReport:
    """Score the router against a labelled query set.

    ``llm_available=False`` exercises the degradation path in ``router.LLM_FALLBACK``:
    with no LLM configured the generative intents must still route somewhere useful
    rather than fail, and this is how that is verified.
    """

    cases = list(queries or LABELLED_QUERIES)
    outcomes: List[RoutingOutcome] = []

    for query, expected_intent in cases:
        plan = build_plan(query, embedding_model=embedding_model, llm_available=llm_available)
        expected_strategy = INTENT_STRATEGY[expected_intent]

        outcomes.append(
            RoutingOutcome(
                query=query,
                expected_intent=expected_intent.value,
                predicted_intent=plan.intent.value,
                expected_strategy=expected_strategy.value,
                predicted_strategy=plan.strategy.value,
                confidence=plan.intent_confidence,
            )
        )

    per_intent: Dict[str, Dict[str, float]] = {}

    for intent in QueryIntent:
        name = intent.value
        expected = [outcome for outcome in outcomes if outcome.expected_intent == name]
        predicted = [outcome for outcome in outcomes if outcome.predicted_intent == name]
        hits = [outcome for outcome in expected if outcome.intent_correct]

        if not expected and not predicted:
            continue

        per_intent[name] = {
            "precision": _ratio(len(hits), len(predicted)),
            "recall": _ratio(len(hits), len(expected)),
            "support": float(len(expected)),
        }

    confusion_counts: Dict[Tuple[str, str], int] = {}

    for outcome in outcomes:
        if outcome.intent_correct:
            continue
        key = (outcome.expected_intent, outcome.predicted_intent)
        confusion_counts[key] = confusion_counts.get(key, 0) + 1

    confusions = sorted(
        (
            (expected, predicted, count)
            for (expected, predicted), count in confusion_counts.items()
        ),
        key=lambda item: (-item[2], item[0], item[1]),
    )

    avoided = sum(
        1 for outcome in outcomes if outcome.predicted_strategy in NON_LLM_STRATEGY_VALUES
    )
    should_avoid = sum(
        1 for outcome in outcomes if outcome.expected_strategy in NON_LLM_STRATEGY_VALUES
    )

    metrics = {
        "cases": float(len(outcomes)),
        "intent_accuracy": _ratio(
            sum(1 for outcome in outcomes if outcome.intent_correct), len(outcomes)
        ),
        "strategy_accuracy": _ratio(
            sum(1 for outcome in outcomes if outcome.strategy_correct), len(outcomes)
        ),
        "mean_confidence": _ratio(
            sum(outcome.confidence for outcome in outcomes), len(outcomes)
        ),
        "llm_avoidance_rate": _ratio(avoided, len(outcomes)),
        "llm_avoidance_target": _ratio(should_avoid, len(outcomes)),
    }

    return RoutingReport(
        outcomes=outcomes, metrics=metrics, per_intent=per_intent, confusions=confusions
    )


def main() -> None:
    report = run_routing_benchmark()
    print(report.render())
    print()
    print("Degraded mode (no LLM configured):")
    degraded = run_routing_benchmark(llm_available=False)
    print(json.dumps(degraded.metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
