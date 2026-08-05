"""Retrieval benchmark.

The integration note asks for retrieval to be benchmarked. This measures retrieval
in isolation, with no answering model involved, so a retrieval regression cannot be
masked by a good answer, and compares three configurations:

    baseline      rag.retrieval.InMemoryClauseRetriever, the pre-hybrid single-pass scan
    hybrid        metadata filter + dense + BM25 fusion, reranking disabled
    hybrid+rerank the full spec section 8-10 pipeline

Metrics, computed against the ``expected_clause_types`` already labelled in
``benchmarks/cases.json``:

    recall@k    did the correct clause appear in the top k at all
    precision@1 was the top hit correct
    MRR         1 / rank of the first correct clause, averaged; this is the metric
                that moves when reranking works, because reranking changes rank
                order without changing which clauses are in the pool

Run it with ``python -m kiran_retrieval.retrieval_benchmark``.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from rag.clause_taxonomy import match_clause_types
from rag.config import load_config
from rag.retrieval import InMemoryClauseRetriever, chunks_from_clause_map

from .knowledge_base import KnowledgeBase

BENCHMARK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmarks")
DOCUMENTS_PATH = os.path.join(BENCHMARK_DIR, "documents.json")

# Retrieval needs a clause-type label on every case. ``cases.json`` is the
# end-to-end answering fixture and labels clause types on only one of its eleven
# questions, so it is left alone and a retrieval-specific set is used here. The
# paraphrased questions ("Can either party walk away from the contract early?")
# matter: they are the ones that separate hybrid retrieval from keyword search,
# because they share no distinctive term with the clause they should return.
CASES_PATH = os.path.join(BENCHMARK_DIR, "retrieval_cases.json")


@dataclass
class RetrievalCase:
    question: str
    expected_clause_types: List[str]
    contract_id: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "RetrievalCase":
        return cls(
            question=str(payload["question"]),
            expected_clause_types=list(payload.get("expected_clause_types", [])),
            contract_id=payload.get("contract_id"),
        )


@dataclass
class ConfigurationResult:
    name: str
    cases: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    ranks: List[Optional[int]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "cases": self.cases,
            "metrics": {name: round(value, 4) for name, value in self.metrics.items()},
        }


@dataclass
class RetrievalReport:
    results: List[ConfigurationResult] = field(default_factory=list)
    top_k: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {"configurations": [result.to_dict() for result in self.results]}

    def by_name(self, name: str) -> Optional[ConfigurationResult]:
        for result in self.results:
            if result.name == name:
                return result
        return None

    def render(self) -> str:
        recall_key = f"recall@{self.top_k}"
        lines = ["Retrieval benchmark", "=" * 62]
        lines.append(
            f"{'configuration':<18}{recall_key:>10}{'recall@1':>10}"
            f"{'precision@1':>13}{'MRR':>10}"
        )

        for result in self.results:
            lines.append(
                f"{result.name:<18}"
                f"{result.metrics[recall_key]:>10.3f}"
                f"{result.metrics['recall@1']:>10.3f}"
                f"{result.metrics['precision@1']:>13.3f}"
                f"{result.metrics['mrr']:>10.3f}"
            )

        return "\n".join(lines)


def load_documents(path: str = DOCUMENTS_PATH) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_cases(path: str = CASES_PATH) -> List[RetrievalCase]:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    return [
        RetrievalCase.from_dict(item)
        for item in payload
        if item.get("expected_clause_types")
    ]


def _first_correct_rank(hits, expected: Sequence[str]) -> Optional[int]:
    wanted = {clause.lower() for clause in expected}

    for position, chunk in enumerate(hits, start=1):
        if chunk.clause_type.lower() in wanted:
            return position

    return None


def _score(name: str, ranks: Sequence[Optional[int]], top_k: int) -> ConfigurationResult:
    total = len(ranks) or 1
    found = [rank for rank in ranks if rank is not None]

    return ConfigurationResult(
        name=name,
        cases=len(ranks),
        ranks=list(ranks),
        metrics={
            f"recall@{top_k}": len(found) / total,
            "recall@1": sum(1 for rank in found if rank == 1) / total,
            "precision@1": sum(1 for rank in found if rank == 1) / total,
            "mrr": sum(1.0 / rank for rank in found) / total,
        },
    )


def _evaluate(retriever, cases: Sequence[RetrievalCase], top_k: int) -> List[Optional[int]]:
    ranks: List[Optional[int]] = []

    for case in cases:
        hits = retriever.retrieve(
            case.question,
            top_k=top_k,
            clause_filters=match_clause_types(case.question),
            contract_id=case.contract_id,
        )
        ranks.append(_first_correct_rank(hits, case.expected_clause_types))

    return ranks


def _baseline_retriever(documents: Sequence[Dict[str, Any]], embedding_model: str):
    retriever = InMemoryClauseRetriever(embedding_model=embedding_model)

    for document in documents:
        retriever.add(
            chunks_from_clause_map(
                document.get("clauses") or {},
                str(document.get("docId", "")),
                str(document.get("filename", "")),
            )
        )

    return retriever


class _NoOpReranker:
    """Stands in for the cross encoder so the rerank stage can be isolated."""

    def backend_name(self) -> str:
        return "disabled"

    def rerank(self, query, chunks, top_k=None):
        return list(chunks)[:top_k] if top_k else list(chunks)


def run_retrieval_benchmark(
    documents: Optional[Sequence[Dict[str, Any]]] = None,
    cases: Optional[Sequence[RetrievalCase]] = None,
    top_k: int = 5,
) -> RetrievalReport:
    config = load_config()
    documents = list(documents if documents is not None else load_documents())
    cases = list(cases if cases is not None else load_cases())

    baseline = _baseline_retriever(documents, config.embedding_model)

    hybrid = KnowledgeBase(config)
    reranked = KnowledgeBase(config)

    for document in documents:
        hybrid.index_document(document)
        reranked.index_document(document)

    # Identical index, rerank stage removed, so the delta between the two rows is
    # attributable to reranking alone.
    hybrid.retriever._reranker = _NoOpReranker()

    return RetrievalReport(
        results=[
            _score("baseline", _evaluate(baseline, cases, top_k), top_k),
            _score("hybrid", _evaluate(hybrid, cases, top_k), top_k),
            _score("hybrid+rerank", _evaluate(reranked, cases, top_k), top_k),
        ],
        top_k=top_k,
    )


def main() -> None:
    report = run_retrieval_benchmark()
    print(report.render())
    print()
    print(json.dumps(report.to_dict(), indent=2))


if __name__ == "__main__":
    main()
