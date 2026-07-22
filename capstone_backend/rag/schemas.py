from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class QueryIntent(str, Enum):
    FACT_LOOKUP = "fact_lookup"
    CLAUSE_RETRIEVAL = "clause_retrieval"
    ENTITY_LOOKUP = "entity_lookup"
    CLAUSE_COMPARISON = "clause_comparison"
    SUMMARIZATION = "summarization"
    RISK_EXPLANATION = "risk_explanation"
    LEGAL_REASONING = "legal_reasoning"


class AnswerStrategy(str, Enum):
    EXTRACTIVE_QA = "deberta_extractive_qa"
    RETRIEVER_ONLY = "retriever_only"
    ENTITY_EXTRACTION = "legalbert_entity_extraction"
    GROUNDED_LLM = "grounded_llm"


@dataclass
class Entity:
    text: str
    label: str
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "label": self.label, "score": round(self.score, 4)}


@dataclass
class RetrievedChunk:
    chunk_id: str
    contract_id: str
    text: str
    clause_type: str = ""
    section: str = ""
    page: int = 0
    risk: str = ""
    entities: List[Entity] = field(default_factory=list)
    dense_score: float = 0.0
    sparse_score: float = 0.0
    rerank_score: float = 0.0
    retrieval_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "contract_id": self.contract_id,
            "text": self.text,
            "clause_type": self.clause_type,
            "section": self.section,
            "page": self.page,
            "risk": self.risk,
            "entities": [e.to_dict() for e in self.entities],
            "dense_score": round(self.dense_score, 4),
            "sparse_score": round(self.sparse_score, 4),
            "rerank_score": round(self.rerank_score, 4),
            "retrieval_score": round(self.retrieval_score, 4),
            "metadata": self.metadata,
        }


@dataclass
class QueryPlan:
    query: str
    intent: QueryIntent
    intent_confidence: float
    strategy: AnswerStrategy
    clause_filters: List[str] = field(default_factory=list)
    intent_scores: Dict[str, float] = field(default_factory=dict)
    routing_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "intent": self.intent.value,
            "intent_confidence": round(self.intent_confidence, 4),
            "strategy": self.strategy.value,
            "clause_filters": self.clause_filters,
            "intent_scores": {k: round(v, 4) for k, v in self.intent_scores.items()},
            "routing_reason": self.routing_reason,
        }


@dataclass
class Citation:
    chunk_id: str
    contract_id: str
    clause_type: str
    section: str
    page: int
    quote: str
    support_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "contract_id": self.contract_id,
            "clause_type": self.clause_type,
            "section": self.section,
            "page": self.page,
            "quote": self.quote,
            "support_score": round(self.support_score, 4),
        }


@dataclass
class FaithfulnessReport:
    score: float
    supported: bool
    threshold: float
    unsupported_statements: List[str] = field(default_factory=list)
    statement_scores: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "supported": self.supported,
            "threshold": round(self.threshold, 4),
            "unsupported_statements": self.unsupported_statements,
            "statement_scores": [round(s, 4) for s in self.statement_scores],
        }


@dataclass
class ConfidenceReport:
    score: float
    band: str
    signals: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 4),
            "band": self.band,
            "signals": {k: round(v, 4) for k, v in self.signals.items()},
        }


@dataclass
class AnswerResult:
    query: str
    answer: str
    plan: QueryPlan
    citations: List[Citation] = field(default_factory=list)
    evidence: List[RetrievedChunk] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    confidence: Optional[ConfidenceReport] = None
    faithfulness: Optional[FaithfulnessReport] = None
    warnings: List[str] = field(default_factory=list)
    prompt: str = ""
    generator: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "plan": self.plan.to_dict(),
            "citations": [c.to_dict() for c in self.citations],
            "evidence": [e.to_dict() for e in self.evidence],
            "entities": [e.to_dict() for e in self.entities],
            "confidence": self.confidence.to_dict() if self.confidence else None,
            "faithfulness": self.faithfulness.to_dict() if self.faithfulness else None,
            "warnings": self.warnings,
            "prompt": self.prompt,
            "generator": self.generator,
            "latency_ms": round(self.latency_ms, 2),
        }
