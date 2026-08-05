"""Venkata's ownership area: query understanding and grounded answering.

Scope (LegalEase v3 Hybrid RAG spec, section 15 - Venkata):
    * Query classifier            -> query_classifier
    * Adaptive routing            -> router
    * DeBERTa integration         -> extractive_qa (models.get_qa_pipeline)
    * LegalBERT integration       -> entity_lookup (models.get_ner_pipeline)
    * Grounded LLM                -> grounded_llm, llm_client
    * Prompt builder              -> prompt_builder
    * Citation engine             -> citation_engine
    * Confidence engine           -> confidence_engine
    * Faithfulness verification   -> faithfulness
    * Grounding validation        -> grounding_validator
    * Section 12 response payload -> answer_contract
    * Routing benchmark           -> routing_benchmark

Everything here consumes evidence through the ``rag.retrieval.ClauseRetriever``
protocol, so it works unchanged against Kiran's hybrid knowledge base or against
the ``InMemoryClauseRetriever`` baseline.
"""

from .answer_contract import ContractViolation, LegalResponse, build_response
from .answer_engine import NO_EVIDENCE_ANSWER, AnswerEngine
from .grounding_validator import GroundingReport, validate_grounding
from .llm_client import LLMClient, LLMResponse, LLMUnavailableError
from .router import build_plan, strategy_prior
from .routing_benchmark import RoutingReport, run_routing_benchmark

__all__ = [
    "AnswerEngine",
    "ContractViolation",
    "GroundingReport",
    "LLMClient",
    "LLMResponse",
    "LLMUnavailableError",
    "LegalResponse",
    "NO_EVIDENCE_ANSWER",
    "RoutingReport",
    "build_plan",
    "build_response",
    "run_routing_benchmark",
    "strategy_prior",
    "validate_grounding",
]
