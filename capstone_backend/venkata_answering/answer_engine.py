import logging
import time
from typing import Dict, List, Optional, Sequence

from . import citation_engine
from . import confidence_engine
from . import entity_lookup
from . import extractive_qa
from . import faithfulness
from . import grounded_llm
from . import router
from rag.config import RagConfig, load_config
from .llm_client import LLMClient
from .prompt_builder import build_grounded_prompt
from rag.retrieval import ClauseRetriever
from rag.schemas import AnswerResult, AnswerStrategy, Entity, QueryPlan, RetrievedChunk

logger = logging.getLogger(__name__)

NO_EVIDENCE_ANSWER = "The provided contract evidence does not answer this question."


class AnswerEngine:
    def __init__(
        self,
        retriever: ClauseRetriever,
        config: Optional[RagConfig] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        self._retriever = retriever
        self._config = config or load_config()
        self._llm = llm_client or LLMClient(self._config)

    @property
    def config(self) -> RagConfig:
        return self._config

    def llm_available(self) -> bool:
        return self._llm.is_available()

    def plan(self, query: str) -> QueryPlan:
        return router.build_plan(
            query,
            embedding_model=self._config.embedding_model,
            llm_available=self._llm.is_available(),
        )

    def answer(
        self,
        query: str,
        contract_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> AnswerResult:
        started = time.perf_counter()
        query = (query or "").strip()
        plan = self.plan(query)

        if not query:
            return self._empty_result(plan, "empty query", started)

        evidence = self._retrieve(plan, contract_id, top_k)

        if not evidence:
            return self._empty_result(plan, "no evidence retrieved", started)

        if plan.strategy is AnswerStrategy.GROUNDED_LLM:
            return self._answer_with_llm(plan, evidence, started)

        if plan.strategy is AnswerStrategy.ENTITY_EXTRACTION:
            return self._answer_with_entities(plan, evidence, started)

        if plan.strategy is AnswerStrategy.RETRIEVER_ONLY:
            return self._answer_with_retriever(plan, evidence, started)

        return self._answer_with_extractive_qa(plan, evidence, started)

    def _retrieve(
        self,
        plan: QueryPlan,
        contract_id: Optional[str],
        top_k: Optional[int],
    ) -> List[RetrievedChunk]:
        limit = top_k or self._config.top_k_context

        try:
            retrieved = self._retriever.retrieve(
                plan.query,
                top_k=limit,
                clause_filters=plan.clause_filters,
                contract_id=contract_id,
            )
        except Exception as error:
            logger.error("Retrieval failed: %s", error)
            return []

        return list(retrieved or [])

    def _finalize(
        self,
        plan: QueryPlan,
        answer: str,
        evidence: Sequence[RetrievedChunk],
        answer_model_score: float,
        generator: str,
        started: float,
        prompt: str = "",
        evidence_map: Optional[Dict[str, RetrievedChunk]] = None,
        entities: Optional[Sequence[Entity]] = None,
        warnings: Optional[Sequence[str]] = None,
    ) -> AnswerResult:
        collected_warnings = list(warnings or [])

        citations = citation_engine.build_citations(
            answer,
            evidence,
            evidence_map=evidence_map,
            embedding_model=self._config.embedding_model,
            limit=self._config.top_k_context,
        )

        report = faithfulness.verify_answer(
            answer,
            evidence,
            threshold=self._config.faithfulness_threshold,
            embedding_model=self._config.embedding_model,
        )

        confidence = confidence_engine.score_answer(
            plan=plan,
            evidence=evidence,
            citations=citations,
            faithfulness=report,
            answer_model_score=answer_model_score,
        )

        if not report.supported:
            collected_warnings.append("answer is not fully supported by retrieved evidence")

        if confidence.score < self._config.low_confidence_threshold:
            collected_warnings.append("confidence below reliability threshold")

        final_answer = answer

        if confidence.score < self._config.min_answer_confidence:
            final_answer = NO_EVIDENCE_ANSWER
            collected_warnings.append("answer suppressed because confidence was critically low")

        return AnswerResult(
            query=plan.query,
            answer=final_answer,
            plan=plan,
            citations=citations,
            evidence=list(evidence),
            entities=list(entities or []),
            confidence=confidence,
            faithfulness=report,
            warnings=collected_warnings,
            prompt=prompt,
            generator=generator,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _answer_with_extractive_qa(
        self,
        plan: QueryPlan,
        evidence: Sequence[RetrievedChunk],
        started: float,
        extra_warnings: Optional[Sequence[str]] = None,
        prompt: str = "",
    ) -> AnswerResult:
        extracted = extractive_qa.answer_from_chunks(
            plan.query, evidence, self._config.qa_model
        )

        warnings = list(extra_warnings or [])

        if not extracted.model_used:
            warnings.append("extractive model unavailable, lexical span selection used")

        ordered = list(evidence)

        if extracted.chunk is not None:
            ordered = [extracted.chunk] + [
                chunk for chunk in evidence if chunk.chunk_id != extracted.chunk.chunk_id
            ]

        generator = (
            f"deberta:{self._config.qa_model}" if extracted.model_used else "lexical_span"
        )

        return self._finalize(
            plan=plan,
            answer=extracted.text or NO_EVIDENCE_ANSWER,
            evidence=ordered,
            answer_model_score=extracted.score,
            generator=generator,
            started=started,
            prompt=prompt,
            warnings=warnings,
        )

    def _answer_with_entities(
        self,
        plan: QueryPlan,
        evidence: Sequence[RetrievedChunk],
        started: float,
    ) -> AnswerResult:
        labels = entity_lookup.requested_labels(plan.query)
        entities, model_used = entity_lookup.extract_entities(
            evidence, self._config.ner_model, labels
        )

        warnings: List[str] = []

        if not model_used:
            warnings.append("entity model unavailable, stored clause entities used")

        generator = (
            f"legalbert:{self._config.ner_model}" if model_used else "stored_entities"
        )

        return self._finalize(
            plan=plan,
            answer=entity_lookup.format_entity_answer(entities),
            evidence=evidence,
            answer_model_score=entities[0].score if entities else 0.0,
            generator=generator,
            started=started,
            entities=entities,
            warnings=warnings,
        )

    def _answer_with_retriever(
        self,
        plan: QueryPlan,
        evidence: Sequence[RetrievedChunk],
        started: float,
    ) -> AnswerResult:
        blocks: List[str] = []

        for chunk in evidence:
            location = chunk.section or chunk.clause_type or "Unspecified section"
            page = f"page {chunk.page}" if chunk.page else "page not recorded"
            label = chunk.clause_type or "Clause"
            blocks.append(f"{label} ({location}, {page}):\n{chunk.text}")

        return self._finalize(
            plan=plan,
            answer="\n\n".join(blocks),
            evidence=evidence,
            answer_model_score=evidence[0].retrieval_score if evidence else 0.0,
            generator="retriever_only",
            started=started,
        )

    def _answer_with_llm(
        self,
        plan: QueryPlan,
        evidence: Sequence[RetrievedChunk],
        started: float,
    ) -> AnswerResult:
        generation = grounded_llm.generate(
            plan.query, plan.intent, evidence, self._config, self._llm
        )

        if generation.succeeded and generation.answer:
            return self._finalize(
                plan=plan,
                answer=generation.answer,
                evidence=evidence,
                answer_model_score=0.7,
                generator=generation.generator,
                started=started,
                prompt=generation.prompt,
                evidence_map=generation.evidence_map,
            )

        reason = generation.error or "unknown error"

        return self._answer_with_extractive_qa(
            plan,
            evidence,
            started,
            extra_warnings=[
                f"grounded generation failed ({reason}), extractive fallback used"
            ],
            prompt=generation.prompt,
        )

    def _empty_result(self, plan: QueryPlan, reason: str, started: float) -> AnswerResult:
        prompt = build_grounded_prompt(
            plan.query, plan.intent, [], self._config.max_context_chars
        ).text

        report = faithfulness.verify_answer(
            NO_EVIDENCE_ANSWER, [], self._config.faithfulness_threshold
        )

        confidence = confidence_engine.score_answer(
            plan=plan,
            evidence=[],
            citations=[],
            faithfulness=report,
            answer_model_score=0.0,
        )

        return AnswerResult(
            query=plan.query,
            answer=NO_EVIDENCE_ANSWER,
            plan=plan,
            citations=[],
            evidence=[],
            entities=[],
            confidence=confidence,
            faithfulness=report,
            warnings=[reason],
            prompt=prompt,
            generator="grounding_guard",
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )
