import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from .config import RagConfig
from .llm_client import LLMClient, LLMUnavailableError
from .prompt_builder import build_grounded_prompt
from .schemas import QueryIntent, RetrievedChunk

logger = logging.getLogger(__name__)

NO_EVIDENCE_ANSWER = "The provided contract evidence does not answer this question."


@dataclass
class GroundedGeneration:
    answer: str
    prompt: str
    generator: str
    evidence_map: Dict[str, RetrievedChunk] = field(default_factory=dict)
    succeeded: bool = True
    error: str = ""


def generate(
    query: str,
    intent: QueryIntent,
    chunks: Sequence[RetrievedChunk],
    config: RagConfig,
    client: LLMClient,
) -> GroundedGeneration:
    prompt = build_grounded_prompt(query, intent, chunks, config.max_context_chars)

    if not chunks:
        return GroundedGeneration(
            answer=NO_EVIDENCE_ANSWER,
            prompt=prompt.text,
            generator="grounding_guard",
            evidence_map={},
            succeeded=False,
            error="no evidence retrieved",
        )

    try:
        response = client.generate(prompt.system, prompt.user)
    except LLMUnavailableError as error:
        logger.warning("Grounded generation unavailable: %s", error)
        return GroundedGeneration(
            answer="",
            prompt=prompt.text,
            generator=f"{client.provider}:{client.model}",
            evidence_map=prompt.evidence_map,
            succeeded=False,
            error=str(error),
        )

    answer = response.text.strip()

    if not answer:
        return GroundedGeneration(
            answer="",
            prompt=prompt.text,
            generator=f"{response.provider}:{response.model}",
            evidence_map=prompt.evidence_map,
            succeeded=False,
            error="empty generation",
        )

    return GroundedGeneration(
        answer=answer,
        prompt=prompt.text,
        generator=f"{response.provider}:{response.model}",
        evidence_map=prompt.evidence_map,
        succeeded=True,
    )
