from dataclasses import dataclass
from typing import Dict, List, Sequence

from .schemas import QueryIntent, RetrievedChunk

SYSTEM_PROMPT = (
    "You are a legal analysis assistant operating under strict grounding rules. "
    "You answer only from the numbered contract evidence supplied in the prompt. "
    "You never rely on outside legal knowledge, never invent clause numbers, "
    "and never speculate about terms that are not present in the evidence. "
    "If the evidence does not answer the question, reply exactly with: "
    "The provided contract evidence does not answer this question."
)

INTENT_INSTRUCTIONS: Dict[QueryIntent, str] = {
    QueryIntent.CLAUSE_COMPARISON: (
        "Compare the relevant clauses point by point. State what each clause provides, "
        "then state the practical difference. Cite every statement."
    ),
    QueryIntent.SUMMARIZATION: (
        "Summarise the evidence in at most six sentences. Cover only clauses present in the "
        "evidence and cite each sentence."
    ),
    QueryIntent.RISK_EXPLANATION: (
        "Identify the risks that the evidence actually supports. For each risk state the "
        "triggering clause language and the resulting exposure. Cite every risk."
    ),
    QueryIntent.LEGAL_REASONING: (
        "Reason step by step using only the evidence. State the governing clause language "
        "first, then the conclusion that follows from it. Cite every step."
    ),
    QueryIntent.FACT_LOOKUP: (
        "Give the direct factual answer in one or two sentences and cite the source clause."
    ),
    QueryIntent.CLAUSE_RETRIEVAL: (
        "Present the relevant clause text and identify where it appears. Do not paraphrase."
    ),
    QueryIntent.ENTITY_LOOKUP: (
        "List the requested entities exactly as written in the evidence and cite each one."
    ),
}

OUTPUT_CONTRACT = (
    "Response rules:\n"
    "1. Every factual sentence must end with a citation in the form [E<number>].\n"
    "2. Use only the evidence numbers supplied below.\n"
    "3. Do not add headings, disclaimers, or legal advice.\n"
    "4. If evidence conflicts, state the conflict instead of choosing arbitrarily."
)


@dataclass
class GroundedPrompt:
    system: str
    user: str
    evidence_map: Dict[str, RetrievedChunk]

    @property
    def text(self) -> str:
        return f"{self.system}\n\n{self.user}"


def _format_evidence(chunks: Sequence[RetrievedChunk], max_chars: int) -> tuple:
    lines: List[str] = []
    evidence_map: Dict[str, RetrievedChunk] = {}
    used = 0

    for index, chunk in enumerate(chunks, start=1):
        label = f"E{index}"
        location = chunk.section or chunk.clause_type or "Unspecified section"
        page = f"page {chunk.page}" if chunk.page else "page not recorded"
        header = f"[{label}] {chunk.clause_type or 'Clause'} | {location} | {page}"
        body = chunk.text.strip()

        entry = f"{header}\n{body}"
        if used + len(entry) > max_chars and lines:
            break

        used += len(entry)
        lines.append(entry)
        evidence_map[label] = chunk

    return "\n\n".join(lines), evidence_map


def build_grounded_prompt(
    query: str,
    intent: QueryIntent,
    chunks: Sequence[RetrievedChunk],
    max_context_chars: int = 9000,
) -> GroundedPrompt:
    evidence_text, evidence_map = _format_evidence(chunks, max_context_chars)
    instruction = INTENT_INSTRUCTIONS.get(intent, INTENT_INSTRUCTIONS[QueryIntent.FACT_LOOKUP])

    user_prompt = (
        f"Question:\n{query.strip()}\n\n"
        f"Task:\n{instruction}\n\n"
        f"{OUTPUT_CONTRACT}\n\n"
        f"Contract evidence:\n{evidence_text if evidence_text else 'No evidence retrieved.'}"
    )

    return GroundedPrompt(system=SYSTEM_PROMPT, user=user_prompt, evidence_map=evidence_map)
