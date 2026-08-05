"""The section 12 response contract.

Spec section 12: every response must include the final answer, a confidence score,
supporting evidence, clause numbers and page numbers. ``AnswerResult.to_dict`` can
serialise all of that, but it cannot tell you whether it is actually present. An
answer that ships with an empty citation list still serialises perfectly.

This module is the enforcement point. :func:`build_response` produces the payload the
API returns and raises :class:`ContractViolation` when a required element is missing,
so an uncited or unscored legal answer fails loudly instead of reaching a user who
has no way to tell it apart from a grounded one.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rag.schemas import AnswerResult, AnswerStrategy

from .grounding_validator import GroundingReport, validate_grounding

REFUSAL_MARKER = "does not answer this question"

# Only the generative strategy is asked to emit [E<n>] markers. The others quote
# retrieved text directly, so their grounding is structural.
CITATION_MARKER_STRATEGIES = {AnswerStrategy.GROUNDED_LLM}


class ContractViolation(ValueError):
    """Raised when a response does not satisfy the section 12 requirements."""

    def __init__(self, violations: List[str]) -> None:
        self.violations = violations
        super().__init__("; ".join(violations))


@dataclass
class EvidenceReference:
    """One citation, rendered the way section 12's worked example shows it."""

    clause_type: str
    clause_number: str
    page: Optional[int]
    quote: str
    support_score: float
    chunk_id: str
    contract_id: str

    @property
    def display(self) -> str:
        location = self.clause_number or self.clause_type or "Unspecified clause"
        page = f"Page {self.page}" if self.page else "Page not recorded"
        return f"{location} {page}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clause_type": self.clause_type,
            "clause_number": self.clause_number,
            "page": self.page,
            "quote": self.quote,
            "support_score": round(self.support_score, 4),
            "chunk_id": self.chunk_id,
            "contract_id": self.contract_id,
            "display": self.display,
        }


@dataclass
class LegalResponse:
    query: str
    answer: str
    confidence: float
    confidence_band: str
    evidence: List[EvidenceReference] = field(default_factory=list)
    intent: str = ""
    strategy: str = ""
    generator: str = ""
    faithfulness: float = 0.0
    faithful: bool = False
    grounding: Optional[GroundingReport] = None
    warnings: List[str] = field(default_factory=list)
    latency_ms: float = 0.0
    is_refusal: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "confidence": round(self.confidence, 4),
            "confidence_band": self.confidence_band,
            "evidence": [reference.to_dict() for reference in self.evidence],
            "clause_numbers": [reference.clause_number for reference in self.evidence],
            "pages": [reference.page for reference in self.evidence],
            "intent": self.intent,
            "strategy": self.strategy,
            "generator": self.generator,
            "faithfulness": round(self.faithfulness, 4),
            "faithful": self.faithful,
            "grounding": self.grounding.to_dict() if self.grounding else None,
            "warnings": self.warnings,
            "latency_ms": round(self.latency_ms, 2),
            "is_refusal": self.is_refusal,
        }


def _clause_number(citation) -> str:
    """Prefer the section label ("Clause 8.2") over the clause type ("Termination")."""

    return (citation.section or citation.clause_type or "").strip()


def _check(result: AnswerResult, is_refusal: bool) -> List[str]:
    violations: List[str] = []

    if not (result.answer or "").strip():
        violations.append("answer is empty")

    if result.confidence is None:
        violations.append("confidence score is missing")

    if result.faithfulness is None:
        violations.append("faithfulness report is missing")

    if is_refusal:
        # A refusal carries no assertions, so it needs no supporting evidence.
        return violations

    if not result.citations:
        violations.append("answer carries no supporting evidence")
        return violations

    if not any(_clause_number(citation) for citation in result.citations):
        violations.append("no citation identifies a clause or section")

    if not any(citation.page for citation in result.citations):
        violations.append("no citation identifies a page number")

    return violations


def build_response(result: AnswerResult, strict: bool = True) -> LegalResponse:
    """Turn an ``AnswerResult`` into the validated section 12 payload.

    With ``strict=True`` a missing required element raises. With ``strict=False`` the
    same problems are appended to ``warnings`` instead, which is what the benchmark
    harnesses want: they need to count contract failures across a run rather than
    stop at the first one.
    """

    is_refusal = REFUSAL_MARKER in (result.answer or "").lower()
    violations = _check(result, is_refusal)

    if violations and strict:
        raise ContractViolation(violations)

    evidence = [
        EvidenceReference(
            clause_type=citation.clause_type,
            clause_number=_clause_number(citation),
            page=citation.page or None,
            quote=citation.quote,
            support_score=citation.support_score,
            chunk_id=citation.chunk_id,
            contract_id=citation.contract_id,
        )
        for citation in result.citations
    ]

    grounding = validate_grounding(
        result.answer,
        evidence_map=None,
        faithfulness=result.faithfulness,
        require_citations=result.plan.strategy in CITATION_MARKER_STRATEGIES,
    )

    return LegalResponse(
        query=result.query,
        answer=result.answer,
        confidence=result.confidence.score if result.confidence else 0.0,
        confidence_band=result.confidence.band if result.confidence else "low",
        evidence=evidence,
        intent=result.plan.intent.value,
        strategy=result.plan.strategy.value,
        generator=result.generator,
        faithfulness=result.faithfulness.score if result.faithfulness else 0.0,
        faithful=bool(result.faithfulness and result.faithfulness.supported),
        grounding=grounding,
        warnings=list(result.warnings) + violations,
        latency_ms=result.latency_ms,
        is_refusal=is_refusal,
    )


def render(response: LegalResponse) -> str:
    """Human-readable rendering, in the shape of the section 12 worked example."""

    lines = [response.answer.strip(), ""]
    lines.append(f"Confidence: {response.confidence:.2f} ({response.confidence_band})")

    if response.evidence:
        lines.append("Evidence:")
        for reference in response.evidence:
            lines.append(f"  {reference.display} - {reference.quote}")
    else:
        lines.append("Evidence: none retrieved")

    return "\n".join(lines)
