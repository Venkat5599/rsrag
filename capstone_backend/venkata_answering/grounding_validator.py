"""Grounding validation.

``prompt_builder.OUTPUT_CONTRACT`` tells the model that every factual sentence must
end with an ``[E<n>]`` citation drawn from the supplied evidence. Nothing checked
whether the model complied. ``faithfulness`` measures whether a statement is
*supported* by the evidence; this module measures whether the answer is *attributed*
to it. Both can fail independently: a correct sentence with no marker is
unattributed, and a marker pointing at E9 when only E1 to E5 exist is a fabricated
citation, which is the more dangerous failure because it looks rigorous.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from rag.schemas import FaithfulnessReport, RetrievedChunk
from rag.semantic import split_sentences

from .citation_engine import CITATION_MARKER
from .prompt_builder import GroundedPrompt

MIN_FACTUAL_WORDS = 4
REFUSAL_MARKER = "does not answer this question"


@dataclass
class GroundingReport:
    valid: bool
    citation_coverage: float
    factual_sentences: int
    cited_sentences: int
    invalid_markers: List[str] = field(default_factory=list)
    uncited_sentences: List[str] = field(default_factory=list)
    unsupported_statements: List[str] = field(default_factory=list)
    violations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "citation_coverage": round(self.citation_coverage, 4),
            "factual_sentences": self.factual_sentences,
            "cited_sentences": self.cited_sentences,
            "invalid_markers": self.invalid_markers,
            "uncited_sentences": self.uncited_sentences,
            "unsupported_statements": self.unsupported_statements,
            "violations": self.violations,
        }


def _is_refusal(answer: str) -> bool:
    return REFUSAL_MARKER in (answer or "").lower()


def _factual_sentences(answer: str) -> List[str]:
    return [
        sentence
        for sentence in split_sentences(answer, min_words=MIN_FACTUAL_WORDS)
        if not sentence.strip().startswith("[")
    ]


def validate_grounding(
    answer: str,
    evidence_map: Optional[Dict[str, RetrievedChunk]] = None,
    faithfulness: Optional[FaithfulnessReport] = None,
    require_citations: bool = True,
    min_coverage: float = 1.0,
) -> GroundingReport:
    """Check an answer against the grounding contract.

    ``require_citations`` is False for the extractive and retriever-only strategies,
    which quote evidence directly and are never asked to emit markers. Their
    grounding is structural rather than textual, so only the faithfulness signal and
    marker validity apply.
    """

    answer = (answer or "").strip()
    known = set(evidence_map or {})
    violations: List[str] = []

    if not answer:
        return GroundingReport(
            valid=False,
            citation_coverage=0.0,
            factual_sentences=0,
            cited_sentences=0,
            violations=["answer is empty"],
        )

    used = [f"E{number}" for number in CITATION_MARKER.findall(answer)]
    invalid = sorted({marker for marker in used if known and marker not in known})

    if invalid:
        violations.append(
            f"answer cites evidence that was never supplied: {', '.join(invalid)}"
        )

    if _is_refusal(answer):
        # A refusal is fully grounded by definition: it asserts nothing.
        return GroundingReport(
            valid=not invalid,
            citation_coverage=1.0,
            factual_sentences=0,
            cited_sentences=0,
            invalid_markers=invalid,
            violations=violations,
        )

    sentences = _factual_sentences(answer)
    cited = [sentence for sentence in sentences if CITATION_MARKER.search(sentence)]
    uncited = [sentence for sentence in sentences if not CITATION_MARKER.search(sentence)]
    coverage = len(cited) / len(sentences) if sentences else 1.0

    if require_citations:
        if not used:
            violations.append("answer contains no citation markers")
        if coverage < min_coverage:
            violations.append(
                f"citation coverage {coverage:.2f} is below the required {min_coverage:.2f}"
            )

    unsupported = list(faithfulness.unsupported_statements) if faithfulness else []

    if faithfulness is not None and not faithfulness.supported:
        violations.append(
            f"faithfulness {faithfulness.score:.2f} is below the "
            f"{faithfulness.threshold:.2f} threshold"
        )

    return GroundingReport(
        valid=not violations,
        citation_coverage=coverage,
        factual_sentences=len(sentences),
        cited_sentences=len(cited),
        invalid_markers=invalid,
        uncited_sentences=uncited if require_citations else [],
        unsupported_statements=unsupported,
        violations=violations,
    )


def validate_prompt_response(
    prompt: GroundedPrompt,
    answer: str,
    faithfulness: Optional[FaithfulnessReport] = None,
) -> GroundingReport:
    """Convenience wrapper for the grounded-LLM path, which has the evidence map."""

    return validate_grounding(answer, prompt.evidence_map, faithfulness)


def summarise(reports: Sequence[GroundingReport]) -> Dict[str, Any]:
    if not reports:
        return {
            "reports": 0,
            "valid_ratio": 0.0,
            "mean_coverage": 0.0,
            "fabricated_citations": 0,
        }

    valid = sum(1 for report in reports if report.valid)
    coverage = sum(report.citation_coverage for report in reports) / len(reports)
    fabricated = sum(len(report.invalid_markers) for report in reports)

    return {
        "reports": len(reports),
        "valid_ratio": round(valid / len(reports), 4),
        "mean_coverage": round(coverage, 4),
        "fabricated_citations": fabricated,
    }
