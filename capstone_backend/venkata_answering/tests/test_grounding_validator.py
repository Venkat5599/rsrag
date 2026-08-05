from rag.schemas import FaithfulnessReport, QueryIntent

from venkata_answering.grounding_validator import (
    summarise,
    validate_grounding,
    validate_prompt_response,
)
from venkata_answering.prompt_builder import build_grounded_prompt

CITED = "Either party may terminate on thirty days written notice [E1]."
UNCITED = "Either party may terminate on thirty days written notice."
REFUSAL = "The provided contract evidence does not answer this question."


def _evidence_map(sample_chunks):
    return build_grounded_prompt(
        "Can either party terminate?", QueryIntent.FACT_LOOKUP, sample_chunks
    ).evidence_map


def _supported(score=0.8):
    return FaithfulnessReport(score=score, supported=True, threshold=0.55)


def test_a_fully_cited_answer_is_valid(sample_chunks):
    report = validate_grounding(CITED, _evidence_map(sample_chunks), _supported())

    assert report.valid
    assert report.citation_coverage == 1.0
    assert report.violations == []


def test_an_uncited_answer_is_flagged(sample_chunks):
    report = validate_grounding(UNCITED, _evidence_map(sample_chunks), _supported())

    assert not report.valid
    assert report.citation_coverage == 0.0
    assert "no citation markers" in " ".join(report.violations)


def test_partial_coverage_is_reported(sample_chunks):
    answer = f"{CITED} The agreement is governed by Delaware law."
    report = validate_grounding(answer, _evidence_map(sample_chunks), _supported())

    assert not report.valid
    assert report.factual_sentences == 2
    assert report.cited_sentences == 1
    assert report.citation_coverage == 0.5
    assert len(report.uncited_sentences) == 1


def test_a_fabricated_citation_is_caught(sample_chunks):
    answer = "Either party may terminate on thirty days written notice [E99]."
    report = validate_grounding(answer, _evidence_map(sample_chunks), _supported())

    assert not report.valid
    assert report.invalid_markers == ["E99"]


def test_a_refusal_is_grounded_by_definition(sample_chunks):
    report = validate_grounding(REFUSAL, _evidence_map(sample_chunks))

    assert report.valid
    assert report.citation_coverage == 1.0
    assert report.factual_sentences == 0


def test_a_refusal_with_a_fabricated_citation_is_still_invalid(sample_chunks):
    report = validate_grounding(f"{REFUSAL} [E42]", _evidence_map(sample_chunks))

    assert not report.valid
    assert report.invalid_markers == ["E42"]


def test_an_empty_answer_is_invalid():
    report = validate_grounding("")

    assert not report.valid
    assert report.violations == ["answer is empty"]


def test_unfaithful_answers_are_flagged_even_when_cited(sample_chunks):
    unsupported = FaithfulnessReport(
        score=0.1,
        supported=False,
        threshold=0.55,
        unsupported_statements=["Either party may terminate on thirty days written notice"],
    )
    report = validate_grounding(CITED, _evidence_map(sample_chunks), unsupported)

    assert not report.valid
    assert report.unsupported_statements
    assert "faithfulness" in " ".join(report.violations)


def test_extractive_strategies_are_not_required_to_emit_markers(sample_chunks):
    report = validate_grounding(
        UNCITED, _evidence_map(sample_chunks), _supported(), require_citations=False
    )

    assert report.valid
    assert report.uncited_sentences == []


def test_markers_are_not_validated_when_no_evidence_map_is_supplied():
    report = validate_grounding(CITED, None, _supported())

    assert report.invalid_markers == []


def test_validate_prompt_response_uses_the_prompt_evidence_map(sample_chunks):
    prompt = build_grounded_prompt(
        "Can either party terminate?", QueryIntent.FACT_LOOKUP, sample_chunks
    )

    assert validate_prompt_response(prompt, CITED, _supported()).valid


def test_report_serialises_every_field(sample_chunks):
    payload = validate_grounding(CITED, _evidence_map(sample_chunks), _supported()).to_dict()

    for key in ("valid", "citation_coverage", "invalid_markers", "violations"):
        assert key in payload


def test_summarise_aggregates_across_reports(sample_chunks):
    evidence_map = _evidence_map(sample_chunks)
    reports = [
        validate_grounding(CITED, evidence_map, _supported()),
        validate_grounding(UNCITED, evidence_map, _supported()),
    ]

    summary = summarise(reports)

    assert summary["reports"] == 2
    assert summary["valid_ratio"] == 0.5


def test_summarise_handles_no_reports():
    assert summarise([])["reports"] == 0
