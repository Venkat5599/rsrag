import pytest

from rag.retrieval import InMemoryClauseRetriever

from venkata_answering.answer_contract import ContractViolation, build_response, render
from venkata_answering.answer_engine import NO_EVIDENCE_ANSWER, AnswerEngine


@pytest.fixture
def result(retriever, config):
    return AnswerEngine(retriever, config).answer("What is the notice period for termination?")


def test_response_carries_every_section_twelve_element(result):
    payload = build_response(result).to_dict()

    assert payload["answer"]
    assert payload["confidence"] >= 0.0
    assert payload["evidence"]
    assert any(payload["clause_numbers"])
    assert any(payload["pages"])


def test_clause_number_prefers_the_section_label(result):
    response = build_response(result)

    assert any(
        reference.clause_number.startswith("Clause") or reference.clause_number == "Preamble"
        for reference in response.evidence
    )


def test_evidence_display_follows_the_worked_example(result):
    reference = build_response(result).evidence[0]

    assert reference.display.endswith(f"Page {reference.page}")


def test_render_shows_the_answer_confidence_and_evidence(result):
    text = render(build_response(result))

    assert "Confidence:" in text
    assert "Evidence:" in text


def test_a_missing_citation_raises_in_strict_mode(result):
    result.citations = []

    with pytest.raises(ContractViolation) as error:
        build_response(result)

    assert "no supporting evidence" in str(error.value)


def test_a_missing_citation_becomes_a_warning_in_lenient_mode(result):
    result.citations = []
    response = build_response(result, strict=False)

    assert any("supporting evidence" in warning for warning in response.warnings)


def test_a_missing_confidence_score_raises(result):
    result.confidence = None

    with pytest.raises(ContractViolation) as error:
        build_response(result)

    assert "confidence score is missing" in str(error.value)


def test_a_citation_without_a_page_raises(result):
    for citation in result.citations:
        citation.page = 0

    with pytest.raises(ContractViolation) as error:
        build_response(result)

    assert "page number" in str(error.value)


def test_a_refusal_needs_no_evidence(config):
    result = AnswerEngine(InMemoryClauseRetriever([]), config).answer("Anything at all?")
    response = build_response(result)

    assert response.is_refusal
    assert response.answer == NO_EVIDENCE_ANSWER
    assert response.evidence == []


def test_grounding_is_attached_to_the_response(result):
    response = build_response(result)

    assert response.grounding is not None
    assert response.grounding.to_dict()["citation_coverage"] >= 0.0


def test_non_generative_strategies_are_not_required_to_emit_markers(result):
    response = build_response(result)

    assert response.grounding.uncited_sentences == []
