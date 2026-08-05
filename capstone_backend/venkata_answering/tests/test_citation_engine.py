from venkata_answering.citation_engine import build_citations, extract_marker_ids, render_citations
from venkata_answering.prompt_builder import build_grounded_prompt
from rag.schemas import QueryIntent


def test_marker_extraction():
    assert extract_marker_ids("Termination requires notice [E2]. Governing law is Delaware [E3].") == ["E2", "E3"]


def test_citations_follow_markers(sample_chunks):
    prompt = build_grounded_prompt("Summarise", QueryIntent.SUMMARIZATION, sample_chunks)
    answer = "Termination requires thirty days notice [E4]."

    citations = build_citations(answer, sample_chunks, prompt.evidence_map)

    assert len(citations) == 1
    assert citations[0].chunk_id == prompt.evidence_map["E4"].chunk_id


def test_citations_fall_back_to_evidence_order(sample_chunks):
    citations = build_citations("Termination requires thirty days written notice.", sample_chunks)

    assert citations
    assert all(citation.quote for citation in citations)


def test_citations_expose_locations(sample_chunks):
    citations = build_citations("Governing law is Delaware.", sample_chunks)
    rendered = render_citations(citations)

    assert "page" in rendered


def test_no_evidence_produces_no_citations():
    assert build_citations("anything", []) == []
    assert render_citations([]) == "No supporting evidence was retrieved."
