from rag.prompt_builder import SYSTEM_PROMPT, build_grounded_prompt
from rag.schemas import QueryIntent


def test_prompt_contains_evidence_labels(sample_chunks):
    prompt = build_grounded_prompt("Summarise the agreement", QueryIntent.SUMMARIZATION, sample_chunks)

    assert "[E1]" in prompt.user
    assert prompt.evidence_map["E1"].chunk_id == sample_chunks[0].chunk_id
    assert prompt.system == SYSTEM_PROMPT


def test_prompt_declares_citation_contract(sample_chunks):
    prompt = build_grounded_prompt("Explain the risks", QueryIntent.RISK_EXPLANATION, sample_chunks)

    assert "[E<number>]" in prompt.user
    assert "Identify the risks" in prompt.user


def test_prompt_respects_context_budget(sample_chunks):
    prompt = build_grounded_prompt(
        "Summarise the agreement", QueryIntent.SUMMARIZATION, sample_chunks, max_context_chars=120
    )

    assert len(prompt.evidence_map) < len(sample_chunks)


def test_prompt_without_evidence_is_explicit():
    prompt = build_grounded_prompt("Any question", QueryIntent.FACT_LOOKUP, [])

    assert "No evidence retrieved." in prompt.user
    assert prompt.evidence_map == {}
