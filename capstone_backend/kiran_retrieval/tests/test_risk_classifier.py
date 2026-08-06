"""Tests for the section 7 ``risk`` field.

The field was in the schema and permanently empty. These tests pin down the two
things that matter: that it is populated at index time, and that the level it
reports is explainable rather than asserted.
"""

from kiran_retrieval import risk_classifier
from kiran_retrieval.knowledge_base import build_knowledge_base
from rag.retrieval import chunks_from_clause_map
from rag.schemas import RetrievedChunk

UNCAPPED_INDEMNITY = (
    "The Supplier shall indemnify and hold harmless the Customer against all claims, "
    "with unlimited liability and no cap, and the Customer may terminate without cause "
    "at its sole discretion."
)

SEVERABILITY = (
    "If any provision of this Agreement is held to be invalid, the remaining provisions "
    "shall continue in full force and effect."
)

CAPPED_LIABILITY = (
    "The aggregate liability of either party shall not exceed the fees paid in the "
    "twelve months preceding the claim."
)


# ----- assess ----------------------------------------------------------------


def test_uncapped_indemnity_is_high_risk():
    assessment = risk_classifier.assess(UNCAPPED_INDEMNITY, "Indemnification")

    assert assessment.level == risk_classifier.HIGH
    assert assessment.score >= risk_classifier.HIGH_THRESHOLD


def test_severability_is_low_risk():
    assessment = risk_classifier.assess(SEVERABILITY, "Severability")

    assert assessment.level == risk_classifier.LOW


def test_every_assessment_reports_the_markers_behind_it():
    assessment = risk_classifier.assess(UNCAPPED_INDEMNITY, "Indemnification")

    assert assessment.factors
    assert "unlimited liability" in assessment.factors
    assert "sole discretion" in assessment.factors


def test_an_explicit_cap_lowers_the_score():
    capped = risk_classifier.assess(CAPPED_LIABILITY, "Limitation of Liability")
    uncapped = risk_classifier.assess(
        "Neither party's liability under this Agreement is subject to any cap.",
        "Limitation of Liability",
    )

    assert capped.score < uncapped.score
    assert any(factor.startswith("mitigated:") for factor in capped.factors)


def test_a_short_cure_period_is_a_risk_marker():
    assessment = risk_classifier.assess(
        "A party in breach shall have five (5) days notice to cure the breach.",
        "Termination for Convenience",
    )

    assert any("short cure" in factor for factor in assessment.factors)


def test_empty_text_is_low_risk_with_no_score():
    assessment = risk_classifier.assess("", "Indemnification")

    assert assessment.level == risk_classifier.LOW
    assert assessment.score == 0.0


def test_scores_stay_inside_the_unit_interval():
    piled_on = " ".join([UNCAPPED_INDEMNITY] * 4) + " irrevocable perpetual exclusive"
    assessment = risk_classifier.assess(piled_on, "Indemnification")

    assert 0.0 <= assessment.score <= 1.0


# ----- annotate --------------------------------------------------------------


def test_annotate_populates_the_chunk():
    chunk = RetrievedChunk(
        chunk_id="c-1",
        contract_id="contract-1",
        text=UNCAPPED_INDEMNITY,
        clause_type="Indemnification",
    )

    risk_classifier.annotate(chunk)

    assert chunk.risk == risk_classifier.HIGH
    assert chunk.risk_score > 0.0
    assert chunk.risk_factors


def test_annotate_respects_a_risk_set_upstream():
    chunk = RetrievedChunk(
        chunk_id="c-1",
        contract_id="contract-1",
        text=UNCAPPED_INDEMNITY,
        clause_type="Indemnification",
        risk="low",
    )

    risk_classifier.annotate(chunk)

    assert chunk.risk == "low"


# ----- wired into indexing ---------------------------------------------------


def test_indexing_never_leaves_risk_empty(sample_clauses):
    base = build_knowledge_base()
    base.index_chunks(chunks_from_clause_map(sample_clauses, "contract-1", "msa.pdf"))

    stored = base.store.chunks()

    assert stored
    assert all(chunk.risk for chunk in stored)
    assert all(chunk.risk in {"high", "medium", "low"} for chunk in stored)


def test_indexing_stores_the_embedding_on_the_chunk(sample_clauses):
    base = build_knowledge_base()
    base.index_chunks(chunks_from_clause_map(sample_clauses, "contract-1", "msa.pdf"))

    stored = base.store.chunks()

    width = len(stored[0].embedding)

    assert width > 0
    assert all(len(chunk.embedding) == width for chunk in stored)


def test_to_dict_hides_the_vector_but_reports_its_width(sample_clauses):
    base = build_knowledge_base()
    base.index_chunks(chunks_from_clause_map(sample_clauses, "contract-1", "msa.pdf"))

    chunk = base.store.chunks()[0]

    assert "embedding" not in chunk.to_dict()
    assert chunk.to_dict()["embedding_dim"] == len(chunk.embedding)
    assert chunk.to_dict(include_embedding=True)["embedding"]
