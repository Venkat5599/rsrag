from rag.confidence_engine import band_for, score_answer
from rag.faithfulness import verify_answer
from rag.router import build_plan
from rag.citation_engine import build_citations


def test_confidence_is_bounded(sample_chunks):
    plan = build_plan("What is the notice period for termination?")
    answer = "Termination requires thirty days prior written notice."
    citations = build_citations(answer, sample_chunks)
    report = verify_answer(answer, sample_chunks, threshold=0.4)

    confidence = score_answer(plan, sample_chunks, citations, report, answer_model_score=0.8)

    assert 0.0 <= confidence.score <= 1.0
    assert set(confidence.signals) == {
        "retrieval",
        "answer_model",
        "faithfulness",
        "citation_support",
        "intent",
        "strategy_prior",
    }


def test_unsupported_answer_lowers_confidence(sample_chunks):
    plan = build_plan("What is the notice period for termination?")
    supported = verify_answer(
        "Termination requires thirty days prior written notice.", sample_chunks, threshold=0.4
    )
    unsupported = verify_answer(
        "The supplier must deliver steel to Rotterdam every Friday.", sample_chunks, threshold=0.9
    )

    high = score_answer(plan, sample_chunks, [], supported, answer_model_score=0.8)
    low = score_answer(plan, sample_chunks, [], unsupported, answer_model_score=0.8)

    assert low.score < high.score


def test_no_evidence_scores_low():
    plan = build_plan("What is the notice period?")
    report = verify_answer("Some claim.", [], threshold=0.5)
    confidence = score_answer(plan, [], [], report, answer_model_score=0.0)

    assert confidence.band == "low"


def test_band_boundaries():
    assert band_for(0.95) == "high"
    assert band_for(0.5) == "medium"
    assert band_for(0.1) == "low"
