from venkata_answering.faithfulness import verify_answer


def test_supported_answer_passes(sample_chunks):
    answer = "Either party may terminate this Agreement for convenience upon thirty days prior written notice."
    report = verify_answer(answer, sample_chunks, threshold=0.4)

    assert report.supported
    assert report.score >= 0.4


def test_unsupported_answer_is_flagged(sample_chunks):
    answer = "The supplier must deliver forty tonnes of steel to the port of Rotterdam every Friday."
    report = verify_answer(answer, sample_chunks, threshold=0.6)

    assert not report.supported
    assert report.unsupported_statements


def test_refusal_is_treated_as_faithful(sample_chunks):
    report = verify_answer(
        "The provided contract evidence does not answer this question.", sample_chunks, threshold=0.6
    )

    assert report.supported
    assert report.score == 1.0


def test_missing_evidence_is_unsupported():
    report = verify_answer("Some claim about the contract terms.", [], threshold=0.5)

    assert not report.supported
    assert report.score == 0.0


def test_empty_answer_is_unsupported(sample_chunks):
    report = verify_answer("", sample_chunks, threshold=0.5)

    assert not report.supported
