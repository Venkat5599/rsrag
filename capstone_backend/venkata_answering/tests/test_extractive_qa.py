from venkata_answering.extractive_qa import answer_from_chunks


def test_answer_selects_relevant_span(sample_chunks, config):
    answer = answer_from_chunks("What is the notice period for termination?", sample_chunks, config.qa_model)

    assert "thirty" in answer.text.lower()
    assert answer.chunk is not None
    assert answer.score > 0.0


def test_answer_without_chunks_is_empty(config):
    answer = answer_from_chunks("Anything?", [], config.qa_model)

    assert answer.text == ""
    assert answer.chunk is None


def test_unrelated_question_still_returns_grounded_span(sample_chunks, config):
    answer = answer_from_chunks("What colour is the office building?", sample_chunks, config.qa_model)

    assert answer.text
    assert answer.chunk is not None
