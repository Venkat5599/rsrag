from rag.schemas import AnswerStrategy, QueryIntent

from venkata_answering.routing_benchmark import (
    LABELLED_QUERIES,
    NON_LLM_STRATEGY_VALUES,
    run_routing_benchmark,
)


def test_every_intent_is_represented_in_the_labelled_set():
    labelled = {intent for _, intent in LABELLED_QUERIES}

    assert labelled == set(QueryIntent)


def test_benchmark_scores_every_case():
    report = run_routing_benchmark()

    assert report.metrics["cases"] == len(LABELLED_QUERIES)
    assert len(report.outcomes) == len(LABELLED_QUERIES)


def test_routing_accuracy_clears_a_useful_bar():
    report = run_routing_benchmark()

    assert report.metrics["intent_accuracy"] >= 0.7
    assert report.metrics["strategy_accuracy"] >= 0.7


def test_per_intent_precision_and_recall_are_reported():
    report = run_routing_benchmark()

    for scores in report.per_intent.values():
        assert 0.0 <= scores["precision"] <= 1.0
        assert 0.0 <= scores["recall"] <= 1.0
        assert scores["support"] >= 0.0


def test_confusions_only_list_actual_mistakes():
    report = run_routing_benchmark()

    for expected, predicted, count in report.confusions:
        assert expected != predicted
        assert count >= 1


def test_the_system_avoids_the_llm_on_the_cheap_intents():
    report = run_routing_benchmark()

    assert report.metrics["llm_avoidance_rate"] > 0.0
    assert report.metrics["llm_avoidance_target"] > 0.0


def test_without_an_llm_nothing_routes_to_the_generative_strategy():
    report = run_routing_benchmark(llm_available=False)
    predicted = {outcome.predicted_strategy for outcome in report.outcomes}

    assert AnswerStrategy.GROUNDED_LLM.value not in predicted
    assert predicted <= NON_LLM_STRATEGY_VALUES


def test_degraded_mode_still_answers_every_query():
    report = run_routing_benchmark(llm_available=False)

    assert report.metrics["llm_avoidance_rate"] == 1.0


def test_a_custom_query_set_is_scored():
    report = run_routing_benchmark(
        [("Show me the confidentiality clause.", QueryIntent.CLAUSE_RETRIEVAL)]
    )

    assert report.metrics["cases"] == 1
    assert report.outcomes[0].intent_correct


def test_report_serialises_and_renders():
    report = run_routing_benchmark()
    payload = report.to_dict()

    assert set(payload) == {"metrics", "per_intent", "confusions", "cases"}
    assert "Adaptive routing benchmark" in report.render()
