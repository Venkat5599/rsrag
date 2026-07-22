from rag.entity_lookup import extract_entities, format_entity_answer, requested_labels


def test_requested_labels_from_query():
    assert "ORG" in requested_labels("Which company signed the agreement?")
    assert "DATE" in requested_labels("What date does the term start?")
    assert "MONEY" in requested_labels("How much is the fee?")


def test_stored_entities_are_used_when_model_missing(sample_chunks, config):
    entities, model_used = extract_entities(sample_chunks, config.ner_model)

    assert entities
    assert isinstance(model_used, bool)


def test_label_filter_restricts_output(sample_chunks, config):
    entities, _ = extract_entities(sample_chunks, config.ner_model, ["ORG"])

    assert entities
    assert all(entity.label == "ORG" for entity in entities)


def test_answer_formatting_groups_labels(sample_chunks, config):
    entities, _ = extract_entities(sample_chunks, config.ner_model, ["ORG"])
    answer = format_entity_answer(entities)

    assert answer.startswith("ORG:")


def test_empty_entities_produce_explicit_message():
    assert "No matching entities" in format_entity_answer([])
