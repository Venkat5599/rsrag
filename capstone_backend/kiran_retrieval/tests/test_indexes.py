"""Tests for the three index layers: metadata, embeddings, vectors, BM25.

These deliberately assert the offline fallback behaviour, because that is the path
CI runs. Where a real library would change the result, the test asserts the property
that must hold either way (ordering, bounds, filtering) rather than a fixed number.
"""

import pytest

from kiran_retrieval.bm25_index import BM25Index
from kiran_retrieval.embeddings import EmbeddingBackend, cosine, hashed_vector, normalise
from kiran_retrieval.metadata_store import MetadataStore, describe
from kiran_retrieval.vector_index import VectorIndex

TERMINATION = "Either party may terminate this Agreement upon thirty days written notice."
LIABILITY = "The aggregate liability of either party shall not exceed the fees paid."
GOVERNING = "This Agreement shall be governed by the laws of the State of Delaware."


# ----- MetadataStore ----------------------------------------------------------


def test_store_indexes_chunks_by_clause_type(sample_chunks):
    store = MetadataStore()
    store.add_many(sample_chunks)

    assert len(store) == len(sample_chunks)
    assert "Governing Law" in store.clause_types()


def test_candidates_narrow_to_the_requested_clause_type(sample_chunks):
    store = MetadataStore()
    store.add_many(sample_chunks)

    candidates = store.candidates(clause_filters=["Governing Law"])

    assert len(candidates) == 1
    assert store.chunk(next(iter(candidates))).clause_type == "Governing Law"


def test_candidates_fall_back_to_the_full_scope_when_a_filter_matches_nothing(sample_chunks):
    store = MetadataStore()
    store.add_many(sample_chunks)

    assert store.candidates(clause_filters=["Nonexistent Clause"]) == set(store.chunk_ids)


def test_candidates_respect_the_contract_scope(sample_chunks):
    store = MetadataStore()
    store.add_many(sample_chunks)

    assert store.candidates(contract_id="contract-1") == set(store.chunk_ids)
    assert store.candidates(contract_id="other-contract") == set()


def test_candidates_match_on_entity_text(sample_chunks):
    store = MetadataStore()
    store.add_many(sample_chunks)

    candidates = store.candidates(entities=["Acme Corporation"])

    assert len(candidates) == 1
    assert store.chunk(next(iter(candidates))).clause_type == "Parties"


def test_describe_carries_the_section_seven_schema(sample_chunks):
    metadata = describe(sample_chunks[0]).to_dict()

    for field in ("chunk_id", "contract_id", "page", "section", "clause_type", "entities"):
        assert field in metadata


def test_clear_empties_every_index(sample_chunks):
    store = MetadataStore()
    store.add_many(sample_chunks)
    store.clear()

    assert len(store) == 0
    assert store.clause_types() == []
    assert store.candidates(clause_filters=["Parties"]) == set()


# ----- EmbeddingBackend -------------------------------------------------------


def test_hashed_vectors_are_unit_length():
    vector = hashed_vector(TERMINATION)

    assert pytest.approx(sum(value * value for value in vector), abs=1e-6) == 1.0


def test_hashed_vectors_are_deterministic():
    assert hashed_vector(TERMINATION) == hashed_vector(TERMINATION)


def test_similar_text_scores_higher_than_unrelated_text():
    query = hashed_vector("Can either party terminate the agreement?")
    related = hashed_vector(TERMINATION)
    unrelated = hashed_vector(GOVERNING)

    assert cosine(query, related) > cosine(query, unrelated)


def test_empty_text_yields_a_zero_vector():
    assert set(hashed_vector("")) == {0.0}


def test_normalise_handles_a_zero_vector():
    assert normalise([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_cosine_returns_zero_for_mismatched_dimensions():
    assert cosine([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


def test_backend_caches_repeated_text():
    backend = EmbeddingBackend("missing-model-for-tests")
    backend.encode([TERMINATION, TERMINATION, LIABILITY])

    assert backend.cache_size() == 2


def test_backend_reports_the_fallback_when_no_model_is_installed():
    backend = EmbeddingBackend("missing-model-for-tests")

    if not backend.is_semantic:
        assert backend.backend_name() == "hashed_bag_of_words"


def test_encoding_nothing_returns_nothing():
    assert EmbeddingBackend("missing-model-for-tests").encode([]) == []


# ----- VectorIndex ------------------------------------------------------------


def _index_with(texts):
    backend = EmbeddingBackend("missing-model-for-tests")
    index = VectorIndex(backend.dimensions)
    ids = [f"c{position}" for position in range(len(texts))]
    index.add(ids, backend.encode(texts))
    return backend, index


def test_vector_search_ranks_the_matching_clause_first():
    backend, index = _index_with([TERMINATION, LIABILITY, GOVERNING])

    hits = index.search(backend.encode_one("Which law governs the agreement?"), top_k=3)

    assert hits[0][0] == "c2"


def test_vector_search_honours_the_metadata_filter():
    backend, index = _index_with([TERMINATION, LIABILITY, GOVERNING])

    hits = index.search(backend.encode_one("terminate"), top_k=3, allowed_ids={"c1"})

    assert [chunk_id for chunk_id, _ in hits] == ["c1"]


def test_vector_search_with_an_empty_filter_returns_nothing():
    backend, index = _index_with([TERMINATION])

    assert index.search(backend.encode_one("terminate"), allowed_ids=set()) == []


def test_vector_index_rejects_a_wrong_sized_vector():
    index = VectorIndex(8)

    with pytest.raises(ValueError):
        index.add(["c0"], [[0.1, 0.2]])


def test_vector_index_rejects_mismatched_id_and_vector_counts():
    index = VectorIndex(2)

    with pytest.raises(ValueError):
        index.add(["c0", "c1"], [[1.0, 0.0]])


def test_vector_index_clear_empties_it():
    _, index = _index_with([TERMINATION, LIABILITY])
    index.clear()

    assert len(index) == 0
    assert index.search([0.0], top_k=3) == []


# ----- BM25Index --------------------------------------------------------------


def _bm25_with(documents):
    index = BM25Index()
    for chunk_id, text in documents:
        index.add(chunk_id, text)
    return index


def test_bm25_ranks_the_document_containing_the_query_terms():
    index = _bm25_with([("c0", TERMINATION), ("c1", LIABILITY), ("c2", GOVERNING)])

    hits = index.search("governed by the laws of Delaware")

    assert hits[0][0] == "c2"


def test_bm25_normalised_score_is_bounded_and_best_is_one():
    index = _bm25_with([("c0", TERMINATION), ("c1", GOVERNING)])

    hits = index.search("terminate this agreement")

    assert hits[0][2] == 1.0
    assert all(0.0 <= normalised <= 1.0 for _, _, normalised in hits)


def test_bm25_drops_documents_with_no_matching_term():
    index = _bm25_with([("c0", TERMINATION), ("c1", GOVERNING)])

    hits = index.search("Delaware")

    assert [chunk_id for chunk_id, _, _ in hits] == ["c1"]


def test_bm25_returns_nothing_for_an_unknown_query():
    index = _bm25_with([("c0", TERMINATION)])

    assert index.search("cryptocurrency mining rig") == []


def test_bm25_honours_the_metadata_filter():
    index = _bm25_with([("c0", TERMINATION), ("c1", GOVERNING)])

    hits = index.search("agreement", allowed_ids={"c1"})

    assert [chunk_id for chunk_id, _, _ in hits] == ["c1"]


def test_bm25_readding_the_same_id_replaces_it():
    index = _bm25_with([("c0", TERMINATION)])
    index.add("c0", GOVERNING)

    assert len(index) == 1
    assert index.search("Delaware")[0][0] == "c0"


def test_bm25_remove_drops_the_document():
    index = _bm25_with([("c0", TERMINATION), ("c1", GOVERNING)])

    assert index.remove("c0") is True
    assert index.remove("c0") is False
    assert len(index) == 1


def test_bm25_rarer_terms_carry_more_weight():
    index = _bm25_with(
        [("c0", "agreement " + GOVERNING), ("c1", "agreement text"), ("c2", "agreement text")]
    )

    assert index.idf("delawar") > index.idf("agreement")


def test_empty_bm25_index_returns_nothing():
    assert BM25Index().search("anything") == []
