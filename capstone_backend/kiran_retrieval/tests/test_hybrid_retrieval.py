"""Tests for reranking, hybrid fusion and the knowledge base as a whole."""

import pytest

from kiran_retrieval.hybrid_retriever import RetrievalSignals
from kiran_retrieval.knowledge_base import KnowledgeBase, build_knowledge_base
from kiran_retrieval.reranker import CrossEncoderReranker, _squash
from rag.retrieval import ClauseRetriever

TERMINATION_QUERY = "Can either party terminate the agreement?"


@pytest.fixture
def knowledge_base(contract_text):
    base = build_knowledge_base()
    base.index_text(contract_text, "contract-1", "msa.pdf")
    return base


# ----- CrossEncoderReranker ---------------------------------------------------


def test_reranker_orders_the_relevant_clause_first(sample_chunks):
    reranked = CrossEncoderReranker("missing-model-for-tests").rerank(
        TERMINATION_QUERY, sample_chunks
    )

    assert reranked[0].clause_type == "Termination for Convenience"


def test_reranker_writes_a_nonzero_rerank_score(sample_chunks):
    reranked = CrossEncoderReranker("missing-model-for-tests").rerank(
        TERMINATION_QUERY, sample_chunks
    )

    assert reranked[0].rerank_score > 0.0
    assert all(0.0 <= chunk.rerank_score <= 1.0 for chunk in reranked)


def test_reranker_does_not_mutate_the_input_chunks(sample_chunks):
    CrossEncoderReranker("missing-model-for-tests").rerank(TERMINATION_QUERY, sample_chunks)

    assert all(chunk.rerank_score == 0.0 for chunk in sample_chunks)


def test_reranker_truncates_to_top_k(sample_chunks):
    reranked = CrossEncoderReranker("missing-model-for-tests").rerank(
        TERMINATION_QUERY, sample_chunks, top_k=2
    )

    assert len(reranked) == 2


def test_reranker_handles_no_candidates():
    assert CrossEncoderReranker("missing-model-for-tests").rerank(TERMINATION_QUERY, []) == []


def test_reranker_reports_the_lexical_fallback():
    reranker = CrossEncoderReranker("missing-model-for-tests")

    if not reranker.is_model_backed:
        assert reranker.backend_name() == "lexical_rerank"


def test_squash_maps_logits_into_the_unit_range():
    assert _squash(0.5) == 0.5
    assert 0.0 < _squash(-8.0) < 0.5
    assert 0.5 < _squash(8.0) < 1.0
    assert _squash(-100000.0) == pytest.approx(0.0, abs=1e-6)


# ----- RetrievalSignals -------------------------------------------------------


def test_signal_weights_sum_to_one():
    assert RetrievalSignals().total() == pytest.approx(1.0)


def test_signal_weights_are_reportable():
    weights = RetrievalSignals().to_dict()

    assert set(weights) == {
        "semantic",
        "bm25",
        "cross_encoder",
        "clause_importance",
        "heading_match",
        "entity_match",
    }


# ----- KnowledgeBase and HybridRetriever --------------------------------------


def test_knowledge_base_satisfies_the_clause_retriever_protocol(knowledge_base):
    assert isinstance(knowledge_base, ClauseRetriever)


def test_indexing_produces_one_chunk_per_clause(knowledge_base):
    assert knowledge_base.size == 4


def test_retrieval_ranks_the_termination_clause_first(knowledge_base):
    hits = knowledge_base.retrieve(TERMINATION_QUERY, top_k=3)

    assert hits[0].clause_type == "Termination for Convenience"


def test_retrieved_chunks_carry_the_correct_page(knowledge_base):
    hit = knowledge_base.retrieve(TERMINATION_QUERY, top_k=1)[0]

    assert hit.page == 12


def test_every_signal_is_recorded_for_explainability(knowledge_base):
    hit = knowledge_base.retrieve(TERMINATION_QUERY, top_k=1)[0]

    assert set(hit.metadata["signals"]) == set(RetrievalSignals().to_dict())


def test_dense_sparse_and_rerank_scores_are_all_populated(knowledge_base):
    hit = knowledge_base.retrieve(TERMINATION_QUERY, top_k=1)[0]

    assert hit.dense_score > 0.0
    assert hit.sparse_score > 0.0
    assert hit.rerank_score > 0.0


def test_retrieval_scores_are_bounded_and_descending(knowledge_base):
    hits = knowledge_base.retrieve(TERMINATION_QUERY, top_k=4)
    scores = [hit.retrieval_score for hit in hits]

    assert all(0.0 <= score <= 1.0 for score in scores)
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_the_result_size(knowledge_base):
    assert len(knowledge_base.retrieve(TERMINATION_QUERY, top_k=2)) == 2


def test_clause_filters_promote_the_matching_clause(knowledge_base):
    query = "What does the agreement say?"

    unfiltered = knowledge_base.retrieve(query, top_k=4)
    filtered = knowledge_base.retrieve(query, top_k=4, clause_filters=["Governing Law"])

    def rank_of(hits, clause_type):
        return next(
            index for index, hit in enumerate(hits) if hit.clause_type == clause_type
        )

    assert rank_of(filtered, "Governing Law") < rank_of(unfiltered, "Governing Law")


def test_a_clause_filter_cannot_outrank_a_much_stronger_match(knowledge_base):
    """The filter is a keyword guess, so it boosts but must not override.

    ``clause_taxonomy.match_clause_types`` guesses from keywords and is often wrong,
    so the bonus is deliberately too small to overturn a large retrieval gap.
    """

    hits = knowledge_base.retrieve(
        "Can either party terminate the agreement?",
        top_k=4,
        clause_filters=["Governing Law"],
    )

    assert hits[0].clause_type == "Termination for Convenience"


def test_a_wrong_clause_filter_never_hides_the_correct_clause(knowledge_base):
    """Regression: "party" makes match_clause_types return the Parties filter.

    Obeying that literally reduced the candidate pool to the Parties clause alone
    and the Termination clause became unreachable, so the query returned nothing
    useful at all.
    """

    hits = knowledge_base.retrieve(
        "Can either party walk away from the contract early?",
        top_k=5,
        clause_filters=["Parties"],
    )

    assert any(hit.clause_type == "Termination for Convenience" for hit in hits)


def test_contract_scope_excludes_other_contracts(knowledge_base):
    assert knowledge_base.retrieve(TERMINATION_QUERY, contract_id="another-contract") == []


def test_empty_query_retrieves_nothing(knowledge_base):
    assert knowledge_base.retrieve("   ") == []


def test_empty_knowledge_base_retrieves_nothing():
    assert build_knowledge_base().retrieve(TERMINATION_QUERY) == []


def test_candidate_pool_is_capped_at_the_configured_size(knowledge_base):
    pool = knowledge_base.retriever.candidates(TERMINATION_QUERY)

    assert len(pool) <= knowledge_base.config.candidate_pool_size


def test_index_document_prefers_the_extracted_clause_map(sample_clauses):
    base = build_knowledge_base()
    indexed = base.index_document(
        {"docId": "doc-9", "filename": "msa.pdf", "clauses": sample_clauses}
    )

    assert indexed == len(sample_clauses)
    assert base.retrieve(TERMINATION_QUERY, top_k=1)[0].page == 12


def test_index_document_falls_back_to_chunking_the_summary(contract_text):
    base = build_knowledge_base()
    indexed = base.index_document(
        {"docId": "doc-9", "filename": "msa.pdf", "summary": contract_text}
    )

    assert indexed == 4


def test_index_document_without_an_id_indexes_nothing():
    assert build_knowledge_base().index_document({"filename": "msa.pdf"}) == 0


def test_add_alias_keeps_it_swappable_with_the_baseline_retriever(sample_chunks):
    base = build_knowledge_base()
    base.add(sample_chunks)

    assert base.size == len(sample_chunks)


def test_clear_empties_the_knowledge_base(knowledge_base):
    knowledge_base.clear()

    assert knowledge_base.size == 0
    assert knowledge_base.retrieve(TERMINATION_QUERY) == []


def test_describe_reports_the_active_backends(knowledge_base):
    description = knowledge_base.describe()

    for key in ("embedding_backend", "vector_backend", "reranker_backend", "indexed_chunks"):
        assert key in description


def test_blank_chunks_are_not_indexed(sample_chunks, empty_chunk):
    base = build_knowledge_base()

    assert base.index_chunks([empty_chunk]) == 0
    assert base.index_chunks(sample_chunks) == len(sample_chunks)


def test_custom_signal_weights_change_the_ranking(contract_text):
    bm25_only = KnowledgeBase(
        signals=RetrievalSignals(
            semantic=0.0,
            bm25=1.0,
            cross_encoder=0.0,
            clause_importance=0.0,
            heading_match=0.0,
            entity_match=0.0,
        )
    )
    bm25_only.index_text(contract_text, "contract-1")

    hit = bm25_only.retrieve(TERMINATION_QUERY, top_k=1)[0]

    assert hit.retrieval_score == pytest.approx(hit.sparse_score)
