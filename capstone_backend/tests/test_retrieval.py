from rag.retrieval import ClauseRetriever, InMemoryClauseRetriever, chunks_from_text


def test_chunks_carry_metadata(sample_chunks):
    assert len(sample_chunks) == 4

    chunk = next(c for c in sample_chunks if c.clause_type == "Termination for Convenience")

    assert chunk.page == 12
    assert chunk.section == "Clause 8.2"
    assert chunk.entities[0].label == "DATE"


def test_retriever_ranks_relevant_clause_first(retriever):
    results = retriever.retrieve("termination notice period", top_k=3)

    assert results
    assert results[0].clause_type == "Termination for Convenience"


def test_contract_filter_excludes_other_documents(retriever):
    assert retriever.retrieve("termination", contract_id="unknown") == []


def test_in_memory_retriever_satisfies_protocol(retriever):
    assert isinstance(retriever, ClauseRetriever)


def test_chunks_from_text_splits_blocks():
    text = "First clause body with enough words to survive filtering.\n\nSecond clause body with enough words too."
    chunks = chunks_from_text(text, contract_id="c9")

    assert len(chunks) == 2
    assert all(chunk.contract_id == "c9" for chunk in chunks)


def test_empty_retriever_returns_no_results():
    assert InMemoryClauseRetriever().retrieve("anything") == []
