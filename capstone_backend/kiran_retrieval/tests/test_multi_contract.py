"""Tests for retrieval spanning more than one contract.

Spec section 11 routes clause comparison to the grounded LLM, and a comparison is
meaningless unless the evidence actually comes from both agreements. The failure
these tests exist to prevent is the quiet one: retrieving from a single contract
and answering as though both had been read.
"""

import pytest

from kiran_retrieval.hybrid_retriever import _allocate_across_contracts
from kiran_retrieval.knowledge_base import build_knowledge_base
from kiran_retrieval.metadata_store import MetadataStore
from rag.retrieval import InMemoryClauseRetriever, chunks_from_clause_map

QUERY = "Compare the termination clauses of these agreements."

SECOND_CONTRACT = {
    "Termination for Convenience": {
        "span": (
            "Either party may terminate this Agreement for convenience upon ninety (90) "
            "days prior written notice to the other party."
        ),
        "score": 0.8,
        "page": 4,
        "section": "Clause 3.1",
    },
    "Governing Law": {
        "span": "This Agreement shall be governed by the laws of the State of New York.",
        "score": 0.75,
        "page": 9,
        "section": "Clause 12.2",
    },
}


@pytest.fixture
def two_contracts(sample_clauses):
    return (
        chunks_from_clause_map(sample_clauses, "contract-1", "msa.pdf"),
        chunks_from_clause_map(SECOND_CONTRACT, "contract-2", "sow.pdf"),
    )


@pytest.fixture
def store(two_contracts):
    first, second = two_contracts
    built = MetadataStore()
    built.add_many(first)
    built.add_many(second)
    return built


# ----- MetadataStore ---------------------------------------------------------


def test_scope_is_the_union_of_the_named_contracts(store, two_contracts):
    first, second = two_contracts

    scope = store.scope_ids(contract_ids=["contract-1", "contract-2"])

    assert len(scope) == len(first) + len(second)


def test_a_single_contract_id_still_scopes_to_one_contract(store):
    scope = store.scope_ids(contract_id="contract-2")

    assert {store.chunk(cid).contract_id for cid in scope} == {"contract-2"}


def test_no_scope_means_the_whole_store(store):
    assert store.scope_ids() == set(store.chunk_ids)


def test_scope_names_dedupes_and_keeps_order():
    assert MetadataStore.scope_names("a", ["b", "a", "", "c"]) == ["a", "b", "c"]


# ----- allocation ------------------------------------------------------------


def test_allocation_gives_every_contract_a_slot(two_contracts):
    first, second = two_contracts
    # Deliberately front-load contract-1 so a plain top-k would exclude contract-2.
    ordered = list(first) + list(second)

    selected = _allocate_across_contracts(ordered, ["contract-1", "contract-2"], 4)

    assert {chunk.contract_id for chunk in selected} == {"contract-1", "contract-2"}
    assert len(selected) == 4


def test_allocation_falls_back_to_global_order_when_a_contract_runs_out(two_contracts):
    first, second = two_contracts
    ordered = list(first) + list(second[:1])

    selected = _allocate_across_contracts(ordered, ["contract-1", "contract-2"], 5)

    assert len(selected) == 5
    assert len({chunk.chunk_id for chunk in selected}) == 5


# ----- end to end ------------------------------------------------------------


def test_hybrid_retrieval_returns_evidence_from_both_contracts(two_contracts):
    base = build_knowledge_base()
    first, second = two_contracts
    base.index_chunks(first)
    base.index_chunks(second)

    chunks = base.retrieve(QUERY, top_k=4, contract_ids=["contract-1", "contract-2"])

    assert {chunk.contract_id for chunk in chunks} == {"contract-1", "contract-2"}


def test_hybrid_retrieval_still_honours_a_single_contract(two_contracts):
    base = build_knowledge_base()
    first, second = two_contracts
    base.index_chunks(first)
    base.index_chunks(second)

    chunks = base.retrieve(QUERY, top_k=4, contract_id="contract-2")

    assert chunks
    assert {chunk.contract_id for chunk in chunks} == {"contract-2"}


def test_baseline_retriever_also_spans_both_contracts(two_contracts):
    first, second = two_contracts
    retriever = InMemoryClauseRetriever(list(first) + list(second))

    chunks = retriever.retrieve(QUERY, top_k=4, contract_ids=["contract-1", "contract-2"])

    assert {chunk.contract_id for chunk in chunks} == {"contract-1", "contract-2"}
