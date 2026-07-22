import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rag.config import load_config
from rag.retrieval import InMemoryClauseRetriever, chunks_from_clause_map
from rag.schemas import RetrievedChunk

SAMPLE_CLAUSES = {
    "Termination for Convenience": {
        "span": (
            "Either party may terminate this Agreement for convenience upon thirty (30) days "
            "prior written notice to the other party."
        ),
        "score": 0.82,
        "page": 12,
        "section": "Clause 8.2",
        "entities": [{"text": "thirty (30) days", "label": "DATE", "score": 0.91}],
    },
    "Governing Law": {
        "span": "This Agreement shall be governed by the laws of the State of Delaware.",
        "score": 0.77,
        "page": 18,
        "section": "Clause 14.1",
        "entities": [{"text": "Delaware", "label": "GPE", "score": 0.88}],
    },
    "Parties": {
        "span": "This Agreement is entered into between Acme Corporation and Globex Limited.",
        "score": 0.9,
        "page": 1,
        "section": "Preamble",
        "entities": [
            {"text": "Acme Corporation", "label": "ORG", "score": 0.95},
            {"text": "Globex Limited", "label": "ORG", "score": 0.93},
        ],
    },
    "Limitation of Liability": {
        "span": (
            "The aggregate liability of either party shall not exceed the fees paid in the "
            "twelve months preceding the claim."
        ),
        "score": 0.71,
        "page": 15,
        "section": "Clause 11.3",
    },
}


@pytest.fixture
def sample_chunks():
    return chunks_from_clause_map(SAMPLE_CLAUSES, "contract-1", "msa.pdf")


@pytest.fixture
def retriever(sample_chunks):
    return InMemoryClauseRetriever(sample_chunks)


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def empty_chunk():
    return RetrievedChunk(chunk_id="x", contract_id="contract-1", text="")
