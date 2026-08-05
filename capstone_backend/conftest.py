"""Root pytest configuration shared by all three suites.

Test layout:
    tests/                      integration - service wiring, Flask routes, evaluation
    kiran_retrieval/tests/      retrieval    - chunking, indexes, hybrid retrieval, rerank
    venkata_answering/tests/    answering    - routing, QA, grounding, citations, confidence

Fixtures live here so an owner suite can be run on its own
(``python -m pytest kiran_retrieval/tests -q``) without depending on the other half.
"""

import os
import sys

import pytest

BACKEND_ROOT = os.path.abspath(os.path.dirname(__file__))

if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

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

SAMPLE_CONTRACT_TEXT = """MASTER SERVICES AGREEMENT

1. PARTIES
This Agreement is entered into between Acme Corporation and Globex Limited on the
terms set out below. Each party represents that it has full authority to enter into
this Agreement.

Page 1

8.2 Termination for Convenience
Either party may terminate this Agreement for convenience upon thirty (30) days prior
written notice to the other party. Termination does not relieve either party of
obligations accrued before the effective date of termination.

Page 12

11.3 Limitation of Liability
The aggregate liability of either party shall not exceed the fees paid in the twelve
months preceding the claim. Neither party is liable for indirect or consequential
damages arising out of this Agreement.

Page 15

14.1 Governing Law
This Agreement shall be governed by the laws of the State of Delaware, without regard
to its conflict of laws principles.

Page 18
"""


@pytest.fixture
def sample_clauses():
    return SAMPLE_CLAUSES


@pytest.fixture
def sample_chunks():
    return chunks_from_clause_map(SAMPLE_CLAUSES, "contract-1", "msa.pdf")


@pytest.fixture
def retriever(sample_chunks):
    return InMemoryClauseRetriever(sample_chunks)


@pytest.fixture
def contract_text():
    return SAMPLE_CONTRACT_TEXT


@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def empty_chunk():
    return RetrievedChunk(chunk_id="x", contract_id="contract-1", text="")
