from rag import service
from rag.retrieval import InMemoryClauseRetriever


def setup_function():
    service.reset()


def teardown_function():
    service.reset()


def test_index_document_registers_chunks():
    document = {
        "docId": "doc-1",
        "filename": "nda.pdf",
        "clauses": {
            "Confidentiality": {
                "span": "Each party shall keep the disclosing party information confidential.",
                "score": 0.8,
                "page": 4,
            }
        },
    }

    assert service.index_document(document) == 1
    assert service.is_indexed("doc-1")


def test_document_without_identifier_is_skipped():
    assert service.index_document({"clauses": {}}) == 0


def test_engine_answers_after_indexing():
    service.index_document(
        {
            "docId": "doc-2",
            "clauses": {
                "Governing Law": {
                    "span": "This Agreement is governed by the laws of Singapore.",
                    "score": 0.9,
                    "page": 9,
                }
            },
        }
    )

    result = service.get_engine().answer("What law governs this agreement?", contract_id="doc-2")

    assert "Singapore" in result.answer


def test_custom_retriever_can_be_injected():
    custom = InMemoryClauseRetriever()
    service.set_retriever(custom)

    assert service.get_retriever() is custom
