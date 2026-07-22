import importlib
import sys
import types

import pytest

pytest.importorskip("flask")
pytest.importorskip("pymongo")
pytest.importorskip("pytesseract")
pytest.importorskip("sentence_transformers")


class FakeCollection:
    def __init__(self):
        self.documents = []

    def insert_one(self, document):
        self.documents.append(document)
        return types.SimpleNamespace(inserted_id="fake-object-id")

    def find_one(self, query):
        doc_id = query.get("docId")
        for document in self.documents:
            if document.get("docId") == doc_id:
                return document
        return None

    def find(self):
        return self

    def sort(self, *args, **kwargs):
        return list(self.documents)


class FakeDatabase:
    def __init__(self):
        self.collection = FakeCollection()

    def __getitem__(self, name):
        return self.collection


class FakeMongoClient:
    def __init__(self, *args, **kwargs):
        self.admin = types.SimpleNamespace(command=lambda *a, **k: {"ok": 1})
        self.database = FakeDatabase()

    def __getitem__(self, name):
        return self.database


@pytest.fixture
def flask_app(monkeypatch):
    monkeypatch.setattr("pymongo.MongoClient", FakeMongoClient)

    for module_name in ("app",):
        sys.modules.pop(module_name, None)

    module = importlib.import_module("app")
    module.app.config.update(TESTING=True)

    from rag import service

    service.reset()
    service.index_document(
        {
            "docId": "doc-1",
            "filename": "msa.pdf",
            "clauses": {
                "Governing Law": {
                    "span": "This Agreement shall be governed by the laws of the State of Delaware.",
                    "score": 0.8,
                    "page": 18,
                }
            },
        }
    )

    yield module

    service.reset()
    sys.modules.pop("app", None)


def test_ask_requires_question(flask_app):
    response = flask_app.app.test_client().post("/ask", json={})

    assert response.status_code == 400


def test_ask_returns_grounded_payload(flask_app):
    response = flask_app.app.test_client().post(
        "/ask", json={"question": "What law governs this agreement?", "docId": "doc-1"}
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert "Delaware" in payload["answer"]
    assert payload["plan"]["strategy"]
    assert payload["confidence"]["band"] in {"low", "medium", "high"}


def test_ask_plan_exposes_routing(flask_app):
    response = flask_app.app.test_client().post(
        "/ask/plan", json={"question": "Summarise this agreement"}
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["intent"] == "summarization"


def test_health_reports_configuration(flask_app):
    payload = flask_app.app.test_client().get("/health").get_json()

    assert payload["status"] == "ok"
    assert "llm_provider" in payload
