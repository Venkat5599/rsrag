import datetime
import logging
import os
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Loaded before rag.config reads the environment, so LEGALEASE_LLM_* from the
# gitignored .env reach load_config(). Without this the .env file is inert and the
# grounded-LLM path silently stays unconfigured.
load_dotenv()

from rag import service
from rag.clause_taxonomy import match_clause_types
from venkata_answering.answer_contract import build_response
from v1_pipeline import run_pipeline as run_v1_pdf
from v1_pipeline import run_pipeline_on_text as run_v1
from v2_pipeline import run_pipeline as run_v2_pdf
from v2_pipeline import run_pipeline_on_text as run_v2
from v3_pipeline import run_pipeline as run_v3_pdf
from v3_pipeline import run_pipeline_on_text as run_v3

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/legalease")

client = MongoClient(
    MONGO_URI,
    serverSelectionTimeoutMS=5000,
    tls=MONGO_URI.startswith("mongodb+srv"),
)

try:
    client.admin.command("ping")
    logger.info("MongoDB connection established")
except PyMongoError as error:
    logger.error("MongoDB connection failed: %s", error)

db = client["legalease"]
results = db["results"]

PIPELINE_MAP = {
    "v1": run_v1,
    "v2": run_v2,
    "v3": run_v3,
}

PIPELINE_PDF_MAP = {
    "v1": run_v1_pdf,
    "v2": run_v2_pdf,
    "v3": run_v3_pdf,
}


def _persist(document):
    inserted = results.insert_one(document)
    document["_id"] = str(inserted.inserted_id)
    return document


def _build_document(result, filename, pipeline):
    clauses = result.get("clauses", {})

    return {
        "docId": str(uuid.uuid4()),
        "filename": filename,
        "pipeline": pipeline,
        "summary": result.get("summary", ""),
        "clauses": clauses,
        "clause_count": len(clauses),
        "timestamp": datetime.datetime.now(datetime.timezone.utc),
    }


def _requested_doc_ids(payload, doc_id=None):
    """Contract scope for a request, from ``docId`` and/or ``docIds``.

    Spec section 11 routes clause comparison to the grounded LLM, and a comparison
    spans more than one agreement, so the API has to accept a list. ``docId`` stays
    supported unchanged for every single-contract caller; when both are sent the
    scopes are merged, order preserved, duplicates dropped.
    """

    requested = []

    for candidate in [doc_id, *(payload.get("docIds") or [])]:
        cleaned = str(candidate or "").strip()
        if cleaned and cleaned not in requested:
            requested.append(cleaned)

    return requested


def _ensure_indexed(doc_id):
    if not doc_id:
        return False

    if service.is_indexed(doc_id):
        return True

    document = results.find_one({"docId": doc_id})

    if not document:
        return False

    return service.index_document(document) > 0


@app.route("/analyze-text", methods=["POST"])
def analyze_text():
    try:
        payload = request.json or {}
        text = payload.get("text")
        pipeline = payload.get("pipeline", "v3")

        if not text:
            return jsonify({"error": "No text provided"}), 400

        if pipeline not in PIPELINE_MAP:
            return jsonify({"error": "Invalid pipeline selected"}), 400

        result = PIPELINE_MAP[pipeline](text, "Text Input")

        if "error" in result:
            return jsonify(result), 400

        document = _build_document(result, "Text Input", pipeline)
        service.index_document(document)

        return jsonify(_persist(document))

    except Exception as error:
        logger.exception("Text analysis failed")
        return jsonify({"error": str(error)}), 500


@app.route("/analyze-pdf", methods=["POST"])
def analyze_pdf():
    try:
        uploaded = request.files.get("pdf")
        pipeline = request.form.get("pipeline", "v3")

        if not uploaded:
            return jsonify({"error": "No file uploaded"}), 400

        if pipeline not in PIPELINE_PDF_MAP:
            return jsonify({"error": "Invalid pipeline selected"}), 400

        result = PIPELINE_PDF_MAP[pipeline](uploaded.read(), uploaded.filename)

        if "error" in result:
            return jsonify(result), 400

        document = _build_document(result, uploaded.filename, pipeline)
        service.index_document(document)

        return jsonify(_persist(document))

    except Exception as error:
        logger.exception("PDF analysis failed")
        return jsonify({"error": str(error)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    try:
        payload = request.json or {}
        question = (payload.get("question") or "").strip()
        doc_id = (payload.get("docId") or "").strip() or None
        doc_ids = _requested_doc_ids(payload, doc_id)
        top_k = payload.get("topK")

        if not question:
            return jsonify({"error": "No question provided"}), 400

        for requested in doc_ids:
            if not _ensure_indexed(requested):
                return jsonify({"error": f"Document not found: {requested}"}), 404

        result = service.get_engine().answer(
            question,
            contract_id=doc_id,
            top_k=top_k,
            contract_ids=doc_ids,
        )
        payload = result.to_dict()

        # Spec section 12: answer, confidence, evidence, clause numbers, page
        # numbers. build_response validates that they are all actually present.
        # strict=False so a contract failure is reported in the payload rather than
        # turning into a 500 - the caller still gets the answer, plus the warning
        # that it did not meet the grounding contract.
        payload["contract"] = build_response(result, strict=False).to_dict()

        return jsonify(payload)

    except Exception as error:
        logger.exception("Question answering failed")
        return jsonify({"error": str(error)}), 500


@app.route("/retrieve", methods=["POST"])
def retrieve():
    """Hybrid retrieval on its own, with no answering model involved.

    This exists so retrieval can be inspected and benchmarked directly: each chunk
    comes back with its dense, sparse and cross-encoder scores plus the full signal
    breakdown, which is how a ranking is explained or a regression is diagnosed.
    """

    try:
        payload = request.json or {}
        question = (payload.get("question") or "").strip()
        doc_id = (payload.get("docId") or "").strip() or None
        doc_ids = _requested_doc_ids(payload, doc_id)
        top_k = payload.get("topK") or service.get_config().top_k_context
        clause_filters = payload.get("clauseFilters")
        include_embedding = bool(payload.get("includeEmbedding"))

        if not question:
            return jsonify({"error": "No question provided"}), 400

        for requested in doc_ids:
            if not _ensure_indexed(requested):
                return jsonify({"error": f"Document not found: {requested}"}), 404

        if clause_filters is None:
            clause_filters = match_clause_types(question)

        chunks = service.get_retriever().retrieve(
            question,
            top_k=int(top_k),
            clause_filters=clause_filters,
            contract_id=doc_id,
            contract_ids=doc_ids,
        )

        return jsonify(
            {
                "question": question,
                "clause_filters": clause_filters,
                "contract_ids": doc_ids,
                "count": len(chunks),
                "chunks": [
                    chunk.to_dict(include_embedding=include_embedding) for chunk in chunks
                ],
            }
        )

    except Exception as error:
        logger.exception("Retrieval failed")
        return jsonify({"error": str(error)}), 500


@app.route("/ask/plan", methods=["POST"])
def ask_plan():
    try:
        payload = request.json or {}
        question = (payload.get("question") or "").strip()

        if not question:
            return jsonify({"error": "No question provided"}), 400

        return jsonify(service.get_engine().plan(question).to_dict())

    except Exception as error:
        logger.exception("Query planning failed")
        return jsonify({"error": str(error)}), 500


@app.route("/results", methods=["GET"])
def get_results():
    documents = list(results.find().sort("timestamp", -1))

    return jsonify(
        [
            {
                "docId": document["docId"],
                "filename": document["filename"],
                "summary": document.get("summary", ""),
            }
            for document in documents
        ]
    )


@app.route("/results/<doc_id>", methods=["GET"])
def get_one(doc_id):
    document = results.find_one({"docId": doc_id})

    if not document:
        return jsonify({"error": "Not found"}), 404

    document["_id"] = str(document["_id"])
    return jsonify(document)


@app.route("/health", methods=["GET"])
def health():
    config = service.get_config()
    retriever = service.get_retriever()

    payload = {
        "status": "ok",
        "llm_provider": config.llm_provider,
        "llm_configured": service.get_engine().llm_available(),
        "top_k": config.top_k_context,
        "retriever": config.retriever_backend,
        "embedding_model": config.embedding_model,
        "reranker_model": config.reranker_model,
    }

    # The hybrid knowledge base reports which backends actually loaded, so a
    # deployment can tell FAISS-and-bge from the offline fallbacks at a glance
    # instead of discovering it from retrieval quality.
    if hasattr(retriever, "describe"):
        payload["retrieval"] = retriever.describe()

    return jsonify(payload)


if __name__ == "__main__":
    app.run(port=5001, debug=True)
