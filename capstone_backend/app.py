import datetime
import logging
import os
import uuid

from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from rag import service
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
        top_k = payload.get("topK")

        if not question:
            return jsonify({"error": "No question provided"}), 400

        if doc_id and not _ensure_indexed(doc_id):
            return jsonify({"error": "Document not found"}), 404

        result = service.get_engine().answer(question, contract_id=doc_id, top_k=top_k)

        return jsonify(result.to_dict())

    except Exception as error:
        logger.exception("Question answering failed")
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

    return jsonify(
        {
            "status": "ok",
            "llm_provider": config.llm_provider,
            "llm_configured": service.get_engine().llm_available(),
            "top_k": config.top_k_context,
        }
    )


if __name__ == "__main__":
    app.run(port=5001, debug=True)
