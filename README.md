# LegalEase

LegalEase is a contract analysis platform that extracts clauses, identifies entities, generates
summaries, and answers grounded legal questions over processed contracts.

The system ships three analysis pipelines (v1, v2, v3) and a Hybrid Retrieval-Augmented
Intelligence layer that turns a processed contract into a searchable legal knowledge base and
routes each user question to the cheapest answering strategy that can answer it correctly.

## Architecture

```
Contract PDF
  -> OCR and text cleaning
  -> clause segmentation
  -> metadata and entity extraction
  -> knowledge base builder
  -> chunk store and metadata store
  -> embeddings and index

User question
  -> query understanding and intent classification
  -> hybrid retrieval (dense, sparse, metadata filter)
  -> cross encoder reranking
  -> adaptive decision engine
  -> retriever, DeBERTa, LegalBERT, or grounded LLM
  -> citation engine
  -> confidence engine
  -> faithfulness verification
  -> final legal response
```

## Repository layout

```
capstone_backend/
  app.py                  Flask API (analysis routes, /ask, /ask/plan, /health)
  v1_pipeline.py          rule-based baseline pipeline
  v2_pipeline.py          machine learning pipeline
  v3_pipeline.py          production pipeline (OCR, clauses, NER, summary)
  evaluation_metrics.py   pipeline comparison harness
  cuad_loader.py          CUAD dataset loader
  rag/                    hybrid retrieval-augmented intelligence layer
  tests/                  test suite for the rag layer and API routes
capstone_frontend/        React (Vite) client
```

## The rag package

| Module | Responsibility |
| --- | --- |
| `config.py` | environment driven configuration for models, provider, thresholds |
| `schemas.py` | typed contracts: `RetrievedChunk`, `QueryPlan`, `Citation`, `AnswerResult` |
| `clause_taxonomy.py` | clause vocabulary, clause importance weights, entity label aliases |
| `semantic.py` | embedding similarity with a deterministic lexical fallback |
| `models.py` | lazy DeBERTa and LegalBERT loaders with graceful degradation |
| `retrieval.py` | `ClauseRetriever` protocol plus an in-memory reference retriever |
| `query_classifier.py` | seven-way intent classification (rules blended with prototypes) |
| `router.py` | adaptive decision engine mapping intent to answering strategy |
| `extractive_qa.py` | DeBERTa answering over retrieved evidence |
| `entity_lookup.py` | LegalBERT entity answering over retrieved evidence |
| `prompt_builder.py` | grounded prompt construction with numbered evidence |
| `llm_client.py` | configurable LLM transport (OpenAI, Gemini, Ollama) |
| `grounded_llm.py` | grounded generation with refusal on missing evidence |
| `citation_engine.py` | clause, section, and page citations with support scores |
| `confidence_engine.py` | six-signal confidence score and reliability band |
| `faithfulness.py` | statement level verification of the answer against evidence |
| `answer_engine.py` | orchestrator for the full question answering path |
| `service.py` | process wide engine, index, and retriever injection point |

## Adaptive routing

| Intent | Strategy |
| --- | --- |
| Fact lookup | DeBERTa extractive QA |
| Clause retrieval | retriever only |
| Entity lookup | LegalBERT entity extraction |
| Clause comparison | grounded LLM |
| Summarization | grounded LLM |
| Risk explanation | grounded LLM |
| Legal reasoning | grounded LLM |

If the LLM provider is not configured, or a generation call fails, the router degrades to
evidence-only or extractive answering instead of failing the request. Every response carries the
strategy that produced it, its citations, a confidence band, and a faithfulness report.

## Retriever integration contract

The answering layer depends only on the `ClauseRetriever` protocol:

```python
class ClauseRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        clause_filters: Optional[Sequence[str]] = None,
        contract_id: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        ...
```

A FAISS, BM25, and cross encoder implementation is injected without touching the answering layer:

```python
from rag import service

service.set_retriever(HybridClauseRetriever(...))
```

`InMemoryClauseRetriever` is the reference implementation used until the hybrid index is wired in.
It scores candidates with semantic similarity, lexical overlap, heading match, entity match,
clause importance, and metadata filters, which mirrors the scoring signals of the hybrid retriever.

## API

| Method | Route | Purpose |
| --- | --- | --- |
| POST | `/analyze-text` | run a pipeline over pasted text and index the result |
| POST | `/analyze-pdf` | run a pipeline over an uploaded PDF and index the result |
| POST | `/ask` | answer a question over an indexed contract |
| POST | `/ask/plan` | return the routing decision without answering |
| GET | `/results` | list processed documents |
| GET | `/results/<docId>` | fetch one processed document |
| GET | `/health` | service and provider status |

Request:

```json
{ "question": "What is the notice period for termination?", "docId": "6f1c2a04", "topK": 5 }
```

Response:

```json
{
  "answer": "Either party may terminate for convenience upon thirty days prior written notice.",
  "plan": { "intent": "fact_lookup", "strategy": "deberta_extractive_qa" },
  "citations": [{ "clause_type": "Termination for Convenience", "section": "Clause 8.2", "page": 12 }],
  "confidence": { "score": 0.71, "band": "high" },
  "faithfulness": { "score": 0.83, "supported": true },
  "warnings": []
}
```

## Configuration

Copy `capstone_backend/.env.example` to `.env` and set the values you need. All keys have safe
defaults, and the service runs without an LLM provider by degrading to extractive answering.

`MONGO_URI` is required for persistence and defaults to a local MongoDB instance. Credentials must
never be committed to the repository.

## Setup

Backend:

```
cd capstone_backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The API serves on http://localhost:5001.

Frontend:

```
cd capstone_frontend
npm install
npm run dev
```

The client serves on http://localhost:5173.

## Tests

```
cd capstone_backend
python -m pytest
```

The suite covers intent classification, routing and degradation, retrieval ranking, prompt
construction, citation binding, faithfulness verification, confidence scoring, the orchestrator,
and the API routes. Tests run without model downloads and without network access.
