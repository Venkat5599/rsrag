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
  app.py                  Flask API (analysis routes, /retrieve, /ask, /ask/plan, /health)
  v1_pipeline.py          rule-based baseline pipeline
  v2_pipeline.py          machine learning pipeline
  v3_pipeline.py          production pipeline (OCR, clauses, NER, summary)
  evaluation_metrics.py   pipeline comparison harness
  cuad_loader.py          CUAD dataset loader
  kiran_retrieval/        OWNER: Kiran   knowledge base and hybrid retrieval
  venkata_answering/      OWNER: Venkata query understanding and grounded answering
  rag/                    shared core: schemas, config, taxonomy, protocol, wiring
  tests/                  integration tests (service, API routes, evaluation)
  INTEGRATION.md          how the two halves connect, benchmarks, definition of done
capstone_frontend/        React (Vite) client
```

## Ownership

Implementation is split per section 15 of `LegalEase_v3_Hybrid_RAG_Integration.pdf`,
one directory per owner, so every module has exactly one owner at any point.

| Directory | Owner | Scope |
| --- | --- | --- |
| `kiran_retrieval/` | Kiran | clause chunking, metadata extraction, embeddings, FAISS, BM25, hybrid retrieval, cross encoder reranker |
| `venkata_answering/` | Venkata | query classifier, DeBERTa, LegalBERT, grounded LLM, prompt builder, citation engine, confidence engine, faithfulness verification |
| `rag/` | shared | data contracts and the seam between the two |

Each package has its own README and its own test suite that passes on its own.
`capstone_backend/INTEGRATION.md` is the joint document. The split divides effort
rather than fixing assignments: the boundary is a protocol, so responsibilities can
be exchanged without touching the other half.

## The shared core

| Module | Responsibility |
| --- | --- |
| `config.py` | environment driven configuration for models, provider, thresholds |
| `schemas.py` | typed contracts: `RetrievedChunk`, `QueryPlan`, `Citation`, `AnswerResult` |
| `clause_taxonomy.py` | clause vocabulary, clause importance weights, entity label aliases |
| `semantic.py` | tokenisation, sentence splitting, similarity with a lexical fallback |
| `retrieval.py` | `ClauseRetriever` protocol, chunk builders, baseline retriever |
| `service.py` | process wide engine, index, and retriever selection |
| `evaluation.py` | end-to-end harness exercising both halves together |

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

`kiran_retrieval.KnowledgeBase` implements that protocol with FAISS, BM25 and a cross encoder, and
is selected by default. `rag/service.py` picks the backend from `LEGALEASE_RETRIEVER`, or you can
inject one directly:

```python
from kiran_retrieval import build_knowledge_base
from rag import service

service.set_retriever(build_knowledge_base())
```

`InMemoryClauseRetriever` remains as the pre-hybrid baseline, selected with
`LEGALEASE_RETRIEVER=memory`. It is what `kiran_retrieval/retrieval_benchmark.py` measures the
hybrid pipeline against, so the improvement is a number rather than a claim.

## API

| Method | Route | Purpose |
| --- | --- | --- |
| POST | `/analyze-text` | run a pipeline over pasted text and index the result |
| POST | `/analyze-pdf` | run a pipeline over an uploaded PDF and index the result |
| POST | `/retrieve` | hybrid retrieval only, with per-signal scores and no answering model |
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
python -m pytest -q                            # everything
python -m pytest kiran_retrieval/tests -q      # Kiran's suite alone
python -m pytest venkata_answering/tests -q    # Venkata's suite alone
```

The suite covers clause chunking, metadata filtering, embeddings, the vector and BM25 indexes,
cross encoder reranking, hybrid ranking, intent classification, routing and degradation, prompt
construction, citation binding, grounding validation, faithfulness verification, confidence
scoring, the response contract, the orchestrator, and the API routes. Tests run without model
downloads and without network access.

Each owner suite passes independently, which is what makes the split real rather than cosmetic.

Benchmarks:

```
python -m kiran_retrieval.retrieval_benchmark   # recall@k, precision@1, MRR
python -m venkata_answering.routing_benchmark   # intent and strategy accuracy
python benchmark_answering.py                   # end to end
```
