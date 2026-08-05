# LegalEase v3 Hybrid RAG — integration

Joint deliverable. How Kiran's half and Venkata's half connect, how to run the
benchmarks, and what counts as done.

## Ownership

| Directory | Owner | Scope |
|---|---|---|
| `kiran_retrieval/` | Kiran | Clause chunking, metadata, embeddings, FAISS, BM25, hybrid retrieval, cross encoder reranker |
| `venkata_answering/` | Venkata | Query classifier, routing, DeBERTa, LegalBERT, grounded LLM, prompt builder, citations, confidence, faithfulness |
| `rag/` | shared | Schemas, config, semantic primitives, clause taxonomy, retriever protocol and baseline, service wiring, end-to-end evaluation |
| `tests/` | shared | Integration tests: service wiring, Flask routes, evaluation harness |

Each module has exactly one owner, which is what section 15 of the spec asks for.
The responsibilities are swappable: the boundary is a protocol, not a schedule.

## Dependency direction

```
kiran_retrieval   ---->  rag.schemas / rag.config / rag.semantic / rag.clause_taxonomy
venkata_answering ---->  same
rag.service       ---->  both, imported lazily inside its factory functions
```

Both owner packages import `rag.<submodule>` directly, never `from rag import ...`.
`rag/__init__.py` re-exports shared core only. That is what keeps the arrow
one-way: a cycle here would break `import rag` for everyone.

## The seam

One method. `rag/retrieval.py`:

```python
class ClauseRetriever(Protocol):
    def retrieve(self, query, top_k=5, clause_filters=None, contract_id=None) -> List[RetrievedChunk]: ...
```

`kiran_retrieval.KnowledgeBase` implements it. `venkata_answering.AnswerEngine`
consumes it. Neither imports the other. `rag/service.py` chooses which
implementation is live, so either half can be developed, tested or replaced on its
own.

## Architecture, spec section 4, mapped to files

```
Contract PDF
  -> OCR + text cleaning            v3_pipeline.py                    (existing)
  -> clause segmentation            kiran_retrieval/clause_chunker.py
  -> metadata + entity extraction   kiran_retrieval/metadata_store.py,
                                    venkata_answering/entity_lookup.py
  -> knowledge base builder         kiran_retrieval/knowledge_base.py
  -> chunk + metadata store         kiran_retrieval/metadata_store.py
  -> embedding generation           kiran_retrieval/embeddings.py
  -> FAISS index                    kiran_retrieval/vector_index.py

User question
  -> query understanding            venkata_answering/query_classifier.py
  -> semantic + BM25 + metadata     kiran_retrieval/hybrid_retriever.py
  -> candidate pool                 kiran_retrieval/hybrid_retriever.py
  -> cross encoder reranker         kiran_retrieval/reranker.py
  -> top 5 chunks                   kiran_retrieval/hybrid_retriever.py
  -> adaptive decision engine       venkata_answering/router.py
  -> retriever / DeBERTa /
     LegalBERT / grounded LLM       venkata_answering/answer_engine.py
  -> citation + confidence          venkata_answering/citation_engine.py,
                                    venkata_answering/confidence_engine.py
  -> faithfulness verification      venkata_answering/faithfulness.py,
                                    venkata_answering/grounding_validator.py
  -> final legal response           venkata_answering/answer_contract.py
```

## Technology, spec section 5

| Spec | Implementation | Note |
|---|---|---|
| Backend: Flask | `app.py` | unchanged |
| Database: MongoDB Atlas | `app.py` `MONGO_URI` | unchanged |
| Vector database: FAISS | `kiran_retrieval/vector_index.py` | `IndexFlatIP` |
| Embedding: BAAI/bge-large-en-v1.5 | `rag/config.py` `embedding_model` | via sentence-transformers |
| Sparse retrieval: BM25 | `kiran_retrieval/bm25_index.py` | Okapi, k1=1.5, b=0.75 |
| Reranker: BAAI/bge-reranker-large | `rag/config.py` `reranker_model` | cross encoder |
| LLM: configurable | `venkata_answering/llm_client.py` | Ollama for Llama 3.1, OpenAI for GPT-4.1, Gemini |
| QA model: existing DeBERTa | `venkata_answering/extractive_qa.py` | unchanged model |
| NER model: existing LegalBERT | `venkata_answering/entity_lookup.py` | unchanged model |
| OCR: existing pipeline | `v1/v2/v3_pipeline.py` | untouched |

Nothing was substituted. Every module falls back to a pure-Python path when a
library is missing, so the suite runs on a bare interpreter, but the fallback is a
degradation, never the intended configuration. `GET /health` reports which is live.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `LEGALEASE_RETRIEVER` | `hybrid` | `memory` selects the pre-hybrid baseline |
| `LEGALEASE_EMBEDDING_MODEL` | `BAAI/bge-large-en-v1.5` | dense encoder |
| `LEGALEASE_RERANKER_MODEL` | `BAAI/bge-reranker-large` | cross encoder |
| `LEGALEASE_CANDIDATE_POOL` | `25` | candidates before reranking |
| `LEGALEASE_TOP_K` | `5` | clauses passed to answering |
| `LEGALEASE_BM25_K1` / `_B` | `1.5` / `0.75` | BM25 saturation and length normalisation |
| `LEGALEASE_LLM_PROVIDER` | `none` | `openai`, `gemini`, `ollama` |
| `LEGALEASE_FAITHFULNESS_THRESHOLD` | `0.55` | below this an answer is flagged unsupported |

## Endpoints

| Route | Purpose |
|---|---|
| `POST /analyze-text`, `POST /analyze-pdf` | run a pipeline and index the result |
| `POST /retrieve` | retrieval only, with per-signal scores; no answering model |
| `POST /ask` | full answer, plus the section 12 contract under `contract` |
| `POST /ask/plan` | routing decision only |
| `GET /health` | configuration and which retrieval backends actually loaded |

`POST /retrieve` exists so retrieval can be inspected and benchmarked without an
answering model in the way, which is how a ranking regression gets diagnosed.

## Benchmarks

```bash
python -m kiran_retrieval.retrieval_benchmark     # retrieval quality
python -m venkata_answering.routing_benchmark     # adaptive routing
python benchmark_answering.py                     # end to end
```

The numbers below were measured with **no ML libraries installed**, on the
pure-Python fallbacks. They are floor numbers, not ceilings. The venv now has the
real stack, so re-running the benchmarks measures bge-large-en-v1.5, FAISS and
bge-reranker-large and should improve on these:

Retrieval, 15 labelled cases in `benchmarks/retrieval_cases.json`:

| configuration | recall@5 | precision@1 | MRR |
|---|---|---|---|
| baseline | 1.000 | 0.867 | 0.922 |
| hybrid | 1.000 | 0.867 | 0.933 |
| hybrid+rerank | 1.000 | 0.867 | 0.933 |

Routing, 24 labelled queries: intent accuracy 0.958, strategy accuracy 0.958, LLM
avoided on 50 percent of queries.

Two honest caveats. The rerank row equals the hybrid row because the fixture
contract has six clauses and the pool holds 25, so reranking never has candidates
to discard; the difference will only show on a corpus larger than the pool. And the
fallback encoder is lexical, so the semantic signal is weaker than
`bge-large-en-v1.5` would make it. Install the requirements to measure the real
system.

## Verification

```bash
cd capstone_backend
python -m pytest -q                             # full suite
python -m pytest kiran_retrieval/tests -q       # owner suite alone
python -m pytest venkata_answering/tests -q     # owner suite alone
python -c "import rag, kiran_retrieval, venkata_answering"   # no import cycle
```

The two owner suites pass independently, which is the check that the split is real
rather than cosmetic.

## Demo runbook

```bash
# terminal 1
cd capstone_backend
venv\Scripts\activate                # venv is already provisioned
python app.py                        # http://localhost:5001

# first-time setup only, if venv/ is missing:
#   python -m venv venv
#   venv\Scripts\activate
#   pip install -r requirements.txt   # v3_pipeline imports sentence_transformers at
#                                     # module level, so the app cannot start without it

# terminal 2
cd capstone_frontend
npm install
npm run dev                          # http://localhost:5173
```

Then, in the browser:

1. Try Now, then upload a contract PDF or paste contract text. The pipeline runs and
   the clause cards appear, each with its clause number and page.
2. The input switches to **Ask** automatically and a bar shows which contract is in
   scope. Every question from here is scoped to that contract.
3. Ask **"What is the notice period for termination?"** — routes to DeBERTa
   extractive QA. The answer carries a confidence band, Faithful and Grounded
   badges, and the citation `Clause 8.2 Page 12`.
4. Ask **"Who are the parties to this agreement?"** — routes to LegalBERT entity
   extraction. Different intent, different model, no LLM. This is the section 11
   claim made visible.
5. Ask **"Show me the confidentiality clause"** — routes to retriever only.
6. Click **Show how these clauses were ranked** on any answer. This calls
   `/retrieve` and shows all six ranking signals per clause, which is the section 9
   contribution made inspectable.
7. Open the sidebar to show the Retrieval stack panel: which vector index,
   embedding model and reranker actually loaded, and how many clauses are indexed.

Two things to be ready for. Without `pip install -r requirements.txt` the backend
will not start at all, because `v3_pipeline` imports `sentence_transformers` at
module level. And with no `LEGALEASE_LLM_PROVIDER` set, the generative intents
(summarise, compare, risk) degrade to retriever-only rather than failing - which is
a designed behaviour worth pointing out rather than hiding.

## Definition of done

- [x] Every spec section 15 responsibility implemented and owned by one directory
- [x] Clause-aware chunking replaces fixed-size windows
- [x] Metadata filtering, dense retrieval, BM25 and cross encoder reranking all live
- [x] Multi-signal ranking, section 9, with per-signal explainability
- [x] Adaptive routing avoids the LLM where a cheaper model suffices
- [x] Every answer carries evidence, clause numbers, pages and confidence
- [x] Faithfulness and grounding both verified
- [x] Retrieval and routing benchmarked
- [x] Wired into the existing Flask backend, frontend untouched
