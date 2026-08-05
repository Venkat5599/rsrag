# kiran_retrieval

**Owner: Kiran** — the clause-aware legal knowledge base and hybrid retrieval stack.

Covers the responsibilities listed for Kiran in section 15 of
`LegalEase_v3_Hybrid_RAG_Integration.pdf`: clause chunking, metadata extraction,
embeddings, FAISS, BM25, hybrid retrieval, cross encoder reranker.

## Modules

| File | Spec section | Responsibility |
|---|---|---|
| `clause_chunker.py` | 6 | Segments contract text into whole clauses, not fixed windows. Detects numbered, article and all-caps headings, tracks page numbers, infers clause type. |
| `metadata_store.py` | 7, 8 | The chunk metadata envelope plus inverted indexes on clause type, contract, page and entity. Supplies the pre-retrieval candidate filter. |
| `embeddings.py` | 5 | `BAAI/bge-large-en-v1.5` encoding, unit-normalised, cached per process. |
| `vector_index.py` | 5 | FAISS `IndexFlatIP`. Inner product equals cosine because vectors are normalised. |
| `bm25_index.py` | 5, 8 | Okapi BM25 (k1=1.5, b=0.75), incremental, filter-aware. |
| `reranker.py` | 10 | `BAAI/bge-reranker-large` cross encoder over the top 25 candidates. |
| `hybrid_retriever.py` | 8, 9, 10 | The full pipeline and the six-signal fusion ranking. |
| `knowledge_base.py` | 4 | Orchestrates chunk to metadata to embeddings to both indexes, and exposes retrieval. |
| `retrieval_benchmark.py` | integration | Recall@k, precision@1 and MRR for baseline vs hybrid vs hybrid+rerank. |

## Pipeline

```
contract text
  -> clause segmentation        clause_chunker
  -> metadata extraction        metadata_store
  -> embedding generation       embeddings
  -> FAISS + BM25 indexes       vector_index, bm25_index

query
  -> metadata filter            metadata_store.candidates
  -> dense search + BM25 search vector_index, bm25_index
  -> candidate pool (25)        hybrid_retriever.candidates
  -> cross encoder rerank       reranker
  -> six-signal fusion          hybrid_retriever.retrieve
  -> top 5 clauses
```

## The ranking function

Spec section 9 asks for more than cosine similarity. `RetrievalSignals` holds the
weights, and every component score is written back onto the returned chunk under
`metadata["signals"]`, so any ranking can be explained after the fact.

| Signal | Weight | Source |
|---|---|---|
| Cross encoder | 0.32 | Reranker score, squashed to 0-1 |
| Semantic | 0.28 | Dense cosine from the embedding index |
| BM25 | 0.18 | Normalised sparse score |
| Clause importance | 0.08 | `rag.clause_taxonomy.clause_importance` |
| Heading match | 0.08 | Query against clause type and section |
| Entity match | 0.06 | Query mentions of stored entities |

The weights are constructor arguments, so a sweep is a one-line change.

## Interface with Venkata's half

`KnowledgeBase` satisfies the `rag.retrieval.ClauseRetriever` protocol
(`rag/retrieval.py`). That single `retrieve()` method is the entire contract between
the two halves, so either side can be replaced independently.

## Two behaviours worth knowing

**Clause filters boost, they do not restrict.** `clause_taxonomy.match_clause_types`
guesses clause types from keywords and is often wrong. "Can either party walk away
from the contract early?" matches the keyword "party" and produces the Parties
filter, when the answer is in the Termination clause. Obeying that filter literally
made the correct clause unreachable. So a filter only narrows the candidate pool
when the filtered set is still at least `candidate_pool_size` chunks, and otherwise
survives as a small scoring bonus. See `MetadataStore.candidates`.

**Page markers are read as footers.** A standalone "Page 12" labels the text
*above* it, unless it is on the very first line, in which case it is a running
header and labels the text below. Getting this backwards misattributes every
citation by one page.

## Optional dependencies

Every module runs without `faiss`, `numpy` or `sentence-transformers` installed, by
falling back to a pure-Python path behind the same API. The fallbacks exist so the
test suite runs anywhere; they are not a substitute for the real stack. Check which
is live via `KnowledgeBase.describe()` or `GET /health`:

```json
{"embedding_backend": "hashed_bag_of_words", "vector_backend": "python", "reranker_backend": "lexical_rerank"}
```

means nothing is installed. After `pip install -r requirements.txt` those read
`BAAI/bge-large-en-v1.5`, `faiss` and `BAAI/bge-reranker-large`.

## Running

```bash
python -m pytest kiran_retrieval/tests -q
python -m kiran_retrieval.retrieval_benchmark
```
