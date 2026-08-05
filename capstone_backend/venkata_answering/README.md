# venkata_answering

**Owner: Venkata** — query understanding, adaptive routing and grounded answering.

Covers the responsibilities listed for Venkata in section 15 of
`LegalEase_v3_Hybrid_RAG_Integration.pdf`: query classifier, DeBERTa integration,
LegalBERT integration, grounded LLM, prompt builder, citation engine, confidence
engine, faithfulness verification.

## Modules

| File | Spec section | Responsibility |
|---|---|---|
| `query_classifier.py` | 11 | Classifies a query into one of seven intents. Rule patterns (0.72) blended with semantic prototype similarity (0.28). |
| `router.py` | 11 | The adaptive decision engine: intent to answering strategy, with a confidence downgrade and an LLM-unavailable fallback. |
| `answer_engine.py` | 4, 11 | Orchestrates retrieve, route, answer, cite, score, verify. |
| `extractive_qa.py` | 11 | DeBERTa span extraction for fact lookup. |
| `entity_lookup.py` | 11 | LegalBERT entity extraction for entity lookup. |
| `models.py` | 5 | Loads and caches the DeBERTa and LegalBERT pipelines. |
| `prompt_builder.py` | 11 | The grounded prompt: system rules, intent instruction, output contract, numbered evidence. |
| `grounded_llm.py` | 11 | Calls the LLM under the grounding contract and degrades cleanly when it fails. |
| `llm_client.py` | 5 | Configurable provider: OpenAI, Gemini, or Ollama for Llama 3.1. |
| `citation_engine.py` | 12 | Turns evidence and `[E<n>]` markers into citations with clause and page. |
| `confidence_engine.py` | 12 | Six weighted signals into one score and a low/medium/high band. |
| `faithfulness.py` | 13 | Scores each statement against the evidence and flags the unsupported ones. |
| `grounding_validator.py` | 12, 13 | Checks the answer is *attributed*, not just supported. |
| `answer_contract.py` | 12 | Builds and validates the required response payload. |
| `routing_benchmark.py` | 11, integration | Scores the router against a labelled query set. |

## Adaptive routing

Spec section 11: the system does not always invoke an LLM.

| Intent | Strategy | Model |
|---|---|---|
| Fact lookup | Extractive QA | DeBERTa |
| Clause retrieval | Retriever only | none |
| Entity lookup | Entity extraction | LegalBERT |
| Clause comparison | Grounded LLM | configured LLM |
| Summarization | Grounded LLM | configured LLM |
| Risk explanation | Grounded LLM | configured LLM |
| Legal reasoning | Grounded LLM | configured LLM |

Two safety valves: an intent scoring below 0.25 confidence is downgraded from
generation to extraction, and when no LLM is configured every generative intent
falls back to a retrieval or extractive strategy rather than failing. That second
path is what keeps the system answering with `LEGALEASE_LLM_PROVIDER` unset.

Current measurement on the 24-query labelled set (`python -m
venkata_answering.routing_benchmark`): intent accuracy 0.958, strategy accuracy
0.958, LLM avoided on 50 percent of queries. The one confusion is
`legal_reasoning` read as `fact_lookup`.

## Support versus attribution

These are different failures and both are checked.

`faithfulness.py` asks whether a statement is *supported* by the evidence, by
similarity. `grounding_validator.py` asks whether it is *attributed* to it: does
every factual sentence carry an `[E<n>]` marker, and does every marker resolve to
evidence that was actually supplied. A sentence can be true and unattributed, and a
citation pointing at `[E9]` when only `E1` to `E5` exist is fabricated. The second
is the more dangerous failure, because it looks rigorous.

Marker checking only applies to the grounded-LLM strategy. The extractive,
retriever-only and entity strategies quote retrieved text directly, so their
grounding is structural and they are never asked to emit markers.

## The response contract

Spec section 12 requires answer, confidence, supporting evidence, clause numbers
and page numbers on every response. `build_response` produces that payload and
raises `ContractViolation` when an element is missing, so an uncited answer fails
loudly instead of reaching a user who cannot tell it apart from a grounded one.
`strict=False` downgrades violations to warnings, which is what the benchmarks use
so they can count failures across a run.

A refusal is exempt from the evidence requirement: it asserts nothing, so it needs
nothing to support it.

`POST /ask` returns this under the `contract` key, rendered as in the spec's worked
example: `Clause 8.2 Page 12`.

## Interface with Kiran's half

Everything here consumes evidence through the `rag.retrieval.ClauseRetriever`
protocol, so it runs unchanged against the hybrid knowledge base or the
`InMemoryClauseRetriever` baseline.

## Running

```bash
python -m pytest venkata_answering/tests -q
python -m venkata_answering.routing_benchmark
```
