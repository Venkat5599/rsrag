import React, { useState, useRef, useEffect } from "react";
import "./Dashboard.css";
import { FaPaperPlane, FaBars, FaPaperclip } from "react-icons/fa";

const API = "http://localhost:5001";

const STRATEGY_LABELS = {
  deberta_extractive_qa: "DeBERTa extractive QA",
  retriever_only: "Retriever only",
  legalbert_entity_extraction: "LegalBERT entity extraction",
  grounded_llm: "Grounded LLM"
};

const INTENT_LABELS = {
  fact_lookup: "Fact lookup",
  clause_retrieval: "Clause retrieval",
  entity_lookup: "Entity lookup",
  clause_comparison: "Clause comparison",
  summarization: "Summarization",
  risk_explanation: "Risk explanation",
  legal_reasoning: "Legal reasoning"
};

const SIGNAL_LABELS = {
  semantic: "Semantic",
  bm25: "BM25",
  cross_encoder: "Cross encoder",
  clause_importance: "Clause importance",
  heading_match: "Heading match",
  entity_match: "Entity match"
};

const percent = (value) => `${Math.round((value || 0) * 100)}%`;

/* Confidence band, spec section 12. The band drives the colour, not the raw
   number, so a reader does not have to interpret a decimal to see reliability. */
function ConfidenceBadge({ score, band }) {
  return (
    <span className={`badge band-${band || "low"}`}>
      Confidence {percent(score)} ({band})
    </span>
  );
}

/* Which model answered, and why. This is the visible proof of spec section 11:
   the system did not invoke an LLM unless the intent needed one. */
function RoutingRow({ plan }) {
  if (!plan) return null;

  return (
    <div className="routing-row">
      <span className="badge neutral">{INTENT_LABELS[plan.intent] || plan.intent}</span>
      <span className="badge neutral">
        {STRATEGY_LABELS[plan.strategy] || plan.strategy}
      </span>
      <span className="routing-reason">{plan.routing_reason}</span>
    </div>
  );
}

/* Section 12 requires clause numbers and page numbers on every response. */
function Citations({ evidence }) {
  if (!evidence || evidence.length === 0) {
    return <div className="no-evidence">No supporting evidence was retrieved.</div>;
  }

  return (
    <div className="citation-list">
      {evidence.map((item) => (
        <div key={item.chunk_id} className="citation">
          <div className="citation-head">
            <span className="citation-where">{item.display}</span>
            <span className="citation-type">{item.clause_type}</span>
          </div>
          <div className="citation-quote">{item.quote}</div>
        </div>
      ))}
    </div>
  );
}

/* Section 13. Faithfulness asks whether the answer is supported; grounding asks
   whether it is attributed. They fail independently, so both are shown. */
function VerificationRow({ contract }) {
  if (!contract) return null;

  const grounding = contract.grounding;

  return (
    <div className="verification-row">
      <span className={`badge ${contract.faithful ? "ok" : "warn"}`}>
        {contract.faithful ? "Faithful" : "Not fully supported"} {percent(contract.faithfulness)}
      </span>

      {grounding && (
        <span className={`badge ${grounding.valid ? "ok" : "warn"}`}>
          {grounding.valid ? "Grounded" : "Grounding issues"}
        </span>
      )}

      {grounding && grounding.invalid_markers.length > 0 && (
        <span className="badge warn">
          Fabricated citations: {grounding.invalid_markers.join(", ")}
        </span>
      )}
    </div>
  );
}

/* The retrieval inspector. Every clause carries the six signals that produced its
   rank (spec section 9), so a ranking can be explained rather than trusted. */
function RetrievalPanel({ chunks }) {
  if (!chunks || chunks.length === 0) {
    return <div className="no-evidence">Retrieval returned no clauses.</div>;
  }

  return (
    <div className="retrieval-panel">
      {chunks.map((chunk, index) => {
        const signals = chunk.metadata && chunk.metadata.signals;

        return (
          <div key={chunk.chunk_id} className="retrieval-chunk">
            <div className="retrieval-head">
              <span className="retrieval-rank">{index + 1}</span>
              <span className="retrieval-title">
                {chunk.clause_type || "Unclassified clause"}
              </span>
              <span className="retrieval-where">
                {chunk.section}
                {chunk.page ? ` · Page ${chunk.page}` : ""}
              </span>
              <span className="retrieval-score">{chunk.retrieval_score.toFixed(3)}</span>
            </div>

            {signals && (
              <div className="signal-grid">
                {Object.keys(SIGNAL_LABELS).map((key) => (
                  <div key={key} className="signal">
                    <div className="signal-label">
                      <span>{SIGNAL_LABELS[key]}</span>
                      <span>{(signals[key] || 0).toFixed(2)}</span>
                    </div>
                    <div className="signal-track">
                      <div
                        className="signal-fill"
                        style={{ width: percent(signals[key]) }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

function AnswerMessage({ msg, onInspect }) {
  const contract = msg.contract;

  return (
    <>
      <div className="answer-text">{msg.answer}</div>

      <div className="answer-meta">
        {contract && (
          <ConfidenceBadge score={contract.confidence} band={contract.confidence_band} />
        )}
        {msg.generator && <span className="badge neutral">{msg.generator}</span>}
      </div>

      <RoutingRow plan={msg.plan} />
      <VerificationRow contract={contract} />

      <div className="section-label">Evidence</div>
      <Citations evidence={contract && contract.evidence} />

      {msg.warnings && msg.warnings.length > 0 && (
        <div className="warning-list">
          {msg.warnings.map((warning, index) => (
            <div key={index} className="warning">
              {warning}
            </div>
          ))}
        </div>
      )}

      <button className="inspect-btn" onClick={onInspect}>
        {msg.retrievalOpen ? "Hide retrieval detail" : "Show how these clauses were ranked"}
      </button>

      {msg.retrievalOpen && (
        <>
          <div className="section-label">
            Hybrid retrieval: metadata filter, dense, BM25, cross encoder
          </div>
          {msg.retrievalLoading ? (
            <div className="no-evidence">Loading retrieval detail...</div>
          ) : (
            <RetrievalPanel chunks={msg.retrieval} />
          )}
        </>
      )}
    </>
  );
}

export default function Dashboard() {
  const [text, setText] = useState("");
  const [pipeline, setPipeline] = useState("v3");
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [sidebar, setSidebar] = useState(false);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState("analyze");

  const [history, setHistory] = useState([]);
  const [messages, setMessages] = useState([]);
  const [activeDoc, setActiveDoc] = useState(null);
  /* Extra contracts held alongside activeDoc. Spec section 11 routes clause
     comparison to the grounded LLM, and a comparison needs more than one agreement
     in scope - with only one the backend answers from a single contract and says so
     in a warning. Kept separate from activeDoc so the analysed document stays primary. */
  const [compareDocs, setCompareDocs] = useState([]);
  const [health, setHealth] = useState(null);

  const fileRef = useRef(null);
  const chatEndRef = useRef(null);

  const loadHistory = async () => {
    const res = await fetch(`${API}/results`);
    return res.json();
  };

  const loadHealth = async () => {
    const res = await fetch(`${API}/health`);
    return res.json();
  };

  const fetchHistory = async () => {
    try {
      setHistory(await loadHistory());
    } catch {
      console.error("Failed to fetch history");
    }
  };

  /* The cancelled flag stops a slow response from writing state after unmount,
     which is also why the fetchers return data instead of setting it themselves. */
  useEffect(() => {
    let cancelled = false;

    const bootstrap = async () => {
      try {
        const documents = await loadHistory();
        if (!cancelled) setHistory(documents);
      } catch {
        console.error("Failed to fetch history");
      }

      try {
        const status = await loadHealth();
        if (!cancelled) setHealth(status);
      } catch {
        console.error("Failed to fetch health");
      }
    };

    bootstrap();

    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const pushMessage = (message) => setMessages((prev) => [...prev, message]);

  const updateMessage = (index, patch) =>
    setMessages((prev) =>
      prev.map((message, position) =>
        position === index ? { ...message, ...patch } : message
      )
    );

  const loadHistoryItem = async (docId) => {
    try {
      const res = await fetch(`${API}/results/${docId}`);
      const data = await res.json();

      setMessages([
        { type: "user", text: data.filename || "Previous Document" },
        {
          type: "bot",
          summary: data.summary,
          clauses: data.clauses,
          clauseCount: data.clause_count
        }
      ]);

      setActiveDoc({ docId: data.docId, filename: data.filename });
      // A contract cannot be compared against itself.
      setCompareDocs((previous) => previous.filter((held) => held !== data.docId));
      setMode("ask");
      setSidebar(false);
    } catch {
      console.error("Failed to load history item");
    }
  };

  const handleAnalysisResult = (data) => {
    if (data.error) {
      pushMessage({ type: "bot", text: `Error: ${data.error}` });
      return;
    }

    pushMessage({
      type: "bot",
      summary: data.summary,
      clauses: data.clauses,
      clauseCount: data.clause_count
    });

    // The contract is now indexed, so questions can be scoped to it.
    setActiveDoc({ docId: data.docId, filename: data.filename });
    setMode("ask");
  };

  const handlePDFUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("pdf", file);
    formData.append("pipeline", pipeline);

    pushMessage({ type: "user", text: file.name });
    setLoading(true);

    try {
      const res = await fetch(`${API}/analyze-pdf`, { method: "POST", body: formData });
      handleAnalysisResult(await res.json());
      fetchHistory();
    } catch {
      pushMessage({ type: "bot", text: "PDF analysis failed" });
    } finally {
      setLoading(false);
      e.target.value = "";
    }
  };

  const handleAnalyze = async () => {
    if (!text.trim()) return;

    const submitted = text;
    pushMessage({ type: "user", text: submitted });
    setText("");
    setLoading(true);

    try {
      const res = await fetch(`${API}/analyze-text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: submitted, pipeline })
      });

      handleAnalysisResult(await res.json());
      fetchHistory();
    } catch {
      pushMessage({ type: "bot", text: "Text analysis failed" });
    } finally {
      setLoading(false);
    }
  };

  /* The scope sent to the backend: the analysed contract first, then anything
     picked for comparison. docId stays on the request so single-contract behaviour
     is byte-for-byte what it was before. */
  const scopedDocIds = () =>
    [activeDoc?.docId, ...compareDocs].filter(
      (docId, index, all) => docId && all.indexOf(docId) === index
    );

  const toggleCompare = (docId) => {
    if (activeDoc && activeDoc.docId === docId) return;

    setCompareDocs((previous) =>
      previous.includes(docId)
        ? previous.filter((held) => held !== docId)
        : [...previous, docId]
    );
  };

  const handleAsk = async () => {
    if (!text.trim() || !activeDoc) return;

    const question = text;
    pushMessage({ type: "user", text: question });
    setText("");
    setLoading(true);

    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question,
          docId: activeDoc.docId,
          docIds: scopedDocIds()
        })
      });

      const data = await res.json();

      if (data.error) {
        pushMessage({ type: "bot", text: `Error: ${data.error}` });
        return;
      }

      pushMessage({
        type: "bot",
        kind: "answer",
        question,
        answer: data.answer,
        plan: data.plan,
        contract: data.contract,
        generator: data.generator,
        warnings: data.warnings,
        retrievalOpen: false,
        retrieval: null,
        retrievalLoading: false
      });
    } catch {
      pushMessage({ type: "bot", text: "Question answering failed" });
    } finally {
      setLoading(false);
    }
  };

  /* Retrieval detail is fetched lazily, once, on first expand. It is a separate
     call to /retrieve rather than data smuggled into /ask, so what is shown is
     genuinely what the retriever returns. */
  const toggleRetrieval = async (index) => {
    const message = messages[index];

    if (message.retrievalOpen) {
      updateMessage(index, { retrievalOpen: false });
      return;
    }

    if (message.retrieval) {
      updateMessage(index, { retrievalOpen: true });
      return;
    }

    updateMessage(index, { retrievalOpen: true, retrievalLoading: true });

    try {
      const res = await fetch(`${API}/retrieve`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: message.question,
          docId: activeDoc?.docId,
          docIds: scopedDocIds()
        })
      });

      const data = await res.json();
      updateMessage(index, { retrieval: data.chunks || [], retrievalLoading: false });
    } catch {
      updateMessage(index, { retrieval: [], retrievalLoading: false });
    }
  };

  const submit = () => (mode === "ask" ? handleAsk() : handleAnalyze());

  const askDisabled = mode === "ask" && !activeDoc;

  return (
    <div className="dashboard">
      <div className={`sidebar ${sidebar ? "show" : ""}`}>
        <div className="sidebar-header">
          <span>History</span>
          <span className="close" onClick={() => setSidebar(false)}>
            ×
          </span>
        </div>

        <div
          className="history-item new-chat"
          onClick={() => {
            setMessages([]);
            setActiveDoc(null);
            setCompareDocs([]);
            setMode("analyze");
            setSidebar(false);
          }}
        >
          + New Chat
        </div>

        {history.map((item) => {
          const isActive = activeDoc && activeDoc.docId === item.docId;
          const isCompared = compareDocs.includes(item.docId);

          return (
            <div
              key={item.docId}
              className={`history-item ${isActive ? "active" : ""} ${
                isCompared ? "compared" : ""
              }`}
              onClick={() => loadHistoryItem(item.docId)}
            >
              <span className="history-name">{item.filename}</span>

              {!isActive && activeDoc && (
                <button
                  type="button"
                  className={`compare-toggle ${isCompared ? "on" : ""}`}
                  title={
                    isCompared
                      ? "Remove from comparison scope"
                      : "Add to comparison scope"
                  }
                  onClick={(event) => {
                    event.stopPropagation();
                    toggleCompare(item.docId);
                  }}
                >
                  {isCompared ? "In comparison" : "Compare"}
                </button>
              )}
            </div>
          );
        })}

        {health && health.retrieval && (
          <div className="health-panel">
            <div className="health-title">Retrieval stack</div>
            <div className="health-row">
              <span>Retriever</span>
              <span>{health.retriever}</span>
            </div>
            <div className="health-row">
              <span>Vector index</span>
              <span>{health.retrieval.vector_backend}</span>
            </div>
            <div className="health-row">
              <span>Embeddings</span>
              <span>
                {health.retrieval.embedding_backend === "hashed_bag_of_words"
                  ? "lexical fallback"
                  : "bge-large-en-v1.5"}
              </span>
            </div>
            <div className="health-row">
              <span>Reranker</span>
              <span>
                {health.retrieval.reranker_backend === "lexical_rerank"
                  ? "lexical fallback"
                  : "bge-reranker-large"}
              </span>
            </div>
            <div className="health-row">
              <span>Indexed clauses</span>
              <span>{health.retrieval.indexed_chunks}</span>
            </div>
            <div className="health-row">
              <span>LLM</span>
              <span>{health.llm_configured ? health.llm_provider : "not configured"}</span>
            </div>
          </div>
        )}
      </div>

      <div className="nav">
        <FaBars className="menu" onClick={() => setSidebar(true)} />
        <span className="nav-logo">LegalEase</span>
      </div>

      <div className="chat-area">
        {messages.length === 0 && (
          <div className="empty-state">
            <div className="empty-title">Analyse a contract, then ask about it</div>
            <div className="empty-body">
              Upload a PDF or paste contract text. Once it is indexed, switch to Ask and
              every answer comes back with its clause and page citations, a confidence
              band, and the retrieval detail behind it.
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`chat-row ${msg.type}`}>
            <div className={`bubble ${msg.type} ${msg.kind === "answer" ? "answer" : ""}`}>
              {msg.type === "user" && msg.text}

              {msg.type === "bot" && msg.kind === "answer" && (
                <AnswerMessage msg={msg} onInspect={() => toggleRetrieval(i)} />
              )}

              {msg.type === "bot" && msg.kind !== "answer" && (
                <>
                  {msg.text && <div>{msg.text}</div>}

                  {msg.summary && (
                    <div className="summary-block">
                      <b>Summary</b>
                      <p>{msg.summary}</p>
                    </div>
                  )}

                  {msg.clauseCount !== undefined && (
                    <div className="clause-count">{msg.clauseCount} clauses extracted</div>
                  )}

                  {msg.clauses && (
                    <div className="clause-grid">
                      {Object.entries(msg.clauses).map(([key, val]) => (
                        <div key={key} className="clause-card">
                          <div className="clause-title">{key}</div>
                          <div className="clause-text">{val.span}</div>
                          <div className="clause-meta">
                            {val.section && <span>{val.section}</span>}
                            {val.page ? <span>Page {val.page}</span> : null}
                            {val.score ? <span>Score {val.score}</span> : null}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        ))}

        {loading && (
          <div className="chat-row bot">
            <div className="bubble bot typing">
              {mode === "ask" ? "Retrieving evidence..." : "Processing..."}
            </div>
          </div>
        )}

        <div ref={chatEndRef} />
      </div>

      <div className="input-wrap">
        <div className="input-column">
          {activeDoc && (
            <div className="context-bar">
              <span className="context-label">Asking about</span>
              <span className="context-file">{activeDoc.filename}</span>

              {compareDocs.map((docId) => {
                const item = history.find((entry) => entry.docId === docId);

                return (
                  <span key={docId} className="context-file compared">
                    + {item ? item.filename : docId.slice(0, 8)}
                    <span
                      className="context-drop"
                      title="Remove from comparison scope"
                      onClick={() => toggleCompare(docId)}
                    >
                      ×
                    </span>
                  </span>
                );
              })}

              <span
                className="context-clear"
                onClick={() => {
                  setActiveDoc(null);
                  setCompareDocs([]);
                  setMode("analyze");
                }}
              >
                clear
              </span>
            </div>
          )}

          <div className="input-box">
            <input type="file" hidden ref={fileRef} onChange={handlePDFUpload} />

            <div className="icon-btn" onClick={() => fileRef.current.click()}>
              <FaPaperclip />
            </div>

            {mode === "analyze" ? (
              <div className="dropdown">
                <div className="pipeline" onClick={() => setDropdownOpen(!dropdownOpen)}>
                  {pipeline.toUpperCase()} ▾
                </div>

                {dropdownOpen && (
                  <div className="dropdown-menu">
                    {["v1", "v2", "v3"].map((p) => (
                      <div
                        key={p}
                        className="item"
                        onClick={() => {
                          setPipeline(p);
                          setDropdownOpen(false);
                        }}
                      >
                        {p.toUpperCase()}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ) : (
              <div className="pipeline static">ASK</div>
            )}

            <input
              className="text-input"
              value={text}
              placeholder={
                mode === "ask"
                  ? "Ask about this contract..."
                  : "Paste contract or upload PDF..."
              }
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") submit();
              }}
            />

            <div
              className={`send ${askDisabled ? "disabled" : ""}`}
              onClick={() => !askDisabled && submit()}
            >
              <FaPaperPlane />
            </div>
          </div>

          <div className="mode-row">
            <span
              className={`mode ${mode === "analyze" ? "on" : ""}`}
              onClick={() => setMode("analyze")}
            >
              Analyse
            </span>
            <span
              className={`mode ${mode === "ask" ? "on" : ""} ${!activeDoc ? "locked" : ""}`}
              onClick={() => activeDoc && setMode("ask")}
            >
              Ask
            </span>
            {!activeDoc && <span className="mode-hint">Analyse a contract to enable Ask</span>}
          </div>
        </div>
      </div>
    </div>
  );
}
