import React, { useState, useRef, useEffect } from "react";
import "./Dashboard.css";
import { FaPaperPlane, FaBars, FaPaperclip } from "react-icons/fa";

export default function Dashboard() {
  const [text, setText] = useState("");
  const [pipeline, setPipeline] = useState("v3");
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [sidebar, setSidebar] = useState(false);
  const [loading, setLoading] = useState(false);

  const [history, setHistory] = useState([]);
  const [messages, setMessages] = useState([]);

  const fileRef = useRef(null);
  const chatEndRef = useRef(null);

  const fetchHistory = async () => {
    try {
      const res = await fetch("http://localhost:5001/results");
      const data = await res.json();
      setHistory(data);
    } catch {
      console.error("Failed to fetch history");
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const loadHistoryItem = async (docId) => {
    try {
      const res = await fetch(`http://localhost:5001/results/${docId}`);
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

      setSidebar(false);
    } catch {
      console.error("Failed to load history item");
    }
  };

  const handlePDFUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("pdf", file);
    formData.append("pipeline", pipeline);

    setMessages((prev) => [
      ...prev,
      { type: "user", text: file.name }
    ]);

    setLoading(true);

    try {
      const res = await fetch("http://localhost:5001/analyze-pdf", {
        method: "POST",
        body: formData
      });

      const data = await res.json();

      if (data.error) {
        setMessages((prev) => [
          ...prev,
          { type: "bot", text: "Error: " + data.error }
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            type: "bot",
            summary: data.summary,
            clauses: data.clauses,
            clauseCount: data.clause_count
          }
        ]);
      }

      fetchHistory();

    } catch {
      setMessages((prev) => [
        ...prev,
        { type: "bot", text: "PDF analysis failed" }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!text.trim()) return;

    setMessages((prev) => [
      ...prev,
      { type: "user", text }
    ]);

    setText("");
    setLoading(true);

    try {
      const res = await fetch("http://localhost:5001/analyze-text", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ text, pipeline })
      });

      const data = await res.json();

      if (data.error) {
        setMessages((prev) => [
          ...prev,
          { type: "bot", text: "Error: " + data.error }
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          {
            type: "bot",
            summary: data.summary,
            clauses: data.clauses,
            clauseCount: data.clause_count
          }
        ]);
      }

      fetchHistory();

    } catch {
      setMessages((prev) => [
        ...prev,
        { type: "bot", text: "Text analysis failed" }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard">

      <div className={`sidebar ${sidebar ? "show" : ""}`}>
        <div className="sidebar-header">
          <span>History</span>
          <span className="close" onClick={() => setSidebar(false)}>×</span>
        </div>

        <div
          className="history-item new-chat"
          onClick={() => {
            setMessages([]);
            setSidebar(false);
          }}
        >
          + New Chat
        </div>

        {history.map((item) => (
          <div
            key={item.docId}
            className="history-item"
            onClick={() => loadHistoryItem(item.docId)}
          >
            {item.filename}
          </div>
        ))}
      </div>

      <div className="nav">
        <FaBars className="menu" onClick={() => setSidebar(true)} />
        <span className="nav-logo">LegalEase</span>
      </div>

      <div className="chat-area">
        {messages.map((msg, i) => (
          <div key={i} className={`chat-row ${msg.type}`}>
            <div className={`bubble ${msg.type}`}>

              {msg.type === "user" && msg.text}

              {msg.type === "bot" && (
                <>
                  {msg.text && <div>{msg.text}</div>}

                  {msg.summary && (
                    <div className="summary-block">
                      <b>Summary</b>
                      <p>{msg.summary}</p>
                    </div>
                  )}

                  {msg.clauseCount !== undefined && (
                    <div className="clause-count">
                      {msg.clauseCount} clauses extracted
                    </div>
                  )}

                  {msg.clauses && (
                    <div className="clause-grid">
                      {Object.entries(msg.clauses).map(([key, val]) => (
                        <div key={key} className="clause-card">
                          <div className="clause-title">{key}</div>
                          <div className="clause-text">{val.span}</div>
                          {val.score && (
                            <div className="clause-score">
                              Score: {val.score}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </>
              )}

            </div>
          </div>
        ))}

        {loading && <div className="bubble bot">Processing...</div>}

        <div ref={chatEndRef} />
      </div>

      <div className="input-wrap">
        <div className="input-box">

          <input
            type="file"
            hidden
            ref={fileRef}
            onChange={handlePDFUpload}
          />

          <div className="icon-btn" onClick={() => fileRef.current.click()}>
            <FaPaperclip />
          </div>

          <div className="dropdown">
            <div
              className="pipeline"
              onClick={() => setDropdownOpen(!dropdownOpen)}
            >
              {pipeline.toUpperCase()} ▾
            </div>

            {dropdownOpen && (
              <div className="dropdown-menu">
                {["v1","v2","v3"].map(p => (
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

          <input
            className="text-input"
            value={text}
            placeholder="Paste contract or upload PDF..."
            onChange={(e) => setText(e.target.value)}
          />

          <div className="send" onClick={handleAnalyze}>
            <FaPaperPlane />
          </div>

        </div>
      </div>

    </div>
  );
}
