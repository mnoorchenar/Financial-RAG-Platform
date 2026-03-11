"""
app.py — Flask entry point + all routes + inline HTML/CSS/JS UI.
No templates/ or static/ folders needed.

Run locally:  python app.py
"""

import os
import uuid
import logging
from pathlib import Path

from flask import Flask, render_template_string, request, jsonify, session

from pipeline import pipeline_query, add_document_to_index, clear_session_memory
from ingest import FAISS_INDEX_PATH, BASE_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s  [%(levelname)-8s]  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)

UPLOADED_DOCS_PATH = str(BASE_DIR / "data" / "uploads")
FLASK_SECRET_KEY   = os.environ.get("FLASK_SECRET_KEY", os.urandom(24).hex())

app = Flask(__name__)
app.secret_key = FLASK_SECRET_KEY

# ══════════════════════════════════════════════════════════════════════
# INLINE HTML / CSS / JS
# ══════════════════════════════════════════════════════════════════════

_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Financial Document Intelligence Platform</title>
  <style>
    :root{--bg0:#070c18;--bg1:#0f1929;--bg2:#1a2236;--bg3:#1e2d47;--accent:#3b82f6;--accent2:#4f46e5;--accent3:#06b6d4;--green:#22c55e;--amber:#f59e0b;--red:#ef4444;--txt1:#e8eaf6;--txt2:#94a3b8;--txt3:#475569;--border:#1e3a5f;--r:12px;}
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0;}
    html,body{height:100%;}
    body{background:var(--bg0);color:var(--txt1);font-family:'Inter',system-ui,sans-serif;font-size:15px;line-height:1.6;}
    code{background:var(--bg3);border-radius:4px;padding:1px 5px;font-size:.85em;color:var(--accent3);}
    pre{overflow-x:auto;}
    ul{padding-left:1.4em;}li{margin-bottom:4px;}
    ::-webkit-scrollbar{width:5px;height:5px;}::-webkit-scrollbar-track{background:var(--bg0);}::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px;}::-webkit-scrollbar-thumb:hover{background:var(--accent);}

    /* Header */
    .fin-header{background:linear-gradient(135deg,var(--bg1) 0%,var(--bg2) 60%,var(--bg1) 100%);border-bottom:1px solid var(--border);padding:28px 40px 22px;text-align:center;}
    .fin-title{font-size:clamp(1.6rem,3vw,2.4rem);font-weight:900;background:linear-gradient(90deg,var(--accent),var(--accent2),var(--accent3));-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;letter-spacing:-.5px;}
    .fin-sub{color:var(--txt2);font-size:.88rem;margin-top:8px;}
    .fin-pills{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;margin-top:14px;}
    .pill{background:var(--bg3);border:1px solid var(--border);border-radius:20px;padding:3px 12px;font-size:.75rem;color:var(--txt2);}

    /* Tabs */
    .tab-nav{display:flex;gap:4px;padding:12px 40px 0;background:var(--bg1);border-bottom:1px solid var(--border);}
    .tab-btn{background:transparent;border:1px solid transparent;border-bottom:none;border-radius:var(--r) var(--r) 0 0;color:var(--txt2);cursor:pointer;font-size:.88rem;font-weight:600;padding:9px 20px;transition:color .15s,background .15s;}
    .tab-btn:hover{color:var(--txt1);background:var(--bg2);}
    .tab-btn.active{background:var(--bg2);border-color:var(--border);color:var(--accent);}
    .tab-content{display:none;padding:24px 40px;}
    .tab-content.active{display:block;}

    /* Chat */
    .chat-layout{display:grid;grid-template-columns:1fr 300px;gap:20px;align-items:start;}
    @media(max-width:900px){.chat-layout{grid-template-columns:1fr;}}
    .chat-messages{background:var(--bg1);border:1px solid var(--border);border-radius:var(--r);padding:16px;height:440px;overflow-y:auto;display:flex;flex-direction:column;gap:12px;}
    .msg{display:flex;}.user-msg{justify-content:flex-end;}
    .bubble{max-width:80%;padding:10px 14px;border-radius:12px;font-size:.9rem;line-height:1.6;white-space:pre-wrap;word-break:break-word;}
    .user-msg .bubble{background:linear-gradient(135deg,#1e3a5f,#162d4a);border:1px solid #2563eb;border-radius:12px 12px 4px 12px;}
    .bot-msg .bubble{background:var(--bg2);border:1px solid var(--border);border-radius:12px 12px 12px 4px;}
    .typing-indicator{display:flex;gap:5px;align-items:center;padding:12px 14px;}
    .dot{width:7px;height:7px;background:var(--txt3);border-radius:50%;animation:bounce 1.2s infinite;}
    .dot:nth-child(2){animation-delay:.2s;}.dot:nth-child(3){animation-delay:.4s;}
    @keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-6px)}}
    .examples-bar{margin-top:12px;}
    .examples-label{color:var(--txt3);font-size:.8rem;display:block;margin-bottom:6px;}
    .examples-list{display:flex;flex-wrap:wrap;gap:6px;}
    .example-chip{background:var(--bg2);border:1px solid var(--border);border-radius:20px;color:var(--txt2);cursor:pointer;font-size:.78rem;padding:4px 12px;transition:border-color .15s,color .15s;}
    .example-chip:hover{border-color:var(--accent);color:var(--accent);}
    .input-row{display:flex;gap:10px;align-items:flex-end;margin-top:12px;}
    .input-row textarea{flex:1;resize:none;background:var(--bg1);border:1px solid var(--border);border-radius:10px;color:var(--txt1);font-size:.9rem;padding:10px 14px;transition:border-color .15s;font-family:inherit;}
    .input-row textarea:focus{border-color:var(--accent);outline:none;box-shadow:0 0 0 2px rgba(59,130,246,.2);}
    .input-actions{display:flex;flex-direction:column;gap:6px;}

    /* Buttons */
    .btn-primary{background:linear-gradient(135deg,var(--accent),var(--accent2));border:none;border-radius:10px;color:#fff;cursor:pointer;font-size:.9rem;font-weight:700;padding:10px 18px;transition:opacity .15s,transform .15s;white-space:nowrap;}
    .btn-primary:hover:not(:disabled){opacity:.88;transform:translateY(-1px);}
    .btn-primary:disabled{opacity:.5;cursor:not-allowed;}
    .btn-secondary{background:var(--bg3);border:1px solid var(--border);border-radius:10px;color:var(--txt2);cursor:pointer;font-size:.85rem;padding:8px 14px;transition:background .15s,color .15s;}
    .btn-secondary:hover{background:var(--bg2);color:var(--txt1);}

    /* Sidebar */
    .chat-sidebar{display:flex;flex-direction:column;gap:16px;}
    .sidebar-block{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);padding:14px 16px;}
    .sidebar-label{color:var(--txt2);font-size:.78rem;font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-bottom:10px;}
    .sidebar-content{color:var(--txt1);font-size:.87rem;line-height:1.6;}
    .sidebar-content.muted{color:var(--txt3);font-style:italic;}
    .confidence-badge{font-size:1rem;font-weight:700;margin-bottom:6px;}
    .confidence-note{color:var(--txt2);font-size:.82rem;}
    .source-item{display:flex;align-items:baseline;gap:6px;margin-bottom:6px;font-size:.83rem;}
    .source-item code{font-size:.8rem;}.source-page{color:var(--txt3);font-size:.78rem;}

    /* Upload */
    .upload-panel{max-width:800px;}
    .upload-info-box{background:var(--bg2);border:1px solid var(--border);border-radius:var(--r);padding:22px 26px;margin-bottom:24px;}
    .upload-info-box h3{color:var(--accent);margin-bottom:10px;font-size:1.05rem;}
    .upload-info-box p{color:var(--txt2);font-size:.9rem;margin-bottom:12px;}
    .upload-info-box ul{color:var(--txt3);font-size:.85rem;}
    .upload-form-row{display:flex;flex-direction:column;gap:14px;}
    .upload-zone{background:var(--bg1);border:2px dashed var(--border);border-radius:var(--r);cursor:pointer;padding:40px;text-align:center;transition:border-color .2s;}
    .upload-zone:hover,.upload-zone.drag-over{border-color:var(--accent);}
    .upload-icon{font-size:2.5rem;display:block;margin-bottom:10px;}
    .upload-zone p{color:var(--txt2);font-size:.9rem;}.upload-hint{color:var(--txt3);font-size:.8rem;margin-top:6px;}
    .upload-status{font-size:.9rem;padding:10px 14px;background:var(--bg2);border:1px solid var(--border);border-radius:8px;}
    .upload-status.muted{color:var(--txt3);font-style:italic;}
    .hidden{display:none!important;}

    /* Architecture */
    .arch-panel{max-width:960px;}
    .arch-panel h2{color:var(--accent);font-size:1.3rem;margin-bottom:20px;}
    .arch-diagram{background:var(--bg1);border:1px solid var(--border);border-radius:var(--r);padding:20px 24px;overflow-x:auto;margin-bottom:24px;}
    .arch-diagram pre{color:var(--txt2);font-family:'Fira Code','Courier New',monospace;font-size:.82rem;line-height:1.5;}
    .arch-table{width:100%;border-collapse:collapse;font-size:.88rem;}
    .arch-table th{background:var(--bg2);border:1px solid var(--border);color:var(--accent);font-weight:700;padding:10px 14px;text-align:left;}
    .arch-table td{background:var(--bg1);border:1px solid var(--border);color:var(--txt2);padding:9px 14px;vertical-align:top;}
    .arch-table tr:hover td{background:var(--bg2);}
  </style>
</head>
<body>

  <header class="fin-header">
    <h1 class="fin-title">🏦 Financial Document Intelligence Platform</h1>
    <p class="fin-sub">Retrieval-Augmented Generation &nbsp;·&nbsp; Banking &amp; Insurance &nbsp;·&nbsp; LangChain &nbsp;·&nbsp; FAISS &nbsp;·&nbsp; LLaMA 3.3 70B</p>
    <div class="fin-pills">
      <span class="pill">📄 FDIC</span><span class="pill">🛡️ NAIC</span><span class="pill">📊 SEC</span>
      <span class="pill">🏛️ Basel III</span><span class="pill">🔍 MMR Retrieval</span><span class="pill">💬 Multi-turn Memory</span>
    </div>
  </header>

  <nav class="tab-nav">
    <button class="tab-btn active" data-tab="chat">💬 Ask FinRAG</button>
    <button class="tab-btn" data-tab="upload">📤 Upload Documents</button>
    <button class="tab-btn" data-tab="arch">🏗️ Architecture</button>
  </nav>

  <!-- Chat Tab -->
  <section id="tab-chat" class="tab-content active">
    <div class="chat-layout">
      <div class="chat-main">
        <div id="chat-messages" class="chat-messages">
          <div class="msg bot-msg"><div class="bubble">👋 Hello! I'm FinRAG. Ask me anything about FDIC deposit insurance, Basel III, life insurance, SEC filings, AML compliance, or any banking and insurance topic.</div></div>
        </div>
        <div class="examples-bar">
          <span class="examples-label">💡 Try asking:</span>
          <div class="examples-list" id="examples-list"></div>
        </div>
        <div class="input-row">
          <textarea id="question-input" rows="2" placeholder="Ask about deposit insurance, Basel III, life insurance, 10-K filings..."></textarea>
          <div class="input-actions">
            <button id="ask-btn" class="btn-primary">Ask →</button>
            <button id="clear-btn" class="btn-secondary">🗑️ Clear</button>
          </div>
        </div>
      </div>
      <aside class="chat-sidebar">
        <div class="sidebar-block">
          <div class="sidebar-label">📊 Confidence</div>
          <div id="confidence-box" class="sidebar-content muted">Ask a question to see retrieval confidence.</div>
        </div>
        <div class="sidebar-block">
          <div class="sidebar-label">📚 Sources</div>
          <div id="sources-box" class="sidebar-content muted">Source citations appear here after each answer.</div>
        </div>
      </aside>
    </div>
  </section>

  <!-- Upload Tab -->
  <section id="tab-upload" class="tab-content">
    <div class="upload-panel">
      <div class="upload-info-box">
        <h3>Extend the Knowledge Base at Runtime</h3>
        <p>Upload your own financial PDFs and FinRAG will embed and index them immediately — no restart required.</p>
        <ul>
          <li>Format: PDF only &nbsp;·&nbsp; Recommended max size: 50 MB</li>
          <li>Documents are chunked, embedded, and merged into the live FAISS index via <code>vectorstore.merge_from()</code></li>
          <li>Index is re-persisted to disk after each upload</li>
        </ul>
      </div>
      <div class="upload-form-row">
        <div class="upload-zone" id="upload-zone">
          <input type="file" id="pdf-input" accept=".pdf" hidden/>
          <div id="upload-placeholder">
            <span class="upload-icon">📂</span>
            <p>Click to select a PDF or drag and drop here</p>
            <p class="upload-hint">PDF only · Max 50 MB</p>
          </div>
          <div id="upload-selected" class="hidden"></div>
        </div>
        <button id="upload-btn" class="btn-primary">📥 Add to Knowledge Base</button>
        <div id="upload-status" class="upload-status muted">No document uploaded yet.</div>
      </div>
    </div>
  </section>

  <!-- Architecture Tab -->
  <section id="tab-arch" class="tab-content">
    <div class="arch-panel">
      <h2>System Architecture</h2>
      <div class="arch-diagram"><pre>
┌─────────────────────────────────────────────────────────────────┐
│                   Financial-RAG-Platform                        │
│                                                                 │
│  ┌───────────┐   ┌──────────────────┐   ┌──────────────────┐  │
│  │ PDF / TXT │──▶│   ingest.py      │──▶│  FAISS Index     │  │
│  │  Sources  │   │ Chunk + Embed    │   │  (pre-baked in   │  │
│  │ FDIC/NAIC │   │ MiniLM-L6-v2     │   │  Docker image)   │  │
│  └───────────┘   └──────────────────┘   └────────┬─────────┘  │
│                                                   │             │
│  ┌───────────┐   ┌──────────────────┐   ┌────────▼─────────┐  │
│  │Guardrails │──▶│   pipeline.py    │◀──│  MMR Retriever   │  │
│  │  (domain  │   │  LangChain Conv. │   │  top-k = 5       │  │
│  │  filter)  │   │  RetrievalChain  │   └──────────────────┘  │
│  └───────────┘   └────────┬─────────┘                         │
│                           │            ┌──────────────────┐   │
│  ┌───────────┐            │            │  Confidence      │   │
│  │  Session  │◀───────────┤            │  Scoring         │   │
│  │  Memory   │            ▼            └──────────────────┘   │
│  └───────────┘   ┌──────────────────┐                         │
│                  │   Groq API       │                         │
│                  │  LLaMA 3.3 70B   │                         │
│                  └────────┬─────────┘                         │
│                           │                                   │
│                  ┌────────▼─────────┐                         │
│                  │     app.py       │                         │
│                  │  Flask + HTML    │                         │
│                  │  /api/chat       │                         │
│                  │  /api/upload     │                         │
│                  │  /api/clear      │                         │
│                  └──────────────────┘                         │
└─────────────────────────────────────────────────────────────────┘</pre></div>
      <table class="arch-table">
        <thead><tr><th>Component</th><th>Technology</th><th>Role</th></tr></thead>
        <tbody>
          <tr><td>Embeddings</td><td>sentence-transformers/all-MiniLM-L6-v2</td><td>Local, no external API, 384-dim vectors</td></tr>
          <tr><td>Vector Store</td><td>faiss-cpu (IndexFlatL2)</td><td>Millisecond ANN search, pre-baked in Docker image</td></tr>
          <tr><td>LLM</td><td>Groq · LLaMA 3.3 70B Versatile</td><td>Sub-second generation via Groq API</td></tr>
          <tr><td>RAG Orchestration</td><td>LangChain 0.3 ConversationalRetrievalChain</td><td>Multi-turn with condense + QA prompts</td></tr>
          <tr><td>Memory</td><td>ConversationBufferWindowMemory (k=6)</td><td>Per-session isolated conversation history</td></tr>
          <tr><td>Guardrails</td><td>Keyword scan + regex off-topic detection</td><td>Domain relevance enforcement</td></tr>
          <tr><td>Confidence</td><td>Cosine similarity + exponential decay weights</td><td>Hallucination risk flagging per query</td></tr>
          <tr><td>Backend</td><td>Flask 3.x</td><td>REST API: /api/chat, /api/upload, /api/clear</td></tr>
          <tr><td>Frontend</td><td>Vanilla HTML / CSS / JS (inline)</td><td>Dark financial theme, tab-based dashboard</td></tr>
          <tr><td>Deployment</td><td>Dockerfile → HuggingFace Spaces Docker SDK</td><td>Port 7860, non-root user</td></tr>
        </tbody>
      </table>
    </div>
  </section>

  <script>
    "use strict";
    const EXAMPLES = [
      "What is FDIC deposit insurance and what is the coverage limit?",
      "Explain term life vs whole life insurance.",
      "What does a company's balance sheet show?",
      "What is the Basel III CET1 capital ratio?",
      "How does AML compliance work in banks?",
      "What is a deductible in health insurance?",
      "What is credit risk and how do banks manage it?",
      "What is required in an SEC Form 10-K?",
    ];

    const chatMessages  = document.getElementById("chat-messages");
    const questionInput = document.getElementById("question-input");
    const askBtn        = document.getElementById("ask-btn");
    const clearBtn      = document.getElementById("clear-btn");
    const confidenceBox = document.getElementById("confidence-box");
    const sourcesBox    = document.getElementById("sources-box");
    const uploadZone    = document.getElementById("upload-zone");
    const pdfInput      = document.getElementById("pdf-input");
    const uploadBtn     = document.getElementById("upload-btn");
    const uploadStatus  = document.getElementById("upload-status");

    // Tabs
    document.querySelectorAll(".tab-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach(s => s.classList.remove("active"));
        btn.classList.add("active");
        document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
      });
    });

    // Example chips
    EXAMPLES.forEach(q => {
      const chip = document.createElement("button");
      chip.className = "example-chip";
      chip.textContent = q;
      chip.addEventListener("click", () => { questionInput.value = q; questionInput.focus(); });
      document.getElementById("examples-list").appendChild(chip);
    });

    // Chat helpers
    function appendMessage(role, text) {
      const row = document.createElement("div");
      row.className = `msg ${role}-msg`;
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = text;
      row.appendChild(bubble);
      chatMessages.appendChild(row);
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    function showTyping() {
      const row = document.createElement("div");
      row.className = "msg bot-msg"; row.id = "typing-row";
      row.innerHTML = '<div class="bubble typing-indicator"><span class="dot"></span><span class="dot"></span><span class="dot"></span></div>';
      chatMessages.appendChild(row);
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }
    function removeTyping() { const el = document.getElementById("typing-row"); if (el) el.remove(); }

    function renderSidebar(data) {
      confidenceBox.classList.remove("muted");
      confidenceBox.innerHTML = `<div class="confidence-badge">${data.confidence_badge}</div><div class="confidence-note">${data.confidence_note}</div>`;
      if (!data.sources || data.sources.length === 0) {
        sourcesBox.innerHTML = '<span class="muted" style="font-style:italic;">No sources retrieved.</span>';
      } else {
        sourcesBox.classList.remove("muted");
        sourcesBox.innerHTML = data.sources.map((s, i) => `<div class="source-item"><span style="color:var(--txt3)">${i+1}.</span><code>${s.source}</code>${s.page ? `<span class="source-page">${s.page}</span>` : ""}</div>`).join("");
      }
    }

    // Send question
    async function sendQuestion() {
      const question = questionInput.value.trim();
      if (!question) return;
      appendMessage("user", question);
      questionInput.value = "";
      askBtn.disabled = true;
      showTyping();
      try {
        const res = await fetch("/api/chat", { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({question}) });
        const data = await res.json();
        removeTyping();
        appendMessage("bot", data.answer);
        if (!data.error || data.error === "guardrail") renderSidebar(data);
      } catch (err) {
        removeTyping();
        appendMessage("bot", "⚠️ Network error. Please check your connection and try again.");
      } finally {
        askBtn.disabled = false;
        questionInput.focus();
      }
    }
    askBtn.addEventListener("click", sendQuestion);
    questionInput.addEventListener("keydown", e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuestion(); } });

    // Clear
    clearBtn.addEventListener("click", async () => {
      try { await fetch("/api/clear", {method: "POST"}); } catch (_) {}
      chatMessages.innerHTML = "";
      appendMessage("bot", "Conversation cleared. Feel free to ask a new question!");
      confidenceBox.className = "sidebar-content muted";
      confidenceBox.textContent = "Ask a question to see retrieval confidence.";
      sourcesBox.className = "sidebar-content muted";
      sourcesBox.textContent = "Source citations appear here after each answer.";
    });

    // Upload
    uploadZone.addEventListener("click", () => pdfInput.click());
    uploadZone.addEventListener("dragover", e => { e.preventDefault(); uploadZone.classList.add("drag-over"); });
    uploadZone.addEventListener("dragleave", () => uploadZone.classList.remove("drag-over"));
    uploadZone.addEventListener("drop", e => { e.preventDefault(); uploadZone.classList.remove("drag-over"); if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });
    pdfInput.addEventListener("change", () => { if (pdfInput.files[0]) setFile(pdfInput.files[0]); });
    function setFile(file) {
      document.getElementById("upload-placeholder").classList.add("hidden");
      const sel = document.getElementById("upload-selected");
      sel.classList.remove("hidden");
      sel.innerHTML = `<span style="color:var(--accent)">📄 ${file.name}</span><span style="color:var(--txt3);font-size:.8rem;margin-left:8px;">(${(file.size/1024/1024).toFixed(2)} MB)</span>`;
    }
    uploadBtn.addEventListener("click", async () => {
      const file = pdfInput.files[0];
      if (!file) { setStatus("⚠️ Please select a PDF file first.", false); return; }
      uploadBtn.disabled = true;
      setStatus("⏳ Uploading and embedding...", null);
      const fd = new FormData();
      fd.append("file", file);
      try {
        const res = await fetch("/api/upload", {method: "POST", body: fd});
        const data = await res.json();
        setStatus(data.message, res.ok);
      } catch (err) {
        setStatus("❌ Upload failed. Please try again.", false);
      } finally { uploadBtn.disabled = false; }
    });
    function setStatus(msg, ok) {
      uploadStatus.textContent = msg;
      uploadStatus.classList.remove("muted");
      uploadStatus.style.color = ok === true ? "var(--green)" : ok === false ? "var(--red)" : "var(--amber)";
    }
  </script>
</body>
</html>"""


# ══════════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════════

def _get_sid() -> str:
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())
    return session["sid"]


@app.route("/")
def index():
    _get_sid()
    return render_template_string(_PAGE)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Empty question"}), 400
    sid = _get_sid()
    logger.info("[%s] Q: %s", sid[:8], question[:120])
    return jsonify(pipeline_query(question, sid))


@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"message": "No file provided."}), 400
    f = request.files["file"]
    if not f.filename or not f.filename.lower().endswith(".pdf"):
        return jsonify({"message": "Only PDF files are supported."}), 400
    dest_dir = Path(UPLOADED_DOCS_PATH)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f.filename
    f.save(str(dest))
    ok = add_document_to_index(str(dest))
    msg = f"✅ {f.filename} was embedded and added to the knowledge base." if ok else f"⚠️ Could not process {f.filename}. Ensure it is a valid, non-encrypted PDF."
    return jsonify({"message": msg})


@app.route("/api/clear", methods=["POST"])
def clear():
    sid = session.get("sid")
    if sid:
        clear_session_memory(sid)
    return jsonify({"message": "Conversation cleared."})


# ══════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if not Path(FAISS_INDEX_PATH, "index.faiss").exists():
        logger.warning("FAISS index not found — running ingestion now...")
        from ingest import run_ingestion
        run_ingestion()
    app.run(host="0.0.0.0", port=7860, debug=False)