"""
app.py — Gradio user interface and application entry point for FinRAG.

Implements a professional dark-themed financial intelligence dashboard with:
  - Multi-turn conversational chat backed by LangChain ConversationalRetrievalChain
  - Live source document citations rendered alongside each answer
  - Colour-coded retrieval confidence indicator (green / amber / red)
  - Runtime PDF upload to extend the FAISS knowledge base without restart
  - Per-session memory isolation via unique UUID session IDs

Run locally:
    python app.py
"""

import uuid
import logging
from pathlib import Path

import gradio as gr

from config import UPLOADED_DOCS_PATH
from pipeline import query as rag_query, add_documents_to_index, clear_memory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── CSS: Dark Financial Theme ──────────────────────────────────────────────────
_DISPLAY_TITLE = "Financial Document Intelligence Platform"
_DISPLAY_SUBTITLE = "Retrieval-Augmented Generation · Banking &amp; Insurance · LangChain · FAISS · LLaMA 3.3 70B"

_CSS = """
:root {
    --bg0: #070c18; --bg1: #0f1929; --bg2: #1a2236; --bg3: #1e2d47;
    --accent: #3b82f6; --accent2: #4f46e5; --accent3: #06b6d4;
    --green: #22c55e; --amber: #f59e0b; --red: #ef4444;
    --txt1: #e8eaf6; --txt2: #94a3b8; --txt3: #475569;
    --border: #1e3a5f;
}
body, .gradio-container { background: var(--bg0) !important; color: var(--txt1) !important; font-family: 'Inter', system-ui, sans-serif !important; }
.fin-header { background: linear-gradient(135deg, var(--bg1) 0%, var(--bg2) 60%, var(--bg1) 100%); border: 1px solid var(--border); border-radius: 16px; padding: 28px 36px; margin-bottom: 18px; text-align: center; box-shadow: 0 4px 32px rgba(59,130,246,0.12); }
.fin-title { font-size: 2.4rem; font-weight: 900; background: linear-gradient(90deg, var(--accent), var(--accent2), var(--accent3)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin: 0; letter-spacing: -0.5px; }
.fin-sub { color: var(--txt2); font-size: 0.9rem; margin-top: 8px; }
.fin-pills { display: flex; gap: 8px; justify-content: center; flex-wrap: wrap; margin-top: 14px; }
.pill { background: var(--bg3); border: 1px solid var(--border); border-radius: 20px; padding: 3px 12px; font-size: 0.75rem; color: var(--txt2); }
.side-label { color: var(--txt2); font-size: 0.82rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; padding-left: 2px; }
.gradio-chatbot { background: var(--bg1) !important; border: 1px solid var(--border) !important; border-radius: 14px !important; }
.message.user .bubble-wrap { background: linear-gradient(135deg, #1e3a5f, #162d4a) !important; border: 1px solid #2563eb !important; border-radius: 12px 12px 4px 12px !important; }
.message.bot .bubble-wrap { background: var(--bg2) !important; border: 1px solid var(--border) !important; border-radius: 12px 12px 12px 4px !important; }
textarea, input[type=text] { background: var(--bg1) !important; border: 1px solid var(--border) !important; color: var(--txt1) !important; border-radius: 10px !important; }
textarea:focus, input[type=text]:focus { border-color: var(--accent) !important; box-shadow: 0 0 0 2px rgba(59,130,246,0.2) !important; }
.btn-ask { background: linear-gradient(135deg, var(--accent), var(--accent2)) !important; color: #fff !important; border: none !important; border-radius: 10px !important; font-weight: 700 !important; font-size: 1rem !important; transition: opacity .15s, transform .15s !important; }
.btn-ask:hover { opacity: .88 !important; transform: translateY(-1px) !important; box-shadow: 0 6px 18px rgba(59,130,246,.35) !important; }
.btn-clear { background: var(--bg3) !important; color: var(--txt2) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
.info-box { background: var(--bg2) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; padding: 14px 16px !important; min-height: 60px; }
.upload-zone { background: var(--bg1) !important; border: 2px dashed var(--border) !important; border-radius: 12px !important; color: var(--txt2) !important; }
.tab-nav button { background: var(--bg1) !important; color: var(--txt2) !important; border: 1px solid var(--border) !important; border-radius: 8px 8px 0 0 !important; }
.tab-nav button.selected { background: var(--bg2) !important; color: var(--accent) !important; border-bottom-color: var(--bg2) !important; }
::-webkit-scrollbar { width: 5px; } ::-webkit-scrollbar-track { background: var(--bg0); } ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; } ::-webkit-scrollbar-thumb:hover { background: var(--accent); }
"""

_EXAMPLE_QUESTIONS = [
    "What is FDIC deposit insurance and what is the coverage limit per depositor?",
    "Explain the difference between term life insurance and whole life insurance.",
    "What does a company's balance sheet show and how is it structured?",
    "What is the Basel III CET1 capital ratio and why does it matter?",
    "How does Anti-Money Laundering (AML) compliance work in banks?",
    "What is a deductible in health insurance and how does it interact with coinsurance?",
    "What is credit risk and what tools do banks use to manage it?",
    "What information is required in an SEC Form 10-K annual report?",
    "Explain the Net Stable Funding Ratio (NSFR) under Basel III.",
    "What is reinsurance and why do insurance companies use it?",
]


# ── Event Handlers ─────────────────────────────────────────────────────────────

def handle_question(question: str, chat_history: list, session_id: str) -> tuple:
    """
    Route a user question through the RAG pipeline and update the chat UI.

    Args:
        question: The user's typed question.
        chat_history: Current list of (human_msg, ai_msg) tuples.
        session_id: Per-session UUID for memory isolation.

    Returns:
        Tuple of (updated_chat_history, sources_markdown, confidence_markdown, cleared_input).
    """
    if not question or not question.strip():
        return chat_history, "", "", ""

    logger.info("[%s] Q: %s", session_id[:8], question[:120])
    result = rag_query(question, session_id)

    answer = result["answer"]
    note = result["confidence_note"]
    full_answer = f"{answer}\n\n---\n{note}" if note else answer

    updated_history = chat_history + [(question, full_answer)]
    sources_md = result["sources"] or "_No sources retrieved._"
    confidence_md = f"### {result['confidence_badge']}\n\n{result['confidence_note']}"

    return updated_history, sources_md, confidence_md, ""


def handle_upload(file_obj) -> str:
    """
    Accept a user-uploaded PDF and merge it into the live FAISS index.

    Args:
        file_obj: Gradio filepath object from the file upload component.

    Returns:
        Status message string (markdown).
    """
    if file_obj is None:
        return "⚠️ No file selected. Please choose a PDF to upload."
    try:
        dest_dir = Path(UPLOADED_DOCS_PATH)
        dest_dir.mkdir(parents=True, exist_ok=True)
        src = Path(file_obj)
        dest = dest_dir / src.name
        dest.write_bytes(src.read_bytes())
        success = add_documents_to_index(str(dest))
        if success:
            return f"✅ **{src.name}** was embedded and added to the knowledge base."
        return f"⚠️ Could not process **{src.name}**. Ensure it is a valid, non-encrypted PDF."
    except Exception as exc:
        logger.error("Upload handler error: %s", exc)
        return f"❌ Upload error: {exc}"


def handle_clear(session_id: str) -> tuple:
    """
    Clear the conversation memory for the current session.

    Args:
        session_id: Session UUID to clear.

    Returns:
        Tuple of (empty_chat_history, cleared_sources_text, cleared_confidence_text).
    """
    clear_memory(session_id)
    return [], "_Conversation cleared._", ""


# ── UI Builder ─────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    """
    Construct and return the complete Gradio Blocks application.

    Returns:
        Configured gr.Blocks instance ready for .launch().
    """
    with gr.Blocks(
        css=_CSS,
        title="Financial Document Intelligence Platform — RAG, LangChain, FAISS & LLaMA 3.3 70B",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.blue,
            secondary_hue=gr.themes.colors.indigo,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#070c18",
            block_background_fill="#0f1929",
            block_border_color="#1e3a5f",
            button_primary_background_fill="linear-gradient(135deg,#3b82f6,#4f46e5)",
            button_primary_text_color="#ffffff",
        ),
    ) as demo:

        # Per-session state — a new UUID is created for each browser tab
        session_id = gr.State(lambda: str(uuid.uuid4()))

        # ── Header ─────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="fin-header">
            <h1 class="fin-title">🏦 Financial Document Intelligence Platform</h1>
            <p class="fin-sub">Retrieval-Augmented Generation · Banking &amp; Insurance · LangChain · FAISS · LLaMA 3.3 70B</p>
            <p class="fin-sub" style="margin-top:4px; font-size:0.8rem;">
                Multi-turn Conversational Memory · Domain Guardrails · Confidence Scoring · Runtime PDF Ingestion
            </p>
            <div class="fin-pills">
                <span class="pill">📄 FDIC</span>
                <span class="pill">🛡️ NAIC</span>
                <span class="pill">📊 SEC</span>
                <span class="pill">🏛️ Basel III</span>
                <span class="pill">🔍 MMR Retrieval</span>
                <span class="pill">💬 Multi-turn Memory</span>
            </div>
        </div>
        """)

        with gr.Tabs():

            # ── TAB 1: Chat ────────────────────────────────────────────────────
            with gr.Tab("💬 Ask FinRAG"):
                with gr.Row(equal_height=False):

                    with gr.Column(scale=3, min_width=400):
                        chatbot = gr.Chatbot(
                            value=[],
                            height=480,
                            show_label=False,
                            bubble_full_width=False,
                            elem_classes=["gradio-chatbot"],
                        )
                        with gr.Row():
                            question_box = gr.Textbox(
                                placeholder="Ask about deposit insurance, Basel III, life insurance, 10-K filings...",
                                lines=2,
                                scale=5,
                                show_label=False,
                                container=False,
                            )
                            with gr.Column(scale=1, min_width=100):
                                ask_btn = gr.Button("Ask →", variant="primary", size="lg", elem_classes=["btn-ask"])
                                clear_btn = gr.Button("🗑️ Clear", size="sm", elem_classes=["btn-clear"])

                        gr.HTML("<p style='color:#475569;font-size:0.82rem;margin:10px 0 4px;'>💡 Example questions:</p>")
                        gr.Examples(
                            examples=[[q] for q in _EXAMPLE_QUESTIONS],
                            inputs=[question_box],
                            label="",
                        )

                    with gr.Column(scale=2, min_width=260):
                        gr.HTML("<div class='side-label'>📊 Confidence</div>")
                        confidence_box = gr.Markdown(
                            "_Ask a question to see retrieval confidence._",
                            elem_classes=["info-box"],
                        )
                        gr.HTML("<div class='side-label' style='margin-top:18px;'>📚 Sources</div>")
                        sources_box = gr.Markdown(
                            "_Source citations appear here after each answer._",
                            elem_classes=["info-box"],
                        )

            # ── TAB 2: Upload Documents ────────────────────────────────────────
            with gr.Tab("📤 Upload Documents"):
                gr.HTML("""
                <div style='background:#1a2236;border:1px solid #1e3a5f;border-radius:14px;padding:22px 26px;margin-bottom:18px;'>
                    <h3 style='color:#3b82f6;margin:0 0 10px;'>Extend the Knowledge Base at Runtime</h3>
                    <p style='color:#94a3b8;margin:0;font-size:0.9rem;'>
                        Upload your own financial PDFs — annual reports, prospectuses, policy documents, regulatory filings —
                        and FinRAG will embed and index them immediately. No restart required.
                    </p>
                    <ul style='color:#64748b;font-size:0.85rem;margin-top:14px;line-height:1.8;'>
                        <li>Format: PDF only · Recommended max size: 50 MB</li>
                        <li>Documents are merged into the live FAISS index via vectorstore.merge_from()</li>
                        <li>Index is re-persisted to disk after each upload</li>
                    </ul>
                </div>
                """)
                with gr.Row():
                    with gr.Column(scale=2):
                        pdf_upload = gr.File(label="Upload Financial PDF", file_types=[".pdf"], type="filepath", elem_classes=["upload-zone"])
                        upload_btn = gr.Button("📥 Add to Knowledge Base", variant="primary")
                    with gr.Column(scale=3):
                        upload_status = gr.Markdown("_No document uploaded yet._")

            # ── TAB 3: Architecture ────────────────────────────────────────────
            with gr.Tab("🏗️ Architecture"):
                gr.Markdown("""
## FinRAG — System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           FinRAG Pipeline                            │
│                                                                      │
│  ┌─────────────┐   ┌──────────────────┐   ┌───────────────────┐    │
│  │ PDF Sources │──▶│   ingest.py      │──▶│   FAISS Index     │    │
│  │ FDIC / NAIC │   │ Chunk + Embed    │   │ (baked at Docker  │    │
│  │ SEC / User  │   │ MiniLM-L6-v2     │   │  build time)      │    │
│  └─────────────┘   └──────────────────┘   └────────┬──────────┘    │
│                                                     │               │
│  ┌─────────────┐   ┌──────────────────┐   ┌────────▼──────────┐    │
│  │ guardrails  │──▶│  rag_pipeline    │◀──│  MMR Retriever    │    │
│  │ (domain     │   │  LangChain       │   │  top-k=5          │    │
│  │  filter)    │   │  Conv.Retrieval  │   └───────────────────┘    │
│  └─────────────┘   │  Chain           │                             │
│                    └────────┬─────────┘                             │
│  ┌─────────────┐            │           ┌───────────────────┐       │
│  │  memory_    │◀───────────┤           │  confidence.py    │       │
│  │  manager    │            │           │  L2→cosine score  │       │
│  │  (per-sess) │            ▼           └───────────────────┘       │
│  └─────────────┘   ┌──────────────────┐                             │
│                    │  Groq API        │                             │
│                    │  LLaMA 3.3 70B   │                             │
│                    └──────────────────┘                             │
└──────────────────────────────────────────────────────────────────────┘
```

| Component | Technology | Role |
|---|---|---|
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Local, no external API, 384-dim vectors |
| Vector Store | faiss-cpu (IndexFlatL2) | Millisecond ANN search, pre-baked in image |
| LLM | Groq · LLaMA 3.3 70B Versatile | Sub-second generation via Groq API |
| RAG Orchestration | LangChain 0.3 ConversationalRetrievalChain | Multi-turn with condense + QA prompts |
| Memory | ConversationBufferWindowMemory (k=6) | Per-session isolated history |
| Guardrails | Keyword scan + regex off-topic detection | Domain relevance enforcement |
| Confidence | Cosine similarity with exponential decay weights | Hallucination risk flagging |
| UI | Gradio 4.x Blocks + custom dark CSS | Professional financial theme |
| Deployment | Single-stage Dockerfile → HuggingFace Spaces Docker SDK | Port 7860, non-root user |
                """)

        # ── Event Wiring ───────────────────────────────────────────────────────
        submit_inputs = [question_box, chatbot, session_id]
        submit_outputs = [chatbot, sources_box, confidence_box, question_box]

        ask_btn.click(fn=handle_question, inputs=submit_inputs, outputs=submit_outputs)
        question_box.submit(fn=handle_question, inputs=submit_inputs, outputs=submit_outputs)
        clear_btn.click(fn=handle_clear, inputs=[session_id], outputs=[chatbot, sources_box, confidence_box])
        upload_btn.click(fn=handle_upload, inputs=[pdf_upload], outputs=[upload_status])

    return demo


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Auto-ingest if the FAISS index is missing (fresh clone / local dev)
    if not Path("data/faiss_index/index.faiss").exists():
        logger.warning("FAISS index not found — running ingestion pipeline now...")
        from ingest import run_ingestion
        run_ingestion()

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )