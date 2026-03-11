"""
pipeline.py — RAG pipeline: guardrails, confidence, session memory, LangChain chain.

Public API consumed by app.py:
    pipeline_query(question, session_id)  -> dict
    add_document_to_index(file_path)      -> bool
    clear_session_memory(session_id)      -> None
"""

import os
import re
import logging
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

from ingest import load_file, chunk_documents, get_embeddings, FAISS_INDEX_PATH

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL        = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS   = 1024
GROQ_TEMPERATURE  = 0.1
TOP_K_DOCS        = 5
MEMORY_WINDOW_K   = 6
CONFIDENCE_HIGH   = 0.58
CONFIDENCE_MEDIUM = 0.38

FINANCE_KEYWORDS = [
    "bank", "banking", "insurance", "policy", "premium", "claim", "deductible",
    "coverage", "loan", "mortgage", "credit", "interest", "investment", "portfolio",
    "asset", "liability", "equity", "dividend", "bond", "securities", "regulatory",
    "compliance", "underwriting", "reinsurance", "actuarial", "annuity", "beneficiary",
    "pension", "retirement", "fund", "audit", "balance sheet", "income statement",
    "financial", "risk", "disclosure", "sec", "fdic", "naic", "fiduciary", "deposit",
    "account", "transaction", "payment", "fraud", "money", "capital", "revenue",
    "profit", "loss", "tax", "cash", "market", "stock", "share", "indemnity",
    "surety", "guarantee", "collateral", "derivative", "hedge", "swap", "yield",
    "rate", "inflation", "liquidity", "solvency", "leverage", "default", "basel",
    "aml", "kyc", "fintech", "lending", "underwriter", "actuary", "reserve",
]

_OFF_TOPIC = re.compile(r"\b(recipe|cooking|chef|movie|film|actor|netflix|soccer|basketball|nba|nfl|celebrity|gossip|makeup|skincare|weather forecast|alien|ufo|conspiracy|song lyrics|poem|workout routine|diet plan|calorie)\b", re.IGNORECASE)

# ── Prompts ────────────────────────────────────────────────────────────────────
_CONDENSE_PROMPT = PromptTemplate.from_template("""Given the conversation history and a follow-up question, rewrite the follow-up as a fully self-contained question scoped to banking, insurance, and financial services. Do not answer the question.

Conversation History:
{chat_history}

Follow-Up Question: {question}

Standalone Question:""")

_QA_PROMPT = PromptTemplate.from_template("""You are a precise and professional AI assistant for a Financial Document Intelligence Platform specialised in banking, insurance, and financial services. Answer using only the provided context. Never fabricate regulatory figures, policy details, or legal requirements.

Rules:
- Ground every claim in the context below.
- Cite the source document name when referencing specific information.
- State clearly when context is insufficient.
- Keep answers concise and professional.

Context:
{context}

Question: {question}

Answer:""")

# ── Lazy singletons ────────────────────────────────────────────────────────────
_vectorstore: Optional[FAISS] = None
_llm: Optional[ChatGroq] = None
_embeddings = None


def _get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embeddings()
    return _embeddings


def _get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        idx = Path(FAISS_INDEX_PATH)
        if not (idx / "index.faiss").exists():
            raise FileNotFoundError(f"FAISS index not found at '{FAISS_INDEX_PATH}'. Run ingest.py first.")
        _vectorstore = FAISS.load_local(FAISS_INDEX_PATH, _get_embeddings(), allow_dangerous_deserialization=True)
    return _vectorstore


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(api_key=GROQ_API_KEY, model=GROQ_MODEL, max_tokens=GROQ_MAX_TOKENS, temperature=GROQ_TEMPERATURE)
    return _llm


# ── Session memory ─────────────────────────────────────────────────────────────
_lock = threading.Lock()
_sessions: dict = {}


def _get_memory(sid: str) -> ConversationBufferWindowMemory:
    with _lock:
        if sid not in _sessions:
            _sessions[sid] = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=MEMORY_WINDOW_K, output_key="answer")
        return _sessions[sid]


def clear_session_memory(sid: str) -> None:
    with _lock:
        if sid in _sessions:
            _sessions[sid].clear()


# ── Guardrails ─────────────────────────────────────────────────────────────────

def _check_query(question: str) -> Tuple[bool, str]:
    if not question or len(question.strip()) < 4:
        return False, "Please enter a valid question (minimum 4 characters)."
    if sum(1 for kw in FINANCE_KEYWORDS if kw in question.lower()) >= 1:
        return True, ""
    m = _OFF_TOPIC.search(question)
    if m:
        return False, f"⚠️ Out of scope — this platform covers banking and insurance topics. Your question about '{m.group(0)}' falls outside this domain."
    return True, ""


# ── Confidence scoring ─────────────────────────────────────────────────────────

def _score_confidence(l2_distances: list) -> Tuple[float, str, str]:
    cosines = [max(0.0, 1.0 - (d ** 2) / 2.0) for d in l2_distances]
    if not cosines:
        return 0.0, "LOW", "🔴 Low Confidence"
    arr = np.clip(np.array(cosines), 0.0, 1.0)
    w = np.exp(-np.arange(len(arr)) * 0.6)
    w /= w.sum()
    val = float(np.dot(w, arr))
    if val >= CONFIDENCE_HIGH:
        return val, "HIGH", "🟢 High Confidence"
    if val >= CONFIDENCE_MEDIUM:
        return val, "MEDIUM", "🟡 Medium Confidence"
    return val, "LOW", "🔴 Low Confidence"


def _confidence_note(score: float, label: str) -> str:
    pct = f"{score:.0%}"
    return {
        "HIGH":   f"Retrieval confidence: {pct} — Strong match. Answer well-supported by the knowledge base.",
        "MEDIUM": f"Retrieval confidence: {pct} — Partial match. Consider verifying with primary sources.",
        "LOW":    f"Retrieval confidence: {pct} — Weak match. Consult authoritative financial documents directly.",
    }.get(label, "")


# ── Source formatter ───────────────────────────────────────────────────────────

def _format_sources(source_docs: list) -> list:
    seen, results = set(), []
    for doc in source_docs:
        src = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", "")
        key = (src, page)
        if key in seen:
            continue
        seen.add(key)
        page_label = f"p. {int(page) + 1}" if str(page).lstrip("-").isdigit() else ""
        results.append({"source": src, "page": page_label})
    return results


# ── Public API ─────────────────────────────────────────────────────────────────

def pipeline_query(question: str, session_id: str) -> dict:
    blank = {"answer": "", "sources": [], "confidence_score": 0.0, "confidence_label": "LOW", "confidence_badge": "🔴 Low Confidence", "confidence_note": "", "error": None}
    allowed, rejection = _check_query(question)
    if not allowed:
        blank["answer"] = rejection
        blank["error"] = "guardrail"
        return blank
    try:
        raw = _get_vectorstore().similarity_search_with_score(question, k=TOP_K_DOCS)
        conf_score, conf_label, conf_badge = _score_confidence([float(s) for _, s in raw])
        retriever = _get_vectorstore().as_retriever(search_type="mmr", search_kwargs={"k": TOP_K_DOCS, "fetch_k": TOP_K_DOCS * 3, "lambda_mult": 0.7})
        chain = ConversationalRetrievalChain.from_llm(llm=_get_llm(), retriever=retriever, memory=_get_memory(session_id), return_source_documents=True, condense_question_prompt=_CONDENSE_PROMPT, combine_docs_chain_kwargs={"prompt": _QA_PROMPT}, verbose=False)
        result = chain({"question": question})
        return {"answer": result.get("answer", "Unable to generate an answer."), "sources": _format_sources(result.get("source_documents", [])), "confidence_score": conf_score, "confidence_label": conf_label, "confidence_badge": conf_badge, "confidence_note": _confidence_note(conf_score, conf_label), "error": None}
    except Exception as exc:
        logger.error("Pipeline error [%s]: %s", session_id[:8], exc, exc_info=True)
        blank["answer"] = "⚠️ Pipeline error. Check that GROQ_API_KEY is set and the FAISS index exists."
        blank["error"] = str(exc)
        return blank


def add_document_to_index(file_path: str) -> bool:
    try:
        docs = load_file(file_path)
        if not docs:
            return False
        chunks = chunk_documents(docs)
        if not chunks:
            return False
        new_store = FAISS.from_documents(chunks, _get_embeddings())
        _get_vectorstore().merge_from(new_store)
        _get_vectorstore().save_local(FAISS_INDEX_PATH)
        logger.info("Added %d chunks from '%s'", len(chunks), Path(file_path).name)
        return True
    except Exception as exc:
        logger.error("Failed to add document: %s", exc)
        return False