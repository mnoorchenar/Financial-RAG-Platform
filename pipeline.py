"""
pipeline.py — Unified RAG pipeline for Financial-RAG-Platform.

Consolidates: guardrails, confidence scoring, session memory,
ConversationalRetrievalChain, source formatting, and evaluation
into a single file for easier debugging.

Public API (used by app.py):
    query(question, session_id)       -> dict
    add_documents_to_index(file_path) -> bool
    clear_memory(session_id)          -> None
"""

import re
import logging
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

from config import (
    GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_TOKENS, GROQ_TEMPERATURE,
    EMBEDDING_MODEL, EMBEDDING_DEVICE, FAISS_INDEX_PATH, TOP_K_DOCS,
    MEMORY_WINDOW_K, CONFIDENCE_HIGH, CONFIDENCE_MEDIUM,
    FINANCE_KEYWORDS, GUARDRAIL_MIN_KEYWORD_MATCH,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — GUARDRAILS
# ══════════════════════════════════════════════════════════════════════

_OFF_TOPIC_PATTERNS = re.compile(
    r"\b(recipe|bake|cooking|chef|cuisine|ingredient|meal prep"
    r"|movie|film|series|actor|actress|oscar|netflix|streaming"
    r"|soccer|basketball|baseball|nba|nfl|fifa|cricket score|esport"
    r"|celebrity|gossip|makeup|skincare|fashion week|beauty"
    r"|weather forecast|rainfall|hurricane track|tornado warning"
    r"|alien|ufo|paranormal|conspiracy theory|flat earth"
    r"|song lyrics|poem|novel|fiction|fan fiction"
    r"|workout routine|yoga pose|diet plan|calorie count)\b",
    re.IGNORECASE,
)

_REJECTION = (
    "⚠️ **Out of Scope** — This platform is specialised in **banking and insurance** topics "
    "(FDIC · NAIC · SEC · Basel III). Your question about *{topic}* falls outside this domain."
)


def _check_query(query: str) -> Tuple[bool, str]:
    """Return (is_allowed, reason). Reason is 'on_topic' or a rejection message."""
    if not query or len(query.strip()) < 4:
        return False, "Please enter a valid question (minimum 4 characters)."
    q = query.strip()
    # Fast-pass: finance keyword found
    q_lower = q.lower()
    if sum(1 for kw in FINANCE_KEYWORDS if kw in q_lower) >= GUARDRAIL_MIN_KEYWORD_MATCH:
        return True, "on_topic"
    # Reject clear off-topic
    match = _OFF_TOPIC_PATTERNS.search(q)
    if match:
        return False, _REJECTION.format(topic=match.group(0))
    # Borderline — allow through (may be a follow-up)
    return True, "borderline"


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — CONFIDENCE SCORING
# ══════════════════════════════════════════════════════════════════════

def _l2_to_cosine(l2_distances: list) -> list:
    """Convert FAISS L2 distances to cosine similarities: cos = 1 - d²/2."""
    return [max(0.0, 1.0 - (d ** 2) / 2.0) for d in l2_distances]


def _compute_confidence(cosine_scores: list) -> Tuple[float, str, str]:
    """Exponential-decay weighted mean of cosine scores → HIGH/MEDIUM/LOW."""
    if not cosine_scores:
        return 0.0, "LOW", "🔴 Low Confidence"
    scores = np.clip(np.array(cosine_scores), 0.0, 1.0)
    weights = np.exp(-np.arange(len(scores)) * 0.6)
    weights /= weights.sum()
    weighted = float(np.dot(weights, scores))
    if weighted >= CONFIDENCE_HIGH:
        return weighted, "HIGH", "🟢 High Confidence"
    elif weighted >= CONFIDENCE_MEDIUM:
        return weighted, "MEDIUM", "🟡 Medium Confidence"
    return weighted, "LOW", "🔴 Low Confidence"


def _confidence_note(score: float, label: str) -> str:
    """Human-readable markdown note for the confidence level."""
    pct = f"{score:.0%}"
    return {
        "HIGH":   f"_Retrieval confidence: **{pct}** — Strong match. Answer well-supported by the knowledge base._",
        "MEDIUM": f"_Retrieval confidence: **{pct}** — Partial match. Consider verifying with primary sources._",
        "LOW":    f"_Retrieval confidence: **{pct}** — Weak match. Consult authoritative financial documents directly._",
    }.get(label, "")


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — SESSION MEMORY
# ══════════════════════════════════════════════════════════════════════

_LOCK: threading.Lock = threading.Lock()
_SESSIONS: dict = {}


def _get_memory(session_id: str) -> ConversationBufferWindowMemory:
    """Lazily create and return a per-session ConversationBufferWindowMemory."""
    with _LOCK:
        if session_id not in _SESSIONS:
            _SESSIONS[session_id] = ConversationBufferWindowMemory(
                memory_key="chat_history",
                return_messages=True,
                k=MEMORY_WINDOW_K,
                output_key="answer",
            )
        return _SESSIONS[session_id]


def clear_memory(session_id: str) -> None:
    """Wipe the conversation history for the given session."""
    with _LOCK:
        if session_id in _SESSIONS:
            _SESSIONS[session_id].clear()


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — SINGLETONS (embeddings, vectorstore, LLM)
# ══════════════════════════════════════════════════════════════════════

_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[FAISS] = None
_llm: Optional[ChatGroq] = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _get_vectorstore() -> FAISS:
    global _vectorstore
    if _vectorstore is None:
        index_path = Path(FAISS_INDEX_PATH)
        if not (index_path / "index.faiss").exists():
            raise FileNotFoundError(
                f"FAISS index not found at '{FAISS_INDEX_PATH}'. "
                "Run  python ingest.py  to build it first."
            )
        logger.info("Loading FAISS index from %s", FAISS_INDEX_PATH)
        _vectorstore = FAISS.load_local(
            FAISS_INDEX_PATH,
            _get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _vectorstore


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            max_tokens=GROQ_MAX_TOKENS,
            temperature=GROQ_TEMPERATURE,
        )
    return _llm


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — PROMPTS
# ══════════════════════════════════════════════════════════════════════

_CONDENSE_PROMPT = PromptTemplate.from_template(
    """Given the conversation history and a follow-up question, rewrite the follow-up
as a fully self-contained question scoped to banking, insurance, and financial services.
Do not answer the question.

Conversation History:
{chat_history}

Follow-Up Question: {question}

Standalone Question:"""
)

_QA_PROMPT = PromptTemplate.from_template(
    """You are a precise and professional AI assistant for the Financial Document Intelligence
Platform, specialised in banking, insurance, and financial services. Answer using only
the provided context. Never fabricate regulatory figures, policy details, or legal requirements.

Rules:
- Ground every claim in the context below.
- Cite the source document name when referencing specific information.
- State clearly when context is insufficient.
- Keep answers concise and professional.

Context:
{context}

Question: {question}

Answer:"""
)


# ══════════════════════════════════════════════════════════════════════
# SECTION 6 — SOURCE FORMATTER
# ══════════════════════════════════════════════════════════════════════

def _format_sources(source_docs: list) -> str:
    """Deduplicated markdown citation list from retrieved documents."""
    if not source_docs:
        return "_No source documents retrieved._"
    seen, lines = set(), ["**📚 Sources:**"]
    for i, doc in enumerate(source_docs, 1):
        source = doc.metadata.get("source_file", doc.metadata.get("source", "Unknown"))
        page = doc.metadata.get("page", "")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        page_ref = f" — p. {int(page) + 1}" if str(page).lstrip("-").isdigit() else ""
        lines.append(f"{i}. `{source}`{page_ref}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════
# SECTION 7 — PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def query(question: str, session_id: str) -> dict:
    """
    Run the full RAG pipeline for a user question.

    Returns dict with keys:
        answer, sources, confidence_score, confidence_label,
        confidence_badge, confidence_note, error
    """
    blank = {
        "answer": "", "sources": "", "confidence_score": 0.0,
        "confidence_label": "LOW", "confidence_badge": "🔴 Low Confidence",
        "confidence_note": "", "error": None,
    }

    # ── Guardrail ──────────────────────────────────────────────────────
    is_allowed, reason = _check_query(question)
    if not is_allowed:
        blank["answer"] = reason
        blank["error"] = "guardrail_rejection"
        return blank

    try:
        # ── Confidence (separate FAISS call) ───────────────────────────
        raw = _get_vectorstore().similarity_search_with_score(question, k=TOP_K_DOCS)
        cosine = _l2_to_cosine([float(s) for _, s in raw])
        conf_score, conf_label, conf_badge = _compute_confidence(cosine)
        conf_note = _confidence_note(conf_score, conf_label)

        # ── RAG Chain ──────────────────────────────────────────────────
        retriever = _get_vectorstore().as_retriever(
            search_type="mmr",
            search_kwargs={"k": TOP_K_DOCS, "fetch_k": TOP_K_DOCS * 3, "lambda_mult": 0.7},
        )
        chain = ConversationalRetrievalChain.from_llm(
            llm=_get_llm(),
            retriever=retriever,
            memory=_get_memory(session_id),
            return_source_documents=True,
            condense_question_prompt=_CONDENSE_PROMPT,
            combine_docs_chain_kwargs={"prompt": _QA_PROMPT},
            verbose=False,
        )
        result = chain({"question": question})

        return {
            "answer": result.get("answer", "Unable to generate an answer."),
            "sources": _format_sources(result.get("source_documents", [])),
            "confidence_score": conf_score,
            "confidence_label": conf_label,
            "confidence_badge": conf_badge,
            "confidence_note": conf_note,
            "error": None,
        }

    except Exception as exc:
        logger.error("Pipeline error [%s]: %s", session_id[:8], exc, exc_info=True)
        blank["answer"] = (
            "⚠️ Pipeline error. Check that `GROQ_API_KEY` is set and "
            "the FAISS index exists (`python ingest.py`)."
        )
        blank["error"] = str(exc)
        return blank


def add_documents_to_index(file_path: str) -> bool:
    """Embed a new PDF and merge it into the live FAISS index."""
    try:
        from ingest import load_document, chunk_documents
        docs = load_document(file_path)
        if not docs:
            return False
        chunks = chunk_documents(docs)
        if not chunks:
            return False
        vs = _get_vectorstore()
        new_store = FAISS.from_documents(chunks, _get_embeddings())
        vs.merge_from(new_store)
        vs.save_local(FAISS_INDEX_PATH)
        logger.info("Added %d chunks from '%s'", len(chunks), Path(file_path).name)
        return True
    except Exception as exc:
        logger.error("Failed to add document: %s", exc, exc_info=True)
        return False


# ══════════════════════════════════════════════════════════════════════
# SECTION 8 — EVALUATION (optional, call manually for quality checks)
# ══════════════════════════════════════════════════════════════════════

def evaluate_response(question: str, answer: str, retrieved_chunks: list, confidence_score: float) -> dict:
    """RAGAS-inspired local quality metrics — no external API needed."""
    def token_overlap(a: str, b: str, min_len: int = 3) -> float:
        ta = set(re.findall(rf"\b\w{{{min_len},}}\b", a.lower()))
        tb = set(re.findall(rf"\b\w{{{min_len},}}\b", b.lower()))
        return len(ta & tb) / len(ta) if ta else 0.0

    relevance = min(token_overlap(question, answer), 1.0)
    precision = (sum(1 for c in retrieved_chunks if token_overlap(question, c) > 0.2) / len(retrieved_chunks)) if retrieved_chunks else 0.0
    hallucination_pats = [r"\bi think\b", r"\bin my opinion\b", r"\bbased on my training\b", r"\bi believe\b"]
    faithful = not any(re.search(p, answer.lower()) for p in hallucination_pats)
    composite = float(np.mean([relevance, precision, confidence_score, float(faithful)]))

    return {
        "answer_relevance": round(relevance, 3),
        "context_precision": round(precision, 3),
        "faithfulness": faithful,
        "retrieval_confidence": round(confidence_score, 3),
        "composite_score": round(composite, 3),
    }