"""
rag_pipeline.py — Core Retrieval-Augmented Generation pipeline for FinRAG.

Singleton-initialises the embedding model, FAISS vector store, and Groq LLM
on first use, then routes every incoming query through a LangChain
ConversationalRetrievalChain with per-session memory. Confidence scoring
runs via a separate similarity_search_with_score call to avoid interfering
with the main MMR retrieval chain.

Public API:
    query(question, session_id)          -> dict
    add_documents_to_index(file_path)    -> bool
"""

import logging
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.schema import Document

from config import (
    GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_TOKENS, GROQ_TEMPERATURE,
    EMBEDDING_MODEL, EMBEDDING_DEVICE, FAISS_INDEX_PATH, TOP_K_DOCS,
)
from memory_manager import get_memory, clear_memory
from confidence import compute_confidence, l2_to_cosine, format_confidence_note
from guardrails import check_query

logger = logging.getLogger(__name__)

# ── Module-level singletons ────────────────────────────────────────────────────
_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[FAISS] = None
_llm: Optional[ChatGroq] = None


# ── Singleton Initialisers ─────────────────────────────────────────────────────

def _get_embeddings() -> HuggingFaceEmbeddings:
    """Lazily initialise and cache the local HuggingFace embedding model."""
    global _embeddings
    if _embeddings is None:
        logger.info("Initialising embedding model: %s", EMBEDDING_MODEL)
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": EMBEDDING_DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def _get_vectorstore() -> FAISS:
    """Lazily load the pre-built FAISS index from disk."""
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
        logger.info("FAISS index loaded successfully.")
    return _vectorstore


def _get_llm() -> ChatGroq:
    """Lazily initialise the Groq LLM client."""
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            max_tokens=GROQ_MAX_TOKENS,
            temperature=GROQ_TEMPERATURE,
        )
    return _llm


# ── Prompt Templates ───────────────────────────────────────────────────────────

_CONDENSE_PROMPT = PromptTemplate.from_template(
    """Given the conversation history below and a follow-up question, rewrite the
follow-up as a fully self-contained question in the context of banking, insurance,
and financial services. Do not answer the question.

Conversation History:
{chat_history}

Follow-Up Question: {question}

Standalone Question:"""
)

_QA_PROMPT = PromptTemplate.from_template(
    """You are a precise and professional AI assistant for the Financial Document Intelligence Platform,
specialised in banking, insurance, and financial services. You answer questions strictly using the
provided context. Never fabricate regulatory figures, policy details, or legal requirements.

Rules:
- Ground every claim in the context below.
- Cite the source document name when referencing specific information.
- If context is insufficient, clearly state that and suggest consulting primary sources.
- Format monetary values, percentages, and regulatory thresholds clearly.
- Keep answers concise and professional.

Context:
{context}

Question: {question}

Answer:"""
)


# ── Chain Builder ──────────────────────────────────────────────────────────────

def _build_chain(session_id: str) -> ConversationalRetrievalChain:
    """
    Assemble a ConversationalRetrievalChain for the given session.

    Args:
        session_id: Session identifier used to look up persistent memory.

    Returns:
        Configured ConversationalRetrievalChain instance.
    """
    retriever = _get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K_DOCS, "fetch_k": TOP_K_DOCS * 3, "lambda_mult": 0.7},
    )
    return ConversationalRetrievalChain.from_llm(
        llm=_get_llm(),
        retriever=retriever,
        memory=get_memory(session_id),
        return_source_documents=True,
        condense_question_prompt=_CONDENSE_PROMPT,
        combine_docs_chain_kwargs={"prompt": _QA_PROMPT},
        verbose=False,
    )


# ── Source Formatter ───────────────────────────────────────────────────────────

def format_sources(source_docs: list) -> str:
    """
    Render retrieved source documents as a deduplicated markdown citation list.

    Args:
        source_docs: List of Document objects from the retrieval chain result.

    Returns:
        Markdown-formatted string with one citation line per unique (source, page).
    """
    if not source_docs:
        return "_No source documents retrieved._"

    seen = set()
    lines = ["**📚 Sources:**"]
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


# ── Confidence Helper ──────────────────────────────────────────────────────────

def _score_confidence(question: str) -> tuple:
    """
    Run a dedicated similarity search to obtain retrieval scores for confidence
    estimation without interfering with the MMR-based main retrieval.

    Args:
        question: The user question string.

    Returns:
        Tuple (confidence_score, confidence_label, confidence_badge, note).
    """
    raw_results = _get_vectorstore().similarity_search_with_score(question, k=TOP_K_DOCS)
    raw_scores = [float(score) for _, score in raw_results]
    cosine_scores = l2_to_cosine(raw_scores)
    conf_score, conf_label, conf_badge = compute_confidence(cosine_scores)
    conf_note = format_confidence_note(conf_score, conf_label)
    return conf_score, conf_label, conf_badge, conf_note


# ── Public API ─────────────────────────────────────────────────────────────────

def query(question: str, session_id: str) -> dict:
    """
    Process a user question through the full FinRAG pipeline.

    Pipeline:
      1. Guardrail check — reject off-topic queries immediately.
      2. Confidence scoring — separate FAISS call for retrieval scores.
      3. RAG chain — ConversationalRetrievalChain with Groq LLM.
      4. Format sources and assemble result dict.

    Args:
        question: The user's natural language question.
        session_id: Unique session identifier for memory isolation.

    Returns:
        Dict with keys: answer, sources, confidence_score, confidence_label,
        confidence_badge, confidence_note, error.
    """
    blank_result = {
        "answer": "",
        "sources": "",
        "confidence_score": 0.0,
        "confidence_label": "LOW",
        "confidence_badge": "🔴 Low Confidence",
        "confidence_note": "",
        "error": None,
    }

    is_allowed, reason = check_query(question)
    if not is_allowed:
        blank_result["answer"] = reason
        blank_result["error"] = "guardrail_rejection"
        return blank_result

    try:
        conf_score, conf_label, conf_badge, conf_note = _score_confidence(question)
        chain = _build_chain(session_id)
        result = chain({"question": question})

        answer = result.get("answer", "Unable to generate an answer.")
        source_docs = result.get("source_documents", [])

        return {
            "answer": answer,
            "sources": format_sources(source_docs),
            "confidence_score": conf_score,
            "confidence_label": conf_label,
            "confidence_badge": conf_badge,
            "confidence_note": conf_note,
            "error": None,
        }

    except Exception as exc:
        logger.error("RAG pipeline error for session %s: %s", session_id[:8], exc, exc_info=True)
        blank_result["answer"] = (
            "⚠️ A pipeline error occurred. Please verify your `GROQ_API_KEY` is set "
            "correctly and that the FAISS index exists (`python ingest.py`)."
        )
        blank_result["error"] = str(exc)
        return blank_result


def add_documents_to_index(file_path: str) -> bool:
    """
    Embed a new PDF at runtime and merge it into the live FAISS index.

    Args:
        file_path: Absolute path to the uploaded PDF.

    Returns:
        True on success, False on failure.
    """
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
        logger.info("Added %d chunks from '%s' to the live index.", len(chunks), Path(file_path).name)
        return True

    except Exception as exc:
        logger.error("Failed to add document to index: %s", exc, exc_info=True)
        return False
