"""
evaluation.py — Lightweight RAGAS-inspired evaluation utilities for FinRAG.

Provides heuristic metrics computed locally without calling the LLM or an
external scoring API, making them suitable for rapid quality assessment during
development and live demos.

Metrics implemented:
  - answer_relevance:  Keyword overlap between question and answer.
  - context_precision: Fraction of retrieved chunks relevant to the question.
  - faithfulness_flag: Heuristic check for speculation beyond context.
  - composite_score:   Weighted mean of all metrics.
"""

import re
import numpy as np
from typing import Tuple


def answer_relevance_score(question: str, answer: str) -> float:
    """
    Estimate answer relevance via question-token coverage in the answer.

    Args:
        question: Original user question.
        answer: Generated answer string.

    Returns:
        Score in [0, 1]. Higher = answer covers more of the question's key terms.
    """
    if not answer or not question:
        return 0.0

    q_tokens = set(re.findall(r"\b\w{3,}\b", question.lower()))
    a_tokens = set(re.findall(r"\b\w{3,}\b", answer.lower()))

    if not q_tokens:
        return 0.0

    overlap_ratio = len(q_tokens & a_tokens) / len(q_tokens)
    return min(overlap_ratio, 1.0)


def context_precision_score(question: str, retrieved_chunks: list) -> float:
    """
    Estimate the fraction of retrieved chunks that are relevant to the question.

    Args:
        question: Original user question.
        retrieved_chunks: List of document chunk text strings from retrieval.

    Returns:
        Score in [0, 1]. Higher = more retrieved chunks are on-topic.
    """
    if not retrieved_chunks:
        return 0.0

    q_tokens = set(re.findall(r"\b\w{3,}\b", question.lower()))
    if not q_tokens:
        return 0.5

    relevant_count = 0
    for chunk in retrieved_chunks:
        chunk_tokens = set(re.findall(r"\b\w{3,}\b", chunk.lower()))
        if len(q_tokens & chunk_tokens) / len(q_tokens) > 0.2:
            relevant_count += 1

    return relevant_count / len(retrieved_chunks)


def faithfulness_flag(answer: str, context_chunks: list) -> bool:
    """
    Flag answers that may be speculating beyond the retrieved context.

    Args:
        answer: The generated answer.
        context_chunks: Retrieved document text chunks used as context.

    Returns:
        True if the answer appears grounded in context, False if flagged.
    """
    hallucination_patterns = [
        r"\bi think\b", r"\bin my opinion\b", r"\bgenerally speaking\b",
        r"\bcommonly known\b", r"\bbased on my training\b", r"\bi believe\b",
        r"\bit is widely known\b", r"\bmost people know\b",
    ]
    a_lower = answer.lower()
    for pat in hallucination_patterns:
        if re.search(pat, a_lower):
            return False

    context_text = " ".join(context_chunks).lower()
    ans_content_tokens = set(re.findall(r"\b\w{4,}\b", a_lower))
    ctx_tokens = set(re.findall(r"\b\w{4,}\b", context_text))

    if not ans_content_tokens:
        return True

    return (len(ans_content_tokens & ctx_tokens) / len(ans_content_tokens)) >= 0.25


def evaluate_response(question: str, answer: str, retrieved_chunks: list, confidence_score: float) -> dict:
    """
    Run all evaluation metrics and return a consolidated report dictionary.

    Args:
        question: User question.
        answer: Generated answer.
        retrieved_chunks: List of retrieved document chunk strings.
        confidence_score: Confidence score from the retrieval pipeline (0-1).

    Returns:
        Dict with keys: answer_relevance, context_precision, faithfulness,
        retrieval_confidence, composite_score.
    """
    relevance = answer_relevance_score(question, answer)
    precision = context_precision_score(question, retrieved_chunks)
    faithful = faithfulness_flag(answer, retrieved_chunks)

    composite = float(np.mean([relevance, precision, confidence_score, float(faithful)]))

    return {
        "answer_relevance": round(relevance, 3),
        "context_precision": round(precision, 3),
        "faithfulness": faithful,
        "retrieval_confidence": round(confidence_score, 3),
        "composite_score": round(composite, 3),
    }
