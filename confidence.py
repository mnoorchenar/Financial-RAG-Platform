"""
confidence.py — Retrieval confidence scoring for FinRAG.

Converts raw FAISS similarity scores into a human-interpretable confidence level.
FAISS with L2-normalised embeddings returns inner-product scores that approximate
cosine similarity in [0, 1]. This module derives a weighted score using exponential
decay weights that give higher importance to the top-ranked result.
"""

from typing import Tuple
import numpy as np

from config import CONFIDENCE_HIGH, CONFIDENCE_MEDIUM

ConfidenceLabel = str  # "HIGH" | "MEDIUM" | "LOW"


def compute_confidence(similarity_scores: list) -> Tuple[float, ConfidenceLabel, str]:
    """
    Derive a scalar confidence score and a categorical label from FAISS scores.

    Exponential decay weighting ensures the top-ranked document drives the score
    while lower-ranked documents contribute progressively less.

    Args:
        similarity_scores: Cosine similarities for top-k retrieved documents,
                           highest score first, values expected in [0, 1].

    Returns:
        Tuple of (score: float, label: str, badge: str).
    """
    if not similarity_scores:
        return 0.0, "LOW", "🔴 Low Confidence"

    scores = np.clip(np.array(similarity_scores, dtype=float), 0.0, 1.0)
    weights = np.exp(-np.arange(len(scores)) * 0.6)
    weights /= weights.sum()
    weighted = float(np.dot(weights, scores))

    if weighted >= CONFIDENCE_HIGH:
        return weighted, "HIGH", "🟢 High Confidence"
    elif weighted >= CONFIDENCE_MEDIUM:
        return weighted, "MEDIUM", "🟡 Medium Confidence"
    else:
        return weighted, "LOW", "🔴 Low Confidence"


def l2_to_cosine(l2_distances: list) -> list:
    """
    Convert FAISS L2 distances to approximate cosine similarities.

    For L2-normalised (unit) vectors: cos(a, b) = 1 - ||a - b||^2 / 2.

    Args:
        l2_distances: List of L2 distance values returned by FAISS.

    Returns:
        List of cosine similarity values clipped to [0, 1].
    """
    return [max(0.0, 1.0 - (d ** 2) / 2.0) for d in l2_distances]


def format_confidence_note(score: float, label: ConfidenceLabel) -> str:
    """
    Compose a user-facing markdown note explaining the confidence level.

    Args:
        score: Confidence score in [0, 1].
        label: Categorical label ("HIGH", "MEDIUM", or "LOW").

    Returns:
        Markdown-formatted explanatory note.
    """
    pct = f"{score:.0%}"
    notes = {
        "HIGH": f"_Retrieval confidence: **{pct}** — Strong document match. Answer is well-supported by the knowledge base._",
        "MEDIUM": f"_Retrieval confidence: **{pct}** — Partial match found. Consider verifying with primary sources._",
        "LOW": f"_Retrieval confidence: **{pct}** — Limited context retrieved. Please consult authoritative financial documents directly._",
    }
    return notes.get(label, "")
