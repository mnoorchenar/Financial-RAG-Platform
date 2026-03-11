"""
guardrails.py — Domain relevance guardrails for FinRAG.

Implements a two-stage filter to keep conversations on-topic for banking and
insurance. The guardrail is permissive for borderline/follow-up queries while
firmly rejecting clearly off-topic requests.

Stage 1 — Fast keyword scan:   O(n) string search, runs first.
Stage 2 — Off-topic pattern:   Regex match against known non-financial domains.
"""

import re
from typing import Tuple

from config import FINANCE_KEYWORDS, GUARDRAIL_MIN_KEYWORD_MATCH

# ── Off-topic domain patterns ──────────────────────────────────────────────────
_OFF_TOPIC_PATTERNS: list = [
    r"\b(recipe|bake|cooking|chef|cuisine|ingredient|meal prep)\b",
    r"\b(movie|film|series|actor|actress|oscar|netflix|streaming)\b",
    r"\b(soccer|basketball|baseball|nba|nfl|fifa|cricket score|esport)\b",
    r"\b(celebrity|gossip|makeup|skincare|fashion week|beauty)\b",
    r"\b(weather forecast|rainfall|hurricane track|tornado warning)\b",
    r"\b(alien|ufo|paranormal|conspiracy theory|flat earth)\b",
    r"\b(song lyrics|poem|novel|fiction|fan fiction)\b",
    r"\b(workout routine|yoga pose|diet plan|calorie count)\b",
]
_OFF_TOPIC_RE = re.compile("|".join(_OFF_TOPIC_PATTERNS), re.IGNORECASE)

_REJECTION_TEMPLATE = (
    "⚠️ **Out of Scope** — This platform is specialised in **banking and insurance** topics, "
    "including deposit insurance, mortgages, risk management, insurance policies, "
    "financial statements, and regulatory compliance (FDIC · NAIC · SEC · Basel III).\n\n"
    "Your question about *{topic}* falls outside this domain. Please rephrase or ask "
    "a finance-related question."
)


def _keyword_hit_count(query: str) -> int:
    """
    Count how many finance-domain keywords appear in the query.

    Args:
        query: Raw user query string.

    Returns:
        Integer count of matched finance keywords.
    """
    q_lower = query.lower()
    return sum(1 for kw in FINANCE_KEYWORDS if kw in q_lower)


def check_query(query: str) -> Tuple[bool, str]:
    """
    Evaluate whether a user query is within the banking / insurance domain.

    Args:
        query: The user's question.

    Returns:
        Tuple (is_allowed: bool, message: str).
        When is_allowed is False, message contains a user-facing explanation.
        When is_allowed is True, message is 'on_topic' or 'borderline'.
    """
    if not query or len(query.strip()) < 4:
        return False, "Please enter a valid question (minimum 4 characters)."

    text = query.strip()

    # Fast-pass: clear finance signal
    if _keyword_hit_count(text) >= GUARDRAIL_MIN_KEYWORD_MATCH:
        return True, "on_topic"

    # Reject clearly off-topic patterns
    match = _OFF_TOPIC_RE.search(text)
    if match:
        topic_word = match.group(0)
        return False, _REJECTION_TEMPLATE.format(topic=topic_word)

    # Borderline: allow through (may be a contextual follow-up)
    return True, "borderline"
