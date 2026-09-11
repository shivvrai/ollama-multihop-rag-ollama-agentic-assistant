"""Rule-based adaptive router for sub-questions.

Routes to:
- "bm25" for named entities, dates, or numbers (precise keyword matching)
- "dense" for short or conceptual queries (semantic similarity)
- "hybrid" otherwise (reciprocal rank fusion)
"""

import re
from dataclasses import dataclass
from typing import Literal

from hotpotqa_rag.retriever import RetrievalStrategy


@dataclass
class RouteDecision:
    strategy: RetrievalStrategy
    reason: str
    features: list[str]


# Patterns for dates and numbers
DATE_PATTERN = re.compile(
    r"\b(18\d{2}|19\d{2}|20\d{2}|2100)\b|"
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
NUMBER_PATTERN = re.compile(r"\b\d+(?:st|nd|rd|th)?\b")

# Conceptual query starters
CONCEPTUAL_STARTERS = (
    "why",
    "how",
    "what is the meaning",
    "what is the role",
    "what is the concept",
    "what is the mechanism",
    "what is the purpose",
    "what is the function",
    "what is the definition",
    "what kind of",
    "what type of",
    "in what way",
    "what causes",
    "define",
    "describe",
    "explain",
)



def extract_named_entities(text: str) -> list[str]:
    """Find capitalized multi-word phrases, quoted phrases, or acronyms."""
    entities = []

    # Quoted text e.g. "The Great Gatsby"
    quotes = re.findall(r'["\']([^"\']+)["\']', text)
    entities.extend(quotes)

    # Acronyms (e.g. NASA, FBI, WWII)
    acronyms = re.findall(r"\b[A-Z]{2,}\b", text)
    entities.extend(acronyms)

    # Capitalized words not at the beginning of the sentence
    words = text.split()
    for i in range(1, len(words)):
        clean_word = re.sub(r"[^\w]", "", words[i])
        if clean_word and clean_word[0].isupper() and not clean_word.isupper():
            entities.append(clean_word)

    return entities


def route_query(query: str) -> RouteDecision:
    """Classify a sub-question into bm25, dense, or hybrid retrieval strategy."""
    q_stripped = query.strip()
    words = q_stripped.split()
    word_count = len(words)
    features = []

    # 1. Check for Dates, Numbers, Named Entities -> BM25
    dates = DATE_PATTERN.findall(q_stripped)
    numbers = NUMBER_PATTERN.findall(q_stripped)
    entities = extract_named_entities(q_stripped)

    has_date = bool(dates)
    has_number = bool(numbers)
    has_entity = bool(entities)

    if has_date:
        features.append(f"dates: {[d for tup in dates for d in tup if d]}")
    if has_number:
        features.append(f"numbers: {numbers[:3]}")
    if has_entity:
        features.append(f"entities: {entities[:3]}")

    if has_date or has_number or has_entity:
        return RouteDecision(
            strategy="bm25",
            reason="Detected named entities, dates, or numbers favoring exact keyword matching",
            features=features,
        )

    # 2. Check for Short / Conceptual -> Dense
    q_lower = q_stripped.lower()
    is_short = word_count <= 5
    is_conceptual = any(q_lower.startswith(starter) for starter in CONCEPTUAL_STARTERS)

    if is_short:
        features.append(f"short_query ({word_count} words)")
    if is_conceptual:
        features.append("conceptual_starter")

    if is_short or is_conceptual:
        return RouteDecision(
            strategy="dense",
            reason="Query is short or conceptual favoring semantic vector similarity",
            features=features,
        )

    # 3. Otherwise -> Hybrid (RRF)
    return RouteDecision(
        strategy="hybrid",
        reason="General multi-hop query benefiting from reciprocal rank fusion of dense and sparse signals",
        features=["general_query"],
    )
