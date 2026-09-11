"""Cross-Encoder reranking using cross-encoder/ms-marco-MiniLM-L-6-v2."""

import logging
from typing import Optional

from hotpotqa_rag.config import CROSS_ENCODER_MODEL, RERANK_TOP_K
from hotpotqa_rag.indexer import QuestionIndex

logger = logging.getLogger("hotpotqa_rag")

_reranker_instance = None


def get_reranker(model_name: str = CROSS_ENCODER_MODEL):
    """Lazy loader / singleton for the SentenceTransformer CrossEncoder."""
    global _reranker_instance
    if _reranker_instance is None:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading CrossEncoder model: {model_name}...")
        _reranker_instance = CrossEncoder(model_name)
    return _reranker_instance


def rerank_candidates(
    query: str,
    candidate_indices: list[int],
    index: QuestionIndex,
    top_k: int = RERANK_TOP_K,
    model_name: str = CROSS_ENCODER_MODEL,
) -> list[tuple[int, float]]:
    """Rerank candidate passages using the cross-encoder and return the top_k.
    
    Returns:
        List of (paragraph_index, rerank_score) sorted by score descending.
    """
    if not candidate_indices:
        return []

    reranker = get_reranker(model_name)

    # Prepare (query, passage) pairs
    pairs = [
        (query, index.paragraphs[idx].formatted_text)
        for idx in candidate_indices
    ]

    scores = reranker.predict(pairs)

    # Pair each candidate index with its score
    scored = list(zip(candidate_indices, [float(s) for s in scores]))
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]
