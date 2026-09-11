"""Retrievers: Dense (cosine), Sparse (BM25), and Hybrid (Reciprocal Rank Fusion)."""

import logging
from typing import Optional, Literal

import numpy as np
import ollama

from hotpotqa_rag.config import RETRIEVAL_TOP_K, RRF_K, EMBEDDING_MODEL, OLLAMA_BASE_URL
from hotpotqa_rag.indexer import QuestionIndex, get_embedding, tokenize_for_bm25

logger = logging.getLogger("hotpotqa_rag")

RetrievalStrategy = Literal["dense", "bm25", "hybrid"]


def dense_retrieve(
    query: str,
    index: QuestionIndex,
    top_k: int = RETRIEVAL_TOP_K,
    client: Optional[ollama.Client] = None,
) -> list[tuple[int, float]]:
    """Retrieve top-k candidates using cosine similarity over nomic-embed-text embeddings."""
    q_vec = get_embedding(query, client=client, model=EMBEDDING_MODEL)
    # Cosine similarity: since index.dense_embeddings and q_vec are L2 normalized,
    # cosine similarity is the dot product.
    scores = np.dot(index.dense_embeddings, q_vec)
    ranked_indices = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in ranked_indices]


def bm25_retrieve(
    query: str,
    index: QuestionIndex,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[tuple[int, float]]:
    """Retrieve top-k candidates using rank_bm25 scores."""
    tokens = tokenize_for_bm25(query)
    if not tokens:
        # Fallback to returning in natural order with 0 score if query has no tokens
        return [(i, 0.0) for i in range(min(top_k, len(index.paragraphs)))]

    scores = np.array(index.bm25_index.get_scores(tokens), dtype=np.float32)
    ranked_indices = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in ranked_indices]


def hybrid_retrieve(
    query: str,
    index: QuestionIndex,
    top_k: int = RETRIEVAL_TOP_K,
    k_rrf: int = RRF_K,
    client: Optional[ollama.Client] = None,
) -> list[tuple[int, float]]:
    """Retrieve top-k candidates using Reciprocal Rank Fusion (RRF) of Dense + BM25 rankings.
    
    RRF Score: sum(1 / (k + rank_m(d))) for m in {dense, bm25}, where rank is 1-indexed.
    """
    n_docs = len(index.paragraphs)

    # 1. Full dense ranking
    q_vec = get_embedding(query, client=client, model=EMBEDDING_MODEL)
    dense_scores = np.dot(index.dense_embeddings, q_vec)
    dense_ranked = np.argsort(-dense_scores)
    dense_ranks = {doc_idx: rank + 1 for rank, doc_idx in enumerate(dense_ranked)}

    # 2. Full BM25 ranking
    tokens = tokenize_for_bm25(query)
    if tokens:
        bm25_scores = np.array(index.bm25_index.get_scores(tokens), dtype=np.float32)
        bm25_ranked = np.argsort(-bm25_scores)
        bm25_ranks = {doc_idx: rank + 1 for rank, doc_idx in enumerate(bm25_ranked)}
    else:
        bm25_ranks = {i: i + 1 for i in range(n_docs)}

    # 3. Combine with RRF
    rrf_scores: dict[int, float] = {}
    for doc_idx in range(n_docs):
        r_dense = dense_ranks[doc_idx]
        r_bm25 = bm25_ranks[doc_idx]
        score = (1.0 / (k_rrf + r_dense)) + (1.0 / (k_rrf + r_bm25))
        rrf_scores[doc_idx] = score

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return sorted_docs


def retrieve(
    query: str,
    index: QuestionIndex,
    strategy: RetrievalStrategy = "hybrid",
    top_k: int = RETRIEVAL_TOP_K,
    client: Optional[ollama.Client] = None,
) -> list[tuple[int, float]]:
    """Dispatch to the requested retrieval strategy."""
    if strategy == "bm25":
        return bm25_retrieve(query, index, top_k=top_k)
    elif strategy == "dense":
        return dense_retrieve(query, index, top_k=top_k, client=client)
    else:
        return hybrid_retrieve(query, index, top_k=top_k, client=client)
