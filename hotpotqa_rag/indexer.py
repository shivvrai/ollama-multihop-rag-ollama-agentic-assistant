"""Dual indexing for HotpotQA: Dense vector index + Sparse BM25 index."""

import re
import string
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi
import ollama

from hotpotqa_rag.config import EMBEDDING_MODEL, OLLAMA_BASE_URL
from hotpotqa_rag.data_loader import HotpotQuestion, CandidateParagraph

logger = logging.getLogger("hotpotqa_rag")


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens for BM25 indexing."""
    cleaned = text.lower().translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    tokens = [tok for tok in cleaned.split() if tok]
    return tokens


@dataclass
class QuestionIndex:
    """Parallel dense and sparse indexes for the 10 candidate paragraphs of a question."""
    question_id: str
    paragraphs: list[CandidateParagraph]
    dense_embeddings: np.ndarray        # Shape: (10, D), normalized to unit length
    bm25_index: BM25Okapi
    tokenized_corpus: list[list[str]]

    def get_paragraph(self, index: int) -> CandidateParagraph:
        return self.paragraphs[index]

    def get_title(self, index: int) -> str:
        return self.paragraphs[index].title

    def get_titles(self, indices: list[int]) -> list[str]:
        return [self.paragraphs[i].title for i in indices]


def get_embedding(text: str, client: Optional[ollama.Client] = None, model: str = EMBEDDING_MODEL) -> np.ndarray:
    """Compute dense embedding for a single text using nomic-embed-text via Ollama."""
    if client is None:
        client = ollama.Client(host=OLLAMA_BASE_URL)
    response = client.embed(model=model, input=text)
    vec = np.array(response["embeddings"][0], dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def get_embeddings_batch(
    texts: list[str], client: Optional[ollama.Client] = None, model: str = EMBEDDING_MODEL
) -> np.ndarray:
    """Compute dense embeddings for a batch of texts using nomic-embed-text via Ollama."""
    if client is None:
        client = ollama.Client(host=OLLAMA_BASE_URL)
    response = client.embed(model=model, input=texts)
    vecs = np.array(response["embeddings"], dtype=np.float32)
    # L2 normalize each vector row
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def build_question_index(
    question: HotpotQuestion,
    client: Optional[ollama.Client] = None,
    dense_embeddings: Optional[np.ndarray] = None,
) -> QuestionIndex:
    """Build parallel dense (nomic-embed-text) and sparse (BM25) indexes for a question's 10 paragraphs."""
    paragraphs = question.paragraphs
    formatted_texts = [p.formatted_text for p in paragraphs]

    # 1. Sparse BM25 Index
    tokenized_corpus = [tokenize_for_bm25(text) for text in formatted_texts]
    bm25_index = BM25Okapi(tokenized_corpus)

    # 2. Dense Vector Index
    if dense_embeddings is None:
        dense_embeddings = get_embeddings_batch(formatted_texts, client=client)

    return QuestionIndex(
        question_id=question.id,
        paragraphs=paragraphs,
        dense_embeddings=dense_embeddings,
        bm25_index=bm25_index,
        tokenized_corpus=tokenized_corpus,
    )
