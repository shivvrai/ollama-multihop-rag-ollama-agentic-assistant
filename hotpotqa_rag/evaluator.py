"""Evaluation metrics for HotpotQA: Exact Match (EM), F1, Retrieval Recall@3, and Latency."""

import collections
from dataclasses import dataclass
from typing import Sequence

from hotpotqa_rag.utils import normalize_answer


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute exact match score (0.0 or 1.0) after text normalization."""
    norm_pred = normalize_answer(prediction)
    norm_gold = normalize_answer(ground_truth)
    if not norm_gold:
        return 1.0 if not norm_pred else 0.0
    return 1.0 if norm_pred == norm_gold else 0.0


def compute_f1(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Compute token-level Precision, Recall, and F1 (Standard HotpotQA / SQuAD evaluation).
    
    Returns:
        (f1, precision, recall)
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not gold_tokens or not pred_tokens:
        score = 1.0 if pred_tokens == gold_tokens else 0.0
        return score, score, score

    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0, 0.0, 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def compute_recall_at_k(retrieved_titles: Sequence[str], gold_titles: Sequence[str], k: int = 3) -> float:
    """Compute paragraph retrieval recall@k: proportion of gold supporting paragraphs in top-k.
    
    Args:
        retrieved_titles: List of paragraph titles in ranked order.
        gold_titles: List of gold supporting paragraph titles.
        k: Cutoff rank (default 3).
        
    Returns:
        Recall score in [0.0, 1.0].
    """
    if not gold_titles:
        return 1.0

    top_k_titles = set(retrieved_titles[:k])
    gold_set = set(gold_titles)

    hits = len(top_k_titles.intersection(gold_set))
    return hits / len(gold_set)


@dataclass
class QuestionEvaluation:
    question_id: str
    exact_match: float
    f1: float
    precision: float
    recall: float
    recall_at_3: float
    latency: float


@dataclass
class AggregatedMetrics:
    config_name: str
    num_questions: int
    exact_match: float          # Percentage (0 - 100%)
    f1: float                   # Percentage (0 - 100%)
    recall_at_3: float          # Percentage (0 - 100%)
    avg_latency: float          # In seconds

    def to_markdown_row(self) -> str:
        return (
            f"| {self.config_name} | "
            f"{self.exact_match:.2f}% | "
            f"{self.f1:.2f}% | "
            f"{self.recall_at_3:.2f}% | "
            f"{self.avg_latency:.2f}s |"
        )
