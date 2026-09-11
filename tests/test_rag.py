"""Unit tests for Agentic RAG components."""

import pytest
import numpy as np
from rank_bm25 import BM25Okapi

from hotpotqa_rag.utils import extract_json_from_text, normalize_answer
from hotpotqa_rag.evaluator import compute_exact_match, compute_f1, compute_recall_at_k
from hotpotqa_rag.router import route_query
from hotpotqa_rag.retriever import hybrid_retrieve, tokenize_for_bm25
from hotpotqa_rag.indexer import QuestionIndex
from hotpotqa_rag.data_loader import CandidateParagraph
from hotpotqa_rag.grader import grade_evidence_sufficiency, GraderResult
from hotpotqa_rag.decomposer import decompose_question, DecompositionResult


def test_extract_json_from_text_clean():
    text = '{"needs_decomposition": true, "sub_questions": ["q1", "q2"]}'
    res = extract_json_from_text(text)
    assert res == {"needs_decomposition": True, "sub_questions": ["q1", "q2"]}


def test_extract_json_from_text_markdown():
    text = """Here is the result:
```json
{
  "sufficient": false,
  "reason": "Missing founding date"
}
```
Hope that helps!"""
    res = extract_json_from_text(text)
    assert res == {"sufficient": False, "reason": "Missing founding date"}


def test_extract_json_from_text_trailing_comma():
    text = '{"needs_decomposition": false, "sub_questions": ["q1",],}'
    res = extract_json_from_text(text)
    assert res is not None
    assert res["needs_decomposition"] is False


def test_extract_json_from_text_invalid():
    text = "Sorry, I cannot answer this as JSON."
    res = extract_json_from_text(text)
    assert res is None


def test_extract_json_from_text_unquoted_string():
    text = '{"sufficient": true, "reason": The evidence contains sufficient information.}'
    res = extract_json_from_text(text)
    assert res is not None
    assert res["sufficient"] is True
    assert "sufficient information" in res["reason"]


def test_normalize_answer():
    assert normalize_answer("The United States of America!") == "united states of america"
    assert normalize_answer("  An apple, a pear; ") == "apple pear"
    assert normalize_answer("1994") == "1994"
    assert normalize_answer("yes [Passage 1]") == "yes"
    assert normalize_answer("Chief of Protocol [Passage 2][Passage 3]") == "chief of protocol"


def test_exact_match_and_f1():
    assert compute_exact_match("Arthur's Magazine", "Arthur's Magazine") == 1.0
    assert compute_exact_match("The Beatles", "beatles") == 1.0
    assert compute_exact_match("New York", "London") == 0.0

    f1, prec, rec = compute_f1("The quick brown fox", "quick brown fox")
    assert f1 == 1.0  # "The" is stripped by normalize_answer

    f1_partial, _, _ = compute_f1("George Washington university", "George Washington")
    assert 0.0 < f1_partial < 1.0


def test_recall_at_k():
    gold = ["Title A", "Title B"]
    retrieved = ["Title A", "Title C", "Title B", "Title D"]
    assert compute_recall_at_k(retrieved, gold, k=3) == 1.0
    assert compute_recall_at_k(retrieved, gold, k=1) == 0.5
    assert compute_recall_at_k(["Title X", "Title Y"], gold, k=3) == 0.0


def test_router_named_entities_and_dates():
    # Date test -> BM25
    d1 = route_query("In which year was the Eiffel Tower built in 1889?")
    assert d1.strategy == "bm25"

    # Number test -> BM25
    d2 = route_query("Who won the 42nd Academy Awards?")
    assert d2.strategy == "bm25"

    # Named entity test -> BM25
    d3 = route_query("What team did Babe Ruth play for?")
    assert d3.strategy == "bm25"


def test_router_short_and_conceptual():
    # Short query -> Dense
    d1 = route_query("magnetic resonance")
    assert d1.strategy == "dense"

    # Conceptual starter -> Dense
    d2 = route_query("what is the mechanism of action for penicillin")
    assert d2.strategy == "dense"


def test_router_general_hybrid():
    d = route_query("which company produced both vehicles and aerospace engines")
    assert d.strategy == "hybrid"


def test_hybrid_rrf_ranking():
    paragraphs = [
        CandidateParagraph(index=0, title="Doc 0", sentences=["Solar energy panels convert sunlight into electricity."]),
        CandidateParagraph(index=1, title="Doc 1", sentences=["Wind turbines generate renewable kinetic energy."]),
        CandidateParagraph(index=2, title="Doc 2", sentences=["Coal and fossil fuels produce carbon emissions."]),
    ]
    formatted = [p.formatted_text for p in paragraphs]
    tokenized = [tokenize_for_bm25(f) for f in formatted]
    bm25 = BM25Okapi(tokenized)

    # Dummy normalized embeddings
    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    index = QuestionIndex(
        question_id="test_q",
        paragraphs=paragraphs,
        dense_embeddings=embeddings,
        bm25_index=bm25,
        tokenized_corpus=tokenized,
    )

    # BM25 retrieve "solar energy"
    results = index.bm25_index.get_scores(["solar", "energy"])
    assert results[0] > results[1]
