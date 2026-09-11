"""Agentic RAG Pipeline: Orchestrates multi-hop QA across different configurations."""

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Literal

import ollama

from hotpotqa_rag.config import (
    LANGUAGE_MODEL,
    OLLAMA_BASE_URL,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    MAX_SUFFICIENCY_RETRIES,
    MAX_TOTAL_ATTEMPTS,
)
from hotpotqa_rag.indexer import QuestionIndex
from hotpotqa_rag.retriever import retrieve, dense_retrieve, hybrid_retrieve
from hotpotqa_rag.router import route_query, RouteDecision
from hotpotqa_rag.reranker import rerank_candidates
from hotpotqa_rag.decomposer import (
    decompose_question,
    reformulate_sub_question,
    DecompositionResult,
)
from hotpotqa_rag.grader import grade_evidence_sufficiency, GraderResult
from hotpotqa_rag.synthesizer import synthesize_final_answer, SynthesisResult

logger = logging.getLogger("hotpotqa_rag")

PipelineConfig = Literal["config_a", "config_b", "config_c", "config_d"]


@dataclass
class SubQuestionTrace:
    sub_question: str
    route_decision: RouteDecision
    retrieved_candidates: list[tuple[int, float]]       # (paragraph_idx, score)
    reranked_candidates: list[tuple[int, float]]        # (paragraph_idx, score)
    grader_attempts: list[GraderResult] = field(default_factory=list)
    retry_count: int = 0
    final_evidence_indices: list[int] = field(default_factory=list)
    reformulated_queries: list[str] = field(default_factory=list)



@dataclass
class PipelineTrace:
    question: str
    config_name: str
    decomposition: Optional[DecompositionResult]
    sub_traces: list[SubQuestionTrace]
    final_evidence_indices: list[int]
    final_evidence_titles: list[str]
    synthesis: SynthesisResult
    latency_seconds: float
    decomp_malformed: bool = False
    grader_malformed_count: int = 0
    hit_retry_cap_insufficient: bool = False


class AgenticRAGPipeline:
    """Multi-hop Agentic RAG system supporting naive to full agentic configurations."""

    def __init__(
        self,
        client: Optional[ollama.Client] = None,
        model: str = LANGUAGE_MODEL,
    ):
        self.client = client or ollama.Client(host=OLLAMA_BASE_URL)
        self.model = model

    def run(
        self,
        question: str,
        index: QuestionIndex,
        config: PipelineConfig = "config_d",
    ) -> PipelineTrace:
        """Run question through the pipeline using the specified configuration.
        
        Configs:
          - "config_a": Naive RAG (dense only, no reranking, no decomposition, single pass top-3)
          - "config_b": Hybrid retrieval (top-10) + Cross-Encoder reranking (top-3), no decomposition
          - "config_c": Decomposition + Adaptive Routing + Retrieval (top-10) + Reranking (top-3)
          - "config_d": Full Agentic (+ Sufficiency Grader with retry loop bounded to 2)
        """
        start_time = time.perf_counter()

        if config == "config_a":
            trace = self._run_config_a(question, index)
        elif config == "config_b":
            trace = self._run_config_b(question, index)
        elif config == "config_c":
            trace = self._run_config_c(question, index)
        else:
            trace = self._run_config_d(question, index)

        trace.latency_seconds = time.perf_counter() - start_time
        return trace

    def _run_config_a(self, question: str, index: QuestionIndex) -> PipelineTrace:
        """Config (a): Naive RAG - dense retrieval only, no rerank, no decomposition, single pass."""
        # Retrieve top 3 dense directly
        dense_results = dense_retrieve(question, index, top_k=RERANK_TOP_K, client=self.client)
        evidence_indices = [idx for idx, _ in dense_results]
        evidence_texts = [index.paragraphs[idx].formatted_text for idx in evidence_indices]

        # Synthesize answer
        synthesis = synthesize_final_answer(
            question, evidence_texts, client=self.client, model=self.model
        )

        sub_trace = SubQuestionTrace(
            sub_question=question,
            route_decision=RouteDecision("dense", "Config (a) forced dense retrieval", []),
            retrieved_candidates=dense_results,
            reranked_candidates=dense_results,
            final_evidence_indices=evidence_indices,
        )

        return PipelineTrace(
            question=question,
            config_name="config_a (Naive RAG)",
            decomposition=None,
            sub_traces=[sub_trace],
            final_evidence_indices=evidence_indices,
            final_evidence_titles=index.get_titles(evidence_indices),
            synthesis=synthesis,
            latency_seconds=0.0,
        )

    def _run_config_b(self, question: str, index: QuestionIndex) -> PipelineTrace:
        """Config (b): + Hybrid retrieval and reranking, no decomposition."""
        # 1. Hybrid retrieve top 10
        retrieved = hybrid_retrieve(question, index, top_k=RETRIEVAL_TOP_K, client=self.client)
        candidate_indices = [idx for idx, _ in retrieved]

        # 2. Cross-encoder rerank to top 3
        reranked = rerank_candidates(question, candidate_indices, index, top_k=RERANK_TOP_K)
        evidence_indices = [idx for idx, _ in reranked]
        evidence_texts = [index.paragraphs[idx].formatted_text for idx in evidence_indices]

        # 3. Synthesize
        synthesis = synthesize_final_answer(
            question, evidence_texts, client=self.client, model=self.model
        )

        sub_trace = SubQuestionTrace(
            sub_question=question,
            route_decision=RouteDecision("hybrid", "Config (b) forced hybrid RRF", []),
            retrieved_candidates=retrieved,
            reranked_candidates=reranked,
            final_evidence_indices=evidence_indices,
        )

        return PipelineTrace(
            question=question,
            config_name="config_b (+ Hybrid & Rerank)",
            decomposition=None,
            sub_traces=[sub_trace],
            final_evidence_indices=evidence_indices,
            final_evidence_titles=index.get_titles(evidence_indices),
            synthesis=synthesis,
            latency_seconds=0.0,
        )

    def _run_config_c(self, question: str, index: QuestionIndex) -> PipelineTrace:
        """Config (c): + Query decomposition with adaptive router and reranking."""
        # 1. Decompose query
        decomp = decompose_question(question, client=self.client, model=self.model)

        sub_traces: list[SubQuestionTrace] = []
        collected_indices: list[int] = []

        for sq in decomp.sub_questions:
            # 2. Adaptive router
            route_dec = route_query(sq)

            # 3. Retrieve top 10
            retrieved = retrieve(sq, index, strategy=route_dec.strategy, top_k=RETRIEVAL_TOP_K, client=self.client)
            candidate_indices = [idx for idx, _ in retrieved]

            # 4. Rerank to top 3
            reranked = rerank_candidates(sq, candidate_indices, index, top_k=RERANK_TOP_K)
            evidence_indices = [idx for idx, _ in reranked]

            for idx in evidence_indices:
                if idx not in collected_indices:
                    collected_indices.append(idx)

            sub_traces.append(
                SubQuestionTrace(
                    sub_question=sq,
                    route_decision=route_dec,
                    retrieved_candidates=retrieved,
                    reranked_candidates=reranked,
                    final_evidence_indices=evidence_indices,
                )
            )

        # 5. Synthesize final answer from all gathered evidence
        evidence_texts = [index.paragraphs[idx].formatted_text for idx in collected_indices]
        synthesis = synthesize_final_answer(
            question,
            evidence_texts,
            sub_questions=decomp.sub_questions,
            client=self.client,
            model=self.model,
        )

        return PipelineTrace(
            question=question,
            config_name="config_c (+ Query Decomposition)",
            decomposition=decomp,
            sub_traces=sub_traces,
            final_evidence_indices=collected_indices,
            final_evidence_titles=index.get_titles(collected_indices),
            synthesis=synthesis,
            latency_seconds=0.0,
            decomp_malformed=decomp.is_malformed,
        )

    def _run_config_d(self, question: str, index: QuestionIndex) -> PipelineTrace:
        """Config (d): Full Agentic RAG (+ Sufficiency Grader and retry loop)."""
        # 1. Decompose query
        decomp = decompose_question(question, client=self.client, model=self.model)

        sub_traces: list[SubQuestionTrace] = []
        collected_indices: list[int] = []
        total_grader_malformed = 0
        hit_retry_cap_insufficient = False

        for sq in decomp.sub_questions:
            # Adaptive router for initial strategy
            route_dec = route_query(sq)
            current_strategy = route_dec.strategy
            current_query = sq
            reformulated_queries: list[str] = []

            grader_attempts: list[GraderResult] = []
            final_reranked: list[tuple[int, float]] = []
            final_sub_indices: list[int] = []

            # Retry loop: attempt 0 is initial, attempts 1..MAX_SUFFICIENCY_RETRIES are retries
            retry_count = 0
            is_sufficient = False
            last_retrieved: list[tuple[int, float]] = []

            while retry_count <= MAX_SUFFICIENCY_RETRIES and not is_sufficient:
                attempt_num = retry_count + 1

                # On retries, actively reformulate the query based on the grader's stated reason!
                if retry_count > 0 and grader_attempts:
                    prev_reason = grader_attempts[-1].reason
                    current_query = reformulate_sub_question(
                        sub_question=sq,
                        insufficiency_reason=prev_reason,
                        client=self.client,
                        model=self.model,
                    )
                    reformulated_queries.append(current_query)

                    # Re-route the newly reformulated query
                    route_dec = route_query(current_query)
                    current_strategy = route_dec.strategy

                    # If strategy was bm25 or dense, broaden to hybrid on retry 2
                    if retry_count == 2 and current_strategy != "hybrid":
                        current_strategy = "hybrid"

                # Retrieve top 10 using current_query
                last_retrieved = retrieve(
                    current_query, index, strategy=current_strategy, top_k=RETRIEVAL_TOP_K, client=self.client
                )
                candidate_indices = [idx for idx, _ in last_retrieved]

                # Rerank to top 3 using current_query
                final_reranked = rerank_candidates(current_query, candidate_indices, index, top_k=RERANK_TOP_K)
                final_sub_indices = [idx for idx, _ in final_reranked]
                evidence_texts = [index.paragraphs[idx].formatted_text for idx in final_sub_indices]

                # Grade sufficiency against original sub-question
                grade_res = grade_evidence_sufficiency(
                    question=sq,
                    evidence_passages=evidence_texts,
                    attempt=attempt_num,
                    client=self.client,
                    model=self.model,
                    max_attempts=MAX_TOTAL_ATTEMPTS,
                )
                grader_attempts.append(grade_res)

                if grade_res.is_malformed:
                    total_grader_malformed += 1

                if grade_res.sufficient:
                    is_sufficient = True
                    break
                else:
                    retry_count += 1

            # Check if retry cap was reached without confirmation
            if not is_sufficient and retry_count > MAX_SUFFICIENCY_RETRIES:
                hit_retry_cap_insufficient = True

            for idx in final_sub_indices:
                if idx not in collected_indices:
                    collected_indices.append(idx)

            sub_traces.append(
                SubQuestionTrace(
                    sub_question=sq,
                    route_decision=route_dec,
                    retrieved_candidates=last_retrieved,
                    reranked_candidates=final_reranked,
                    grader_attempts=grader_attempts,
                    retry_count=min(retry_count, MAX_SUFFICIENCY_RETRIES),
                    final_evidence_indices=final_sub_indices,
                    reformulated_queries=reformulated_queries,
                )
            )

        # Synthesize final answer
        evidence_texts = [index.paragraphs[idx].formatted_text for idx in collected_indices]
        synthesis = synthesize_final_answer(
            question,
            evidence_texts,
            sub_questions=decomp.sub_questions,
            client=self.client,
            model=self.model,
        )

        return PipelineTrace(
            question=question,
            config_name="config_d (Full Agentic RAG)",
            decomposition=decomp,
            sub_traces=sub_traces,
            final_evidence_indices=collected_indices,
            final_evidence_titles=index.get_titles(collected_indices),
            synthesis=synthesis,
            latency_seconds=0.0,
            decomp_malformed=decomp.is_malformed,
            grader_malformed_count=total_grader_malformed,
            hit_retry_cap_insufficient=hit_retry_cap_insufficient,
        )
