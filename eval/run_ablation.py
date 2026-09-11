"""Evaluation Harness and Ablation Runner for Agentic RAG on HotpotQA."""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


from hotpotqa_rag.config import (
    EVAL_SUBSET_SIZE,
    RESULTS_DIR,
    OLLAMA_BASE_URL,
    LANGUAGE_MODEL,
)
from hotpotqa_rag.data_loader import load_hotpotqa_subset, HotpotQuestion
from hotpotqa_rag.indexer import build_question_index, QuestionIndex
from hotpotqa_rag.pipeline import AgenticRAGPipeline, PipelineTrace
from hotpotqa_rag.evaluator import (
    compute_exact_match,
    compute_f1,
    compute_recall_at_k,
    AggregatedMetrics,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("eval_harness")

CONFIG_NAMES = {
    "config_a": "(a) Naive RAG (Dense only)",
    "config_b": "(b) + Hybrid & Reranking",
    "config_c": "(c) + Query Decomposition",
    "config_d": "(d) + Sufficiency Grader & Retry",
}


def run_evaluation(
    num_questions: int = EVAL_SUBSET_SIZE,
    configs: list[str] = None,
    output_dir: Path = RESULTS_DIR,
    resume: bool = True,
):
    """Run full evaluation across the requested configurations."""
    if configs is None:
        configs = ["config_a", "config_b", "config_c", "config_d"]

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_file = output_dir / "eval_checkpoint.json"

    logger.info(f"Loading {num_questions} HotpotQA questions...")
    questions = load_hotpotqa_subset(subset_size=num_questions)
    logger.info(f"Loaded {len(questions)} questions for evaluation.")

    pipeline = AgenticRAGPipeline(model=LANGUAGE_MODEL)

    # Load existing checkpoint if available
    saved_data = {}
    if resume and checkpoint_file.exists():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
            logger.info(f"Resumed from existing checkpoint with {len(saved_data.get('results', {}))} configs.")
        except Exception as e:
            logger.warning(f"Failed to read checkpoint: {e}")

    results = saved_data.get("results", {})
    reliability_stats = saved_data.get(
        "reliability",
        {
            "total_decomp_calls": 0,
            "malformed_decomp_calls": 0,
            "total_grader_calls": 0,
            "malformed_grader_calls": 0,
            "questions_at_retry_cap": 0,
            "total_config_d_questions": 0,
        },
    )

    # Pre-build indexes for all questions
    logger.info("Indexing all question candidate paragraphs...")
    indexes: dict[str, QuestionIndex] = {}
    for q in tqdm(questions, desc="Building Indexes"):
        indexes[q.id] = build_question_index(q, client=pipeline.client)

    # Run each configuration
    for cfg in configs:
        display_name = CONFIG_NAMES.get(cfg, cfg)
        logger.info(f"\n==========================================")
        logger.info(f"Running Configuration: {display_name}")
        logger.info(f"==========================================")

        if cfg not in results:
            results[cfg] = []

        already_done_ids = {r["question_id"] for r in results[cfg]}
        pending_questions = [q for q in questions if q.id not in already_done_ids]

        pbar = tqdm(pending_questions, desc=f"Eval {cfg}")
        for q in pbar:
            q_index = indexes[q.id]
            trace: PipelineTrace = pipeline.run(q.question, q_index, config=cfg)

            # Compute metrics
            em = compute_exact_match(trace.synthesis.answer, q.answer)
            f1, prec, rec = compute_f1(trace.synthesis.answer, q.answer)
            # In config a/b, top 3 are kept. In config c/d, top 3 are kept per sub-question.
            k_cutoff = 3 if cfg in ("config_a", "config_b") else len(trace.final_evidence_titles)
            rec3 = compute_recall_at_k(trace.final_evidence_titles, q.gold_titles, k=k_cutoff)


            rec_dict = {
                "question_id": q.id,
                "question": q.question,
                "gold_answer": q.answer,
                "gold_titles": q.gold_titles,
                "pred_answer": trace.synthesis.answer,
                "retrieved_titles": trace.final_evidence_titles,
                "exact_match": em,
                "f1": f1,
                "precision": prec,
                "recall": rec,
                "recall_at_3": rec3,
                "latency": trace.latency_seconds,
                "trace_details": {
                    "decomposition": {
                        "needs_decomposition": trace.decomposition.needs_decomposition if trace.decomposition else None,
                        "sub_questions": trace.decomposition.sub_questions if trace.decomposition else [],
                        "fallback_fired": trace.decomposition.fallback_fired if trace.decomposition else False,
                        "raw_output": trace.decomposition.raw_output if trace.decomposition else "",
                    } if trace.decomposition else None,
                    "sub_traces": [
                        {
                            "sub_question": st.sub_question,
                            "route_strategy": st.route_decision.strategy,
                            "route_reason": st.route_decision.reason,
                            "route_features": st.route_decision.features,
                            "retrieved_candidates": st.retrieved_candidates,
                            "reranked_candidates": st.reranked_candidates,
                            "reformulated_queries": st.reformulated_queries,
                            "retry_count": st.retry_count,
                            "grader_attempts": [
                                {
                                    "attempt": ga.attempt,
                                    "sufficient": ga.sufficient,
                                    "reason": ga.reason,
                                    "fallback_fired": ga.fallback_fired,
                                    "is_malformed": ga.is_malformed,
                                    "raw_output": ga.raw_output,
                                }
                                for ga in st.grader_attempts
                            ],
                        }
                        for st in trace.sub_traces
                    ],
                    "synthesis": {
                        "answer": trace.synthesis.answer,
                        "cited_passages": trace.synthesis.cited_passages,
                        "raw_output": trace.synthesis.raw_output,
                    },
                },
            }
            results[cfg].append(rec_dict)


            # Track reliability if config_d
            if cfg == "config_d":
                reliability_stats["total_config_d_questions"] += 1
                if trace.decomposition:
                    reliability_stats["total_decomp_calls"] += 1
                    if trace.decomp_malformed:
                        reliability_stats["malformed_decomp_calls"] += 1

                for st in trace.sub_traces:
                    for ga in st.grader_attempts:
                        reliability_stats["total_grader_calls"] += 1
                        if ga.is_malformed:
                            reliability_stats["malformed_grader_calls"] += 1

                if trace.hit_retry_cap_insufficient:
                    reliability_stats["questions_at_retry_cap"] += 1

            # Update progress bar
            avg_f1_so_far = sum(r["f1"] for r in results[cfg]) / len(results[cfg]) * 100
            pbar.set_postfix({"F1": f"{avg_f1_so_far:.1f}%", "Lat": f"{trace.latency_seconds:.1f}s"})

            # Periodic checkpoint saving
            if len(results[cfg]) % 10 == 0:
                with open(checkpoint_file, "w", encoding="utf-8") as f:
                    json.dump({"results": results, "reliability": reliability_stats}, f, indent=2)

        # Save checkpoint after each config
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump({"results": results, "reliability": reliability_stats}, f, indent=2)

    # Compute Aggregates
    aggregated: list[AggregatedMetrics] = []
    for cfg in configs:
        cfg_results = results[cfg]
        n = len(cfg_results)
        if n == 0:
            continue
        em_avg = sum(r["exact_match"] for r in cfg_results) / n * 100
        f1_avg = sum(r["f1"] for r in cfg_results) / n * 100
        rec3_avg = sum(r["recall_at_3"] for r in cfg_results) / n * 100
        lat_avg = sum(r["latency"] for r in cfg_results) / n

        aggregated.append(
            AggregatedMetrics(
                config_name=CONFIG_NAMES.get(cfg, cfg),
                num_questions=n,
                exact_match=em_avg,
                f1=f1_avg,
                recall_at_3=rec3_avg,
                avg_latency=lat_avg,
            )
        )

    # 1. Output Markdown Table
    md_path = output_dir / "ablation_results.md"
    generate_markdown_report(aggregated, reliability_stats, md_path)
    logger.info(f"Saved ablation markdown report to {md_path}")

    # 2. Output Bar Chart of F1
    chart_path = output_dir / "f1_ablation.png"
    generate_f1_chart(aggregated, chart_path)
    logger.info(f"Saved F1 ablation bar chart to {chart_path}")

    # Print summary to console
    print("\n" + "=" * 80)
    print("ACCURACY ABLATION RESULTS (HotpotQA Distractor)")
    print("=" * 80)
    print(f"{'Configuration':<38} | {'EM (%)':<8} | {'F1 (%)':<8} | {'Recall@3 (%)':<14} | {'Latency (s)':<10}")
    print("-" * 86)
    for m in aggregated:
        print(f"{m.config_name:<38} | {m.exact_match:>7.2f}% | {m.f1:>7.2f}% | {m.recall_at_3:>13.2f}% | {m.avg_latency:>9.2f}s")
    print("=" * 80)

    print("\n" + "=" * 80)
    print("SMALL-MODEL SELF-CRITIQUE RELIABILITY TRACKING")
    print("=" * 80)
    print_reliability_table(reliability_stats)

    return aggregated, reliability_stats


def generate_markdown_report(
    metrics: list[AggregatedMetrics],
    reliability: dict,
    output_path: Path,
):
    """Generate Markdown report containing ablation table and reliability tracking."""
    lines = [
        "# Agentic RAG Multi-Hop QA: Ablation & Reliability Results",
        "",
        "Evaluation performed on the HotpotQA distractor validation split using local `llama3.2` and `nomic-embed-text`.",
        "",
        "## 1. Accuracy and Retrieval Ablation",
        "",
        "| Configuration | Exact Match | F1 Score | Retrieval Recall@3 | Avg Latency |",
        "|---|---|---|---|---|",
    ]
    for m in metrics:
        lines.append(m.to_markdown_row())

    # Reliability tracking
    decomp_total = reliability.get("total_decomp_calls", 0)
    decomp_malformed = reliability.get("malformed_decomp_calls", 0)
    decomp_pct = (decomp_malformed / decomp_total * 100) if decomp_total > 0 else 0.0

    grader_total = reliability.get("total_grader_calls", 0)
    grader_malformed = reliability.get("malformed_grader_calls", 0)
    grader_pct = (grader_malformed / grader_total * 100) if grader_total > 0 else 0.0

    q_total = reliability.get("total_config_d_questions", 0)
    cap_hit = reliability.get("questions_at_retry_cap", 0)
    cap_pct = (cap_hit / q_total * 100) if q_total > 0 else 0.0

    lines.extend([
        "",
        "## 2. Small-Model Self-Critique Reliability Tracking",
        "",
        "Tracks structured output adherence, validation failures, and retry exhaustion for `llama3.2`.",
        "",
        "| Reliability Metric | Count / Total | Failure / Fallback Rate |",
        "|---|---|---|",
        f"| Decomposition calls returning malformed JSON (fallback triggered) | {decomp_malformed} / {decomp_total} | **{decomp_pct:.2f}%** |",
        f"| Sufficiency Grader calls returning malformed JSON | {grader_malformed} / {grader_total} | **{grader_pct:.2f}%** |",
        f"| Questions hitting 2-retry cap without confirmed sufficiency | {cap_hit} / {q_total} | **{cap_pct:.2f}%** |",
        "",
        "![F1 Ablation Chart](f1_ablation.png)",
        "",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def print_reliability_table(reliability: dict):
    decomp_total = reliability.get("total_decomp_calls", 0)
    decomp_malformed = reliability.get("malformed_decomp_calls", 0)
    decomp_pct = (decomp_malformed / decomp_total * 100) if decomp_total > 0 else 0.0

    grader_total = reliability.get("total_grader_calls", 0)
    grader_malformed = reliability.get("malformed_grader_calls", 0)
    grader_pct = (grader_malformed / grader_total * 100) if grader_total > 0 else 0.0

    q_total = reliability.get("total_config_d_questions", 0)
    cap_hit = reliability.get("questions_at_retry_cap", 0)
    cap_pct = (cap_hit / q_total * 100) if q_total > 0 else 0.0

    print(f"{'Metric':<55} | {'Count / Total':<15} | {'Failure Rate':<12}")
    print("-" * 86)
    print(f"{'Decomposition malformed JSON (fallback used)':<55} | {f'{decomp_malformed}/{decomp_total}':<15} | {decomp_pct:>10.2f}%")
    print(f"{'Sufficiency Grader malformed JSON':<55} | {f'{grader_malformed}/{grader_total}':<15} | {grader_pct:>10.2f}%")
    print(f"{'Questions hitting 2-retry cap without sufficiency':<55} | {f'{cap_hit}/{q_total}':<15} | {cap_pct:>10.2f}%")
    print("=" * 80 + "\n")


def generate_f1_chart(metrics: list[AggregatedMetrics], output_path: Path):
    """Generate a high-resolution bar chart of F1 score across configurations."""
    if not metrics:
        return

    configs = [m.config_name for m in metrics]
    f1_scores = [m.f1 for m in metrics]
    em_scores = [m.exact_match for m in metrics]

    colors = ["#4A90E2", "#50E3C2", "#F5A623", "#9013FE"]
    if len(configs) <= len(colors):
        bar_colors = colors[:len(configs)]
    else:
        bar_colors = colors * (len(configs) // len(colors) + 1)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    bars = ax.bar(configs, f1_scores, color=bar_colors, width=0.55, edgecolor="#2C3E50", linewidth=1.2)

    ax.set_ylabel("F1 Score (%)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_title("HotpotQA Multi-Hop QA: F1 Score by RAG Configuration", fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, max(max(f1_scores, default=50) * 1.25, 60))
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Format x-ticks
    plt.xticks(rotation=15, ha="right", fontsize=10)

    # Annotate value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:.2f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HotpotQA Agentic RAG Ablation")
    parser.add_argument("--num-questions", type=int, default=EVAL_SUBSET_SIZE, help="Number of questions to evaluate")
    parser.add_argument("--configs", nargs="+", default=["config_a", "config_b", "config_c", "config_d"])
    parser.add_argument("--no-resume", action="store_true", help="Start evaluation from scratch")
    args = parser.parse_args()

    run_evaluation(
        num_questions=args.num_questions,
        configs=args.configs,
        resume=not args.no_resume,
    )
