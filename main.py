"""CLI Entrypoint for Local Agentic RAG for Multi-Hop QA."""

import sys
import argparse
from hotpotqa_rag.config import LANGUAGE_MODEL
from hotpotqa_rag.data_loader import load_hotpotqa_subset
from hotpotqa_rag.indexer import build_question_index
from hotpotqa_rag.pipeline import AgenticRAGPipeline


def main():
    parser = argparse.ArgumentParser(description="Local Agentic Multi-Hop RAG Assistant")
    parser.add_argument(
        "query",
        nargs="?",
        default="Were Scott Derrickson and Ed Wood of the same nationality?",
        help="Question to ask",
    )
    parser.add_argument(
        "--config",
        default="config_b",
        choices=["config_a", "config_b", "config_c", "config_d"],
        help="Pipeline configuration to execute",
    )
    args = parser.parse_args()

    print(f"\n🚀 Running Agentic RAG ({args.config}) on: '{args.query}'")
    print("Loading benchmark dataset & indexing candidate paragraphs...")
    questions = load_hotpotqa_subset(subset_size=5)
    pipeline = AgenticRAGPipeline(model=LANGUAGE_MODEL)
    index = build_question_index(questions[0], client=pipeline.client)

    trace = pipeline.run(args.query, index, config=args.config)
    print("\n" + "=" * 60)
    print(f"ANSWER:    {trace.synthesis.answer}")
    print(f"CITATIONS: {trace.synthesis.cited_passages}")
    print(f"LATENCY:   {trace.latency_seconds:.2f}s")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
