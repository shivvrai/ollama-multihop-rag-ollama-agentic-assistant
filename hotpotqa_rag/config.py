"""Configuration and constants for the Agentic RAG system."""

import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = PROJECT_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
CACHE_DIR = DATA_DIR / "cache"

DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Ollama Models
LANGUAGE_MODEL = os.environ.get("OLLAMA_LLM", "llama3.2")
EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBED", "nomic-embed-text")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Cross-Encoder Reranker
LOCAL_CROSS_ENCODER_PATH = PROJECT_ROOT / "models" / "ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_MODEL = (
    str(LOCAL_CROSS_ENCODER_PATH)
    if LOCAL_CROSS_ENCODER_PATH.exists()
    else "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Dataset Configuration
DATASET_NAME = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
DATASET_SPLIT = "validation"
EVAL_SUBSET_SIZE = 500


# Retrieval & Ranking Parameters
RETRIEVAL_TOP_K = 10       # Top candidates retrieved per sub-question
RERANK_TOP_K = 3           # Top candidates kept after cross-encoder reranking
RRF_K = 60                 # Reciprocal Rank Fusion smoothing parameter

# Agent Limits
MAX_SUB_QUESTIONS = 3
MAX_SUFFICIENCY_RETRIES = 2
MAX_TOTAL_ATTEMPTS = MAX_SUFFICIENCY_RETRIES + 1  # 1 initial + 2 retries = 3 total attempts

# Generation Settings
LLM_TEMPERATURE = 0.0
LLM_NUM_CTX = 4096
