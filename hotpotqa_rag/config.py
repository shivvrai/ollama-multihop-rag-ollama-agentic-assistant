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

def get_ollama_base_url() -> str:
    """Retrieve Ollama host from env vars, Streamlit secrets, or local default."""
    if "OLLAMA_HOST" in os.environ and os.environ["OLLAMA_HOST"].strip():
        return os.environ["OLLAMA_HOST"].strip().rstrip("/")
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "OLLAMA_HOST" in st.secrets:
            return str(st.secrets["OLLAMA_HOST"]).strip().rstrip("/")
    except Exception:
        pass
    return "http://127.0.0.1:11434"


def set_ollama_base_url(url: str):
    """Dynamically update the active Ollama host at runtime."""
    global OLLAMA_BASE_URL
    clean = url.strip().rstrip("/")
    if clean:
        OLLAMA_BASE_URL = clean
        os.environ["OLLAMA_HOST"] = clean


def check_ollama_connection(host: str = None, timeout: float = 3.0) -> tuple[bool, str, list[str]]:
    """Fast healthcheck for Ollama host.
    
    Returns:
        (is_connected, message, available_models)
    """
    import urllib.request
    import json
    target = host or get_ollama_base_url()
    try:
        url = f"{target.rstrip('/')}/api/tags"
        req = urllib.request.Request(url, headers={"User-Agent": "AgenticRAG-HealthCheck"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
                return True, "Connected", models
            return False, f"HTTP {resp.status}", []
    except Exception as e:
        return False, str(e), []


# Ollama Models
LANGUAGE_MODEL = os.environ.get("OLLAMA_LLM", "llama3.2")
EMBEDDING_MODEL = os.environ.get("OLLAMA_EMBED", "nomic-embed-text")
OLLAMA_BASE_URL = get_ollama_base_url()

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
