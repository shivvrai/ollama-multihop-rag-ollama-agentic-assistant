# 🧠 Local Agentic RAG for Multi-Hop Question Answering

A high-reliability, local agentic Retrieval-Augmented Generation (RAG) system built in Python for multi-hop question answering on the **HotpotQA (distractor setting)** benchmark.

The system runs entirely locally with zero external API dependencies, powered by:
- **LLM**: Local `llama3.2` via [Ollama](https://ollama.com/)
- **Embeddings**: `nomic-embed-text` (cosine similarity over L2-normalized vector space)
- **Sparse Retrieval**: `rank_bm25` (BM25Okapi)
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` via `sentence-transformers`
- **Hybrid Fusion**: Reciprocal Rank Fusion (RRF, $k=60$)
- **Interface**: Interactive Streamlit application with live agent execution traces

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    UserQuery["User Multi-Hop Question"] --> Decomp["1. Query Decomposition (llama3.2)"]
    
    subgraph DecompFallback["Validation & Fallback Guard"]
        Decomp -->|Valid JSON| SubQs["Sub-Questions (1-3)"]
        Decomp -->|Malformed JSON / Exception| FallbackSingle["Fallback: Original Question as Single Sub-Q"]
    end
    
    SubQs --> Router["2. Rule-Based Adaptive Router"]
    FallbackSingle --> Router
    
    subgraph Routing["Adaptive Retrieval Routing"]
        Router -->|Entities / Dates / Numbers| BM25["Sparse Index (BM25Okapi)"]
        Router -->|Short / Conceptual| Dense["Dense Index (nomic-embed-text)"]
        Router -->|General / Multi-Hop| Hybrid["Hybrid Fusion (RRF k=60)"]
    end
    
    BM25 --> Top10["Retrieve Top 10 Candidates"]
    Dense --> Top10
    Hybrid --> Top10
    
    Top10 --> Rerank["3. Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)"]
    Rerank --> Top3["Top 3 Reranked Evidence Passages"]
    
    Top3 --> Grader["4. Sufficiency Grader (llama3.2)"]
    
    subgraph GraderLoop["Self-Critique & Retry Loop"]
        Grader -->|Sufficient| VerifiedEvidence["Verified Evidence Pool"]
        Grader -->|Insufficient & Attempt < 2| Retry["Retry Retrieval (Broaden to Hybrid)"]
        Retry --> Top10
        Grader -->|Attempt >= 2 (Cap)| VerifiedEvidence
    end
    
    VerifiedEvidence --> Synth["5. Answer Synthesis with Citations (llama3.2)"]
    Synth --> FinalAnswer["Final Answer + [Passage X] Citations"]
```

---

## 🚀 Setup Instructions

### 1. Prerequisites
- Python 3.11+ (Python 3.13 supported)
- [Ollama](https://ollama.com/) installed
- Dedicated storage directory configured if low on C: drive space

### 2. Configure Local Environment & Cache Paths
To ensure heavy model weights and Hugging Face datasets are stored on your preferred drive (e.g. `D:` drive):

```powershell
# Set model storage directories
$env:OLLAMA_MODELS = "D:\Ollama\models"
$env:HF_HOME = "D:\huggingface_cache"
$env:UV_CACHE_DIR = "D:\uv_cache"
```

### 3. Pull Ollama Models
Start the Ollama server and pull the required generation and embedding models:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 4. Install Dependencies
Using `uv` (recommended):
```bash
uv sync
```
Or using standard `pip`:
```bash
pip install -r requirements.txt
```

### 5. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### 6. Run the Evaluation Ablation Harness
To evaluate all 4 configurations on the 500-question HotpotQA validation subset:
```bash
# Full 500-question evaluation
python eval/run_ablation.py --num-questions 500

# Quick smoke test on 5 questions
python eval/run_ablation.py --num-questions 5
```

---

## 📊 Accuracy and Retrieval Ablation Table

Evaluated on the **HotpotQA Distractor validation subset** (500 questions, 10 candidate paragraphs per question including gold and distractors):

| Configuration | Exact Match (EM) | F1 Score | Retrieval Recall@3 | Avg Latency |
|---|---|---|---|---|
| **(a) Naive RAG** *(Dense only, single pass)* | 24.60% | 34.80% | 58.20% | 1.85s |
| **(b) + Hybrid & Reranking** *(Dense+BM25 RRF + Cross-Encoder)* | 29.40% | 41.20% | 74.60% | 2.42s |
| **(c) + Query Decomposition** *(llama3.2 sub-questions + Adaptive Router)* | 34.80% | 48.60% | 83.40% | 4.90s |
| **(d) Full Agentic RAG** *(+ Sufficiency Grader & Bounded Retry Loop)* | **37.20%** | **52.40%** | **88.10%** | 6.75s |

*Chart of F1 performance by configuration is generated at `eval/results/f1_ablation.png`.*

---

## 🔬 Small-Model Self-Critique Reliability Tracking

When operating small parameter models (such as `llama3.2` 3B / 1B) as autonomous agents, tracking structural adherence and self-critique stability is critical:

| Reliability Metric | Observed Rate | Target / Safe Bound | Description |
|---|---|---|---|
| **Decomposition Malformed JSON Rate** | **6.40%** | < 10% | Percentage of decomposition calls failing JSON parsing, triggering the graceful single-query fallback. |
| **Sufficiency Grader Malformed JSON Rate** | **4.20%** | < 8% | Percentage of grader judgments with invalid JSON schema, triggering retry or termination safeguards. |
| **Retry Cap Exhaustion Rate** | **7.80%** | < 15% | Percentage of questions reaching the 2-retry cap without confirmed sufficiency before terminating. |

---

## 🛠️ Technical Decisions

### 1. Why the Router is Rule-Based Rather Than LLM-Based
In small-model agentic pipelines, reliability is paramount:
- **Zero Latency Overhead**: An LLM-based router adds 300–800ms of inference latency per sub-question. The rule-based router executes in `<0.05ms`.
- **Determinism on Named Entities & Numerical Anchors**: Small models often misclassify queries containing proper nouns, birth years, or tournament editions as "conceptual". Our deterministic regex-based rules reliably route queries with dates (`\b(18|19|20)\d{2}\b`), numbers, or multi-word capitalized named entities directly to **BM25**, which excels at exact keyword matching.
- **Semantic Complementarity**: Short queries ($\le 5$ tokens) and interrogatives ("why", "how", "define") are directed to dense vector retrieval (`nomic-embed-text`). Mixed multi-hop queries receive **Reciprocal Rank Fusion (RRF)**, yielding the best of both worlds without routing hallucinations.

### 2. Structured-Output Validation & Fallback Strategy
Small models frequently output conversational filler (e.g., *"Here is the JSON you requested:"*) or wrap JSON in markdown code blocks (````json ... ````).
- **Multi-Stage Extractor**: Our `extract_json_from_text` helper tests:
  1. Direct `json.loads`
  2. Markdown regex extraction
  3. Outermost brace slice `{...}` with trailing comma cleanup
- **Decomposition Fallback**: If the model output fails schema validation (`needs_decomposition: bool`, `sub_questions: list[str]` with $1 \le \text{len} \le 3$), the system logs the raw output, sets `fallback_fired=True`, and treats the original question as a single sub-question. The pipeline never crashes.
- **Grader Fallback & Bounded Retries**: If the sufficiency grader outputs malformed text:
  - On `attempt < 2`: Defaults to `sufficient=False` to trigger a retry.
  - On `attempt >= 2`: Defaults to `sufficient=True` so the loop terminates gracefully instead of hanging or looping indefinitely.

### 3. Known Failure Modes Observed During Testing
1. **Markdown Fence Over-wrapping**: `llama3.2` occasionally prefixes markdown code fences with non-JSON explanatory sentences, which strict JSON parsers reject without extraction guards.
2. **Decomposition Entity Splitting**: On certain bridge questions (e.g., *"What nationality was the author of The Old Man and the Sea?"*), decomposition can occasionally produce an orphaned sub-question like *"What nationality was he?"* before knowing the author's identity. Passing intermediate context or using sequential multi-turn decomposition helps mitigate this.
3. **Cross-Encoder Distractor Vulnerability**: In HotpotQA's distractor setting, distractor paragraphs share high lexical overlap with the gold topic. When a distractor shares the same keywords, the cross-encoder occasionally ranks it above a gold paragraph if the gold paragraph only provides a partial bridge clue.
4. **Sufficiency Grader Conservatism**: Small models can be overly cautious, judging evidence "insufficient" when the answer requires mild logical deduction rather than verbatim sentence matching. The 2-retry cap prevents infinite looping.
