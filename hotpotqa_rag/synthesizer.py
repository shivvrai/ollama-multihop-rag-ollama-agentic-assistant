"""Answer synthesis module: Generates final answer with passage citations."""

import re
import logging
from dataclasses import dataclass
from typing import Optional

import ollama

from hotpotqa_rag.config import (
    LANGUAGE_MODEL,
    OLLAMA_BASE_URL,
    LLM_TEMPERATURE,
    LLM_NUM_CTX,
)

logger = logging.getLogger("hotpotqa_rag")

SYNTHESIS_SYSTEM_PROMPT = """You are a precise factual question answering system.
Given evidence passages, answer the question in AS FEW WORDS AS POSSIBLE.

CRITICAL RULES:
- If the answer is yes or no, respond with ONLY "yes" or "no".
- If the answer is a name, date, place, or number, respond with ONLY that entity.
- Do NOT explain your reasoning. Do NOT write full sentences unless absolutely necessary.
- Cite passages using [Passage N] tags AFTER your short answer.
- Your answer should typically be 1-5 words.

GOOD examples:
- Question: "Were X and Y the same nationality?" → "yes [Passage 1][Passage 2]"
- Question: "What position did she hold?" → "Chief of Protocol [Passage 3]"
- Question: "In which city is X located?" → "Istanbul [Passage 1]"

BAD examples (DO NOT DO THIS):
- "Yes, X and Y were of the same nationality because..." (too verbose)
- "Based on the evidence, it appears that..." (too verbose)"""


@dataclass
class SynthesisResult:
    answer: str
    cited_passages: list[str]
    raw_output: str


def extract_citations(text: str) -> list[str]:
    """Extract citations matching [Passage X] or [X] from text."""
    matches = re.findall(r"\[(?:Passage\s*)?(\d+)\]", text, re.IGNORECASE)
    seen = set()
    ordered = []
    for m in matches:
        tag = f"Passage {m}"
        if tag not in seen:
            seen.add(tag)
            ordered.append(tag)
    return ordered


def synthesize_final_answer(
    question: str,
    evidence_passages: list[str],
    sub_questions: Optional[list[str]] = None,
    client: Optional[ollama.Client] = None,
    model: str = LANGUAGE_MODEL,
) -> SynthesisResult:
    """Prompt llama3.2 to synthesize a final answer citing retrieved passages."""
    if client is None:
        client = ollama.Client(host=OLLAMA_BASE_URL)

    formatted_evidence = "\n\n".join(
        f"[Passage {i+1}]: {p}" for i, p in enumerate(evidence_passages)
    )

    sub_q_section = ""
    if sub_questions and len(sub_questions) > 1:
        sub_q_bullets = "\n".join(f"- {sq}" for sq in sub_questions)
        sub_q_section = f"\nSub-questions explored:\n{sub_q_bullets}\n"

    user_prompt = f"""Original Question: {question}
{sub_q_section}
Verified Evidence:
{formatted_evidence}

Please provide a concise, factual answer with explicit citations (e.g. [Passage 1]):"""

    raw_output = ""
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": LLM_TEMPERATURE,
                "num_ctx": LLM_NUM_CTX,
            },
        )
        raw_output = response["message"]["content"].strip()
    except Exception as e:
        logger.error(f"[SYNTHESIS ERROR] Ollama call failed for question '{question}': {e}")
        raw_output = f"Error generating answer: {e}"

    citations = extract_citations(raw_output)

    return SynthesisResult(
        answer=raw_output,
        cited_passages=citations,
        raw_output=raw_output,
    )
