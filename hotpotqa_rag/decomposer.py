"""Query decomposition module with strict JSON validation and fallback handling."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import ollama

from hotpotqa_rag.config import (
    LANGUAGE_MODEL,
    OLLAMA_BASE_URL,
    LLM_TEMPERATURE,
    LLM_NUM_CTX,
    MAX_SUB_QUESTIONS,
)
from hotpotqa_rag.utils import extract_json_from_text

logger = logging.getLogger("hotpotqa_rag")

DECOMPOSITION_SYSTEM_PROMPT = """You are a query decomposition expert for multi-hop question answering.
Analyze the user's question and determine if it requires multiple reasoning steps (multi-hop) across different entities or topics.

If it requires multi-hop reasoning, set "needs_decomposition" to true and break it into 1 to 3 atomic sub-questions that can be answered sequentially.
If it is a single-fact or straightforward question, set "needs_decomposition" to false and return the original question as the single sub-question.

You MUST reply with ONLY a single valid JSON object. No explanation, no intro, no markdown outside the JSON.
Schema:
{
  "needs_decomposition": <true or false>,
  "sub_questions": ["<sub-question 1>", "<optional sub-question 2>", "<optional sub-question 3>"]
}"""


@dataclass
class DecompositionResult:
    original_question: str
    needs_decomposition: bool
    sub_questions: list[str]
    fallback_fired: bool
    is_malformed: bool
    raw_output: str
    error_message: Optional[str] = None


def decompose_question(
    question: str,
    client: Optional[ollama.Client] = None,
    model: str = LANGUAGE_MODEL,
) -> DecompositionResult:
    """Prompt llama3.2 to decompose a multi-hop question into 1-3 sub-questions.
    
    If parsing fails or output is malformed, falls back to the original question as
    a single sub-question, logs the error, and flags fallback_fired=True.
    """
    if client is None:
        client = ollama.Client(host=OLLAMA_BASE_URL)

    user_prompt = f"Question: {question}"

    raw_output = ""
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": DECOMPOSITION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": LLM_TEMPERATURE,
                "num_ctx": LLM_NUM_CTX,
            },
        )
        raw_output = response["message"]["content"]
    except Exception as e:
        logger.error(f"[DECOMPOSITION ERROR] Ollama call failed for question '{question}': {e}")
        return DecompositionResult(
            original_question=question,
            needs_decomposition=False,
            sub_questions=[question],
            fallback_fired=True,
            is_malformed=True,
            raw_output="",
            error_message=f"LLM API Exception: {e}",
        )

    # Parse JSON
    parsed = extract_json_from_text(raw_output)

    # Validate Schema
    if parsed is None:
        error_msg = "Could not parse valid JSON from output"
        logger.warning(
            f"[DECOMPOSITION MALFORMED] Question: '{question}' | Error: {error_msg} | Raw: {raw_output!r}"
        )
        return DecompositionResult(
            original_question=question,
            needs_decomposition=False,
            sub_questions=[question],
            fallback_fired=True,
            is_malformed=True,
            raw_output=raw_output,
            error_message=error_msg,
        )

    needs_decomp = parsed.get("needs_decomposition")
    sub_qs = parsed.get("sub_questions")

    # Validate needs_decomposition
    if not isinstance(needs_decomp, bool):
        error_msg = f"needs_decomposition must be bool, got {type(needs_decomp)}"
        logger.warning(
            f"[DECOMPOSITION MALFORMED] Question: '{question}' | Error: {error_msg} | Raw: {raw_output!r}"
        )
        return DecompositionResult(
            original_question=question,
            needs_decomposition=False,
            sub_questions=[question],
            fallback_fired=True,
            is_malformed=True,
            raw_output=raw_output,
            error_message=error_msg,
        )

    # Validate sub_questions
    if not isinstance(sub_qs, list) or not sub_qs:
        error_msg = "sub_questions must be a non-empty list of strings"
        logger.warning(
            f"[DECOMPOSITION MALFORMED] Question: '{question}' | Error: {error_msg} | Raw: {raw_output!r}"
        )
        return DecompositionResult(
            original_question=question,
            needs_decomposition=False,
            sub_questions=[question],
            fallback_fired=True,
            is_malformed=True,
            raw_output=raw_output,
            error_message=error_msg,
        )

    # Filter and clean sub-questions
    cleaned_sub_qs = [str(q).strip() for q in sub_qs if str(q).strip()]
    if not cleaned_sub_qs:
        error_msg = "All sub-questions were empty strings"
        logger.warning(
            f"[DECOMPOSITION MALFORMED] Question: '{question}' | Error: {error_msg} | Raw: {raw_output!r}"
        )
        return DecompositionResult(
            original_question=question,
            needs_decomposition=False,
            sub_questions=[question],
            fallback_fired=True,
            is_malformed=True,
            raw_output=raw_output,
            error_message=error_msg,
        )

    # Bound sub-questions to 1-3
    if len(cleaned_sub_qs) > MAX_SUB_QUESTIONS:
        cleaned_sub_qs = cleaned_sub_qs[:MAX_SUB_QUESTIONS]

    return DecompositionResult(
        original_question=question,
        needs_decomposition=needs_decomp,
        sub_questions=cleaned_sub_qs,
        fallback_fired=False,
        is_malformed=False,
        raw_output=raw_output,
    )


REFORMULATION_SYSTEM_PROMPT = """You are a search query refinement specialist for multi-hop question answering.
A previous search for a sub-question retrieved evidence that was judged INSUFFICIENT by an evaluator.
Your job is to rewrite the sub-question into a more targeted search query that directly addresses the missing information identified by the critique.

Instructions:
1. Focus on specific entity names, relationships, dates, or attributes that were missing.
2. Output ONLY the refined query string without any quotes, intro, or explanation."""


def reformulate_sub_question(
    sub_question: str,
    insufficiency_reason: str,
    client: Optional[ollama.Client] = None,
    model: str = LANGUAGE_MODEL,
) -> str:
    """Rewrite a sub-question based on the grader's stated insufficiency reason."""
    if client is None:
        client = ollama.Client(host=OLLAMA_BASE_URL)

    user_prompt = f"""Original Sub-Question: {sub_question}
Critique on why evidence was insufficient: {insufficiency_reason}

Refined Search Query:"""

    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": REFORMULATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": LLM_TEMPERATURE,
                "num_ctx": LLM_NUM_CTX,
            },
        )
        refined = response["message"]["content"].strip().strip('"\'')
        return refined if refined else sub_question
    except Exception as e:
        logger.warning(f"Failed to reformulate query: {e}. Keeping original sub-question.")
        return sub_question

