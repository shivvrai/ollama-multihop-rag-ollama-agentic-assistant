"""Sufficiency grader: Judges whether retrieved passages contain enough evidence to answer a question."""

import logging
from dataclasses import dataclass
from typing import Optional

import ollama

from hotpotqa_rag.config import (
    LANGUAGE_MODEL,
    OLLAMA_BASE_URL,
    LLM_TEMPERATURE,
    LLM_NUM_CTX,
    MAX_TOTAL_ATTEMPTS,
)
from hotpotqa_rag.utils import extract_json_from_text

logger = logging.getLogger("hotpotqa_rag")

GRADER_SYSTEM_PROMPT = """You are an evidence sufficiency judge for question answering.
Analyze the given question and the retrieved evidence passages.
Determine if the evidence contains sufficient factual information to answer the question accurately and completely.

You MUST reply with ONLY a single valid JSON object. No explanation, no intro, no markdown outside the JSON.
Schema:
{
  "sufficient": <true or false>,
  "reason": "<one sentence explaining what evidence is present or missing>"
}"""


@dataclass
class GraderResult:
    sufficient: bool
    reason: str
    attempt: int
    fallback_fired: bool
    is_malformed: bool
    raw_output: str
    error_message: Optional[str] = None


def grade_evidence_sufficiency(
    question: str,
    evidence_passages: list[str],
    attempt: int = 1,
    client: Optional[ollama.Client] = None,
    model: str = LANGUAGE_MODEL,
    max_attempts: int = MAX_TOTAL_ATTEMPTS,
) -> GraderResult:
    """Judge if retrieved evidence passages are sufficient to answer the question.
    
    If output is malformed:
      - If attempt < max_attempts: default to sufficient=False to trigger a retry.
      - If attempt >= max_attempts: default to sufficient=True to terminate the loop.
    Logs all malformed outputs with raw text and the question.
    """
    if client is None:
        client = ollama.Client(host=OLLAMA_BASE_URL)

    formatted_evidence = "\n\n".join(
        f"[Passage {i+1}]: {p}" for i, p in enumerate(evidence_passages)
    )

    user_prompt = f"""Question to answer: {question}

Retrieved Evidence Passages:
{formatted_evidence}"""

    raw_output = ""
    try:
        response = client.chat(
            model=model,
            messages=[
                {"role": "system", "content": GRADER_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": LLM_TEMPERATURE,
                "num_ctx": LLM_NUM_CTX,
            },
        )
        raw_output = response["message"]["content"]
    except Exception as e:
        logger.error(f"[GRADER ERROR] Ollama call failed for question '{question}': {e}")
        # On API failure, treat as malformed
        is_terminal = attempt >= max_attempts
        sufficient_fallback = is_terminal
        reason_fallback = (
            "API failure on final retry; defaulting to sufficient to terminate loop"
            if is_terminal
            else "API failure; defaulting to insufficient to trigger retry"
        )
        return GraderResult(
            sufficient=sufficient_fallback,
            reason=reason_fallback,
            attempt=attempt,
            fallback_fired=True,
            is_malformed=True,
            raw_output="",
            error_message=f"LLM API Exception: {e}",
        )

    parsed = extract_json_from_text(raw_output)

    # Check for malformed JSON or invalid schema
    is_valid = (
        parsed is not None
        and isinstance(parsed, dict)
        and "sufficient" in parsed
        and isinstance(parsed["sufficient"], bool)
        and "reason" in parsed
        and isinstance(parsed["reason"], str)
    )

    if not is_valid:
        error_msg = "Could not parse valid JSON with 'sufficient' (bool) and 'reason' (str)"
        logger.warning(
            f"[GRADER MALFORMED] Question: '{question}' | Attempt: {attempt}/{max_attempts} | "
            f"Error: {error_msg} | Raw: {raw_output!r}"
        )

        is_terminal = attempt >= max_attempts
        sufficient_fallback = is_terminal
        reason_fallback = (
            "Malformed JSON on final retry limit; defaulted to sufficient to terminate loop"
            if is_terminal
            else "Malformed JSON from grader; defaulted to insufficient to trigger retry"
        )

        return GraderResult(
            sufficient=sufficient_fallback,
            reason=reason_fallback,
            attempt=attempt,
            fallback_fired=True,
            is_malformed=True,
            raw_output=raw_output,
            error_message=error_msg,
        )

    return GraderResult(
        sufficient=parsed["sufficient"],
        reason=parsed["reason"].strip(),
        attempt=attempt,
        fallback_fired=False,
        is_malformed=False,
        raw_output=raw_output,
    )
