"""Utility functions: JSON extraction, text normalization, and logging."""

import re
import json
import string
import logging
from typing import Any, Optional

logger = logging.getLogger("hotpotqa_rag")


def extract_json_from_text(text: str) -> Optional[dict[str, Any]]:
    """Extract and parse JSON from model output that may include markdown fences or chatter.
    
    Returns parsed dictionary or None if parsing fails.
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = text.strip()

    # 1. Try direct parse first
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 2. Look for markdown code fence ```json ... ``` or ``` ... ```
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if fence_match:
        try:
            data = json.loads(fence_match.group(1).strip())
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    # 3. Find first '{' and last '}'
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = cleaned[first_brace : last_brace + 1]
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except Exception:
            # Try removing trailing commas before closing braces/brackets
            fixed = re.sub(r",\s*([\]}])", r"\1", candidate)
            try:
                data = json.loads(fixed)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

            # 4. Fix unquoted string values (common llama3.2 failure mode)
            # e.g. "reason": The evidence is sufficient...
            try:
                def _quote_unquoted_value(m: re.Match) -> str:
                    key = m.group(1)
                    val = m.group(2).strip()
                    if val.lower() in ("true", "false", "null") or re.match(r"^-?\d+(\.\d+)?$", val):
                        return m.group(0)
                    escaped_val = val.replace('"', '\\"')
                    return f'"{key}": "{escaped_val}"'

                unquoted_fixed = re.sub(
                    r'"(\w+)":\s+([^"{\[\s,][^\n\r,}]*)',
                    _quote_unquoted_value,
                    fixed,
                )
                data = json.loads(unquoted_fixed)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

    return None


def normalize_answer(text: str) -> str:
    """Normalize text for HotpotQA evaluation (standard SQuAD/HotpotQA protocol).
    
    Lower text, strip citations, and remove punctuation, articles and extra whitespace.
    """
    def remove_citations(text: str) -> str:
        return re.sub(r"\[(?:Passage\s*)?\d+\]", " ", text, flags=re.IGNORECASE)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(remove_citations(lower(text)))))
