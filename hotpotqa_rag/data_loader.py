"""Data loader for the HotpotQA distractor dataset."""

import json
import pickle
import logging
from dataclasses import dataclass, asdict
from typing import Optional
from pathlib import Path

from hotpotqa_rag.config import (
    DATASET_NAME,
    DATASET_CONFIG,
    DATASET_SPLIT,
    EVAL_SUBSET_SIZE,
    CACHE_DIR,
)

logger = logging.getLogger("hotpotqa_rag")


@dataclass
class CandidateParagraph:
    """A single candidate passage from the 10 context paragraphs."""
    index: int                  # 0 to 9 index within question context
    title: str
    sentences: list[str]
    is_supporting: bool = False

    @property
    def full_text(self) -> str:
        """Combined paragraph text."""
        return " ".join(self.sentences).strip()

    @property
    def formatted_text(self) -> str:
        """Formatted with title for indexing and retrieval."""
        return f"{self.title}: {self.full_text}"


@dataclass
class HotpotQuestion:
    """A single HotpotQA question instance with 10 candidate paragraphs."""
    id: str
    question: str
    answer: str
    question_type: str          # "comparison" or "bridge"
    level: str                  # "easy", "medium", "hard"
    gold_titles: list[str]      # Titles of gold supporting paragraphs
    paragraphs: list[CandidateParagraph]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "answer": self.answer,
            "question_type": self.question_type,
            "level": self.level,
            "gold_titles": self.gold_titles,
            "paragraphs": [asdict(p) for p in self.paragraphs],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HotpotQuestion":
        paragraphs = [CandidateParagraph(**p) for p in data["paragraphs"]]
        return cls(
            id=data["id"],
            question=data["question"],
            answer=data["answer"],
            question_type=data.get("question_type", ""),
            level=data.get("level", ""),
            gold_titles=data["gold_titles"],
            paragraphs=paragraphs,
        )


def load_hotpotqa_subset(
    subset_size: int = EVAL_SUBSET_SIZE,
    cache_path: Optional[Path] = None,
    force_reload: bool = False,
) -> list[HotpotQuestion]:
    """Load the first `subset_size` questions from the HotpotQA distractor validation split.
    
    Caches parsed questions to disk for fast reload.
    """
    if cache_path is None:
        cache_path = CACHE_DIR / f"hotpotqa_{DATASET_SPLIT}_{subset_size}.pkl"

    if not force_reload and cache_path.exists():
        logger.info(f"Loading cached HotpotQA dataset from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    sample_json = CACHE_DIR.parent / "sample_questions.json"
    local_parquet = CACHE_DIR.parent / "validation.parquet"
    if local_parquet.exists() and local_parquet.stat().st_size > 20_000_000:
        logger.info(f"Loading HotpotQA from local parquet: {local_parquet}...")
        from datasets import Dataset
        ds = Dataset.from_parquet(str(local_parquet))
    else:
        try:
            logger.info(f"Downloading/loading HotpotQA ({DATASET_CONFIG}) split='{DATASET_SPLIT}' from Hugging Face...")
            from datasets import load_dataset
            ds = load_dataset(DATASET_NAME, data_files={"validation": "distractor/validation-00000-of-00001.parquet"}, split=DATASET_SPLIT)
        except Exception as e:
            if sample_json.exists():
                logger.warning(f"Could not reach Hugging Face ({e}). Falling back to bundled {sample_json}...")
                with open(sample_json, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                return [HotpotQuestion.from_dict(d) for d in cached_data[:subset_size]]
            raise e
    logger.info(f"Loaded {len(ds)} raw samples. Slicing first {subset_size}...")


    questions: list[HotpotQuestion] = []
    for idx, item in enumerate(ds):
        if idx >= subset_size:
            break

        gold_titles = list(set(item["supporting_facts"]["title"]))
        context_titles = item["context"]["title"]
        context_sentences = item["context"]["sentences"]

        paragraphs: list[CandidateParagraph] = []
        for p_idx, (title, sents) in enumerate(zip(context_titles, context_sentences)):
            is_gold = title in gold_titles
            paragraphs.append(
                CandidateParagraph(
                    index=p_idx,
                    title=title,
                    sentences=sents,
                    is_supporting=is_gold,
                )
            )

        questions.append(
            HotpotQuestion(
                id=item["id"],
                question=item["question"].strip(),
                answer=item["answer"].strip(),
                question_type=item.get("type", "unknown"),
                level=item.get("level", "unknown"),
                gold_titles=gold_titles,
                paragraphs=paragraphs,
            )
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(questions, f)
    logger.info(f"Saved {len(questions)} parsed questions to {cache_path}")

    return questions
