"""Fixed, documented, simulator-label-free observation features (plan.md §7 Phase 2).

Every feature is finite and bounded; ``FEATURE_NAMES`` is the schema and its
length is the observation dimension. Nothing here may read ``TaskLabels``: the
inputs are the current item, the question, and episode-level token counters.
"""

from __future__ import annotations

import re

import numpy as np
from numpy.typing import NDArray

from memory_rl.memory import MemoryItem, MemoryType

#: Feature schema. Changing this order or length is a breaking change and must
#: bump ``FEATURE_SCHEMA_VERSION`` (checkpoints record it).
FEATURE_NAMES: tuple[str, ...] = (
    "age_norm",  # turn_index / (episode_length - 1), [0, 1]
    "item_tokens_norm",  # item tokens / longest item in episode, [0, 1]
    "type_user",  # one-hot over the five observable memory types
    "type_assistant",
    "type_tool_result",
    "type_fact",
    "type_decision",
    "compression_norm",  # compression_count / COMPRESSION_CAP, clipped [0, 1]
    "budget_used_ratio",  # active memory tokens / budget tokens, clipped [0, 2]
    "episode_progress",  # decided candidates / total candidates, [0, 1]
    "question_relevance",  # share of question keywords present in the item, [0, 1]
)

FEATURE_SCHEMA_VERSION = "1"
FEATURE_DIM = len(FEATURE_NAMES)
COMPRESSION_CAP = 3.0
BUDGET_RATIO_CAP = 2.0

_OBSERVABLE_TYPES: tuple[MemoryType, ...] = (
    MemoryType.USER,
    MemoryType.ASSISTANT,
    MemoryType.TOOL_RESULT,
    MemoryType.FACT,
    MemoryType.DECISION,
)

_WORD = re.compile(r"[a-z0-9]+")
_MIN_KEYWORD_LEN = 3


def question_keywords(question: str) -> frozenset[str]:
    """Lowercased alphanumeric words long enough to carry meaning."""
    return frozenset(
        word for word in _WORD.findall(question.lower()) if len(word) >= _MIN_KEYWORD_LEN
    )


def _clip(value: float, upper: float) -> float:
    return float(min(max(value, 0.0), upper))


def extract_features(
    item: MemoryItem,
    *,
    question: str,
    turn_index: int,
    episode_length: int,
    max_item_tokens: int,
    active_tokens: int,
    budget_tokens: int,
    decided_candidates: int,
    total_candidates: int,
) -> NDArray[np.float32]:
    """Build one observation. Raises if any value is non-finite."""
    denominator = max(episode_length - 1, 1)
    keywords = question_keywords(question)
    item_words = set(_WORD.findall(item.text.lower()))
    relevance = len(keywords & item_words) / len(keywords) if keywords else 0.0

    values: list[float] = [
        _clip(turn_index / denominator, 1.0),
        _clip(item.token_count / max(max_item_tokens, 1), 1.0),
        *(1.0 if item.memory_type is memory_type else 0.0 for memory_type in _OBSERVABLE_TYPES),
        _clip(item.compression_count / COMPRESSION_CAP, 1.0),
        _clip(active_tokens / max(budget_tokens, 1), BUDGET_RATIO_CAP),
        _clip(decided_candidates / max(total_candidates, 1), 1.0),
        _clip(relevance, 1.0),
    ]
    observation = np.asarray(values, dtype=np.float32)
    if observation.shape != (FEATURE_DIM,):
        raise RuntimeError(f"feature dimension mismatch: {observation.shape} != ({FEATURE_DIM},)")
    if not np.isfinite(observation).all():
        raise RuntimeError("observation contains non-finite values")
    return observation


def empty_observation() -> NDArray[np.float32]:
    """Terminal/empty-episode observation (documented as all zeros)."""
    return np.zeros(FEATURE_DIM, dtype=np.float32)
