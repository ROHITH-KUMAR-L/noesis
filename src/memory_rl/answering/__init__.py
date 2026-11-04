"""Fixed answerers and their shared contract."""

from memory_rl.answering.interface import (
    UNKNOWN_ANSWER,
    Answerer,
    exact_match,
    normalize_answer,
)
from memory_rl.answering.synthetic import SyntheticExactMatchAnswerer

__all__ = [
    "UNKNOWN_ANSWER",
    "Answerer",
    "SyntheticExactMatchAnswerer",
    "exact_match",
    "normalize_answer",
]
