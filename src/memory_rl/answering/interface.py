"""Answerer contract shared by every method (plan.md §6.2, §7 Phase 4).

The answerer sees only the serialized prompt (active memory + fixed text) and the
current question. It never receives labels, required-fact ids, or the reference
answer.
"""

from __future__ import annotations

from typing import Protocol

from memory_rl.memory import SerializedPrompt

#: Returned by synthetic answerers when the needed value is absent from memory.
UNKNOWN_ANSWER = "unknown"


class Answerer(Protocol):
    @property
    def name(self) -> str: ...

    def answer(self, prompt: SerializedPrompt, question: str) -> str: ...


def normalize_answer(text: str) -> str:
    """Whitespace-collapsed, case-folded comparison form for exact matching."""
    return " ".join(text.split()).strip().casefold()


def exact_match(predicted: str, expected: str) -> bool:
    """Exact match after normalization; the scorer for synthetic tasks."""
    return normalize_answer(predicted) == normalize_answer(expected)
