"""Deterministic exact-match answerer for the synthetic templates.

This is intentionally template-bound, not a language model: it extracts the value
the question asks for from *active memory only*. If the fact was pruned the
extraction fails and it answers ``unknown``, which is what makes memory loss
measurable. A real answer model replaces this in the external evaluation.
"""

from __future__ import annotations

import re

from memory_rl.answering.interface import UNKNOWN_ANSWER
from memory_rl.memory import SerializedPrompt

#: question -> ordered extraction patterns. ``{0}`` is replaced with the previous
#: capture (escaped), which is how dependent facts are resolved.
_SPECS: dict[str, tuple[str, ...]] = {
    "What is the user's name?": (r"My name is ([^.\n]+)",),
    "When is the deployment window?": (r"The deployment window is ([^.\n]+)",),
    "What is the invoice total?": (r"The invoice total is ([^.\n]+)",),
    "Which ticket is assigned to the team?": (r"Ticket ([A-Z0-9-]+) is assigned to the team",),
    "Which gate and what time does maintenance start?": (
        r"The maintenance gate is ([^.\n]+)",
        r"The maintenance starts at ([^.\n]+)",
    ),
    "Who is the customer and what is their ID?": (
        r"The customer is ([^.\n]+)",
        r"The customer ID is ([^.\n]+)",
    ),
    "Who owns the maintenance window and when is it?": (
        r"The maintenance window owner is ([^.\n]+)",
        r"The maintenance window is ([^.\n]+)",
    ),
    "What is the amount on the fetched invoice?": (
        r"fetched invoice ([A-Z0-9-]+)",
        r"Invoice {0} has amount ([^.\n]+)",
    ),
    "What is the contact email for the placed order?": (
        r"placed order ([A-Z0-9-]+)",
        r"The contact email for {0} is ([^.\n]+)",
    ),
    "Who owns the fetched ticket?": (
        r"fetched ticket ([A-Z0-9-]+)",
        r"Ticket {0} is owned by ([^.\n]+)",
    ),
    "What is the vault code?": (r"The vault code is ([^.\n]+)",),
}


class SyntheticExactMatchAnswerer:
    """Extracts the requested value from the prompt's active memory block."""

    name = "synthetic-exact-match-v1"

    def answer(self, prompt: SerializedPrompt, question: str) -> str:
        patterns = _SPECS.get(question)
        if patterns is None:
            return UNKNOWN_ANSWER

        captures: list[str] = []
        previous: str | None = None
        for pattern in patterns:
            if "{0}" in pattern:
                if previous is None:
                    return UNKNOWN_ANSWER
                pattern = pattern.replace("{0}", re.escape(previous))
            match = re.search(pattern, prompt.memory_text)
            if match is None:
                return UNKNOWN_ANSWER
            previous = match.group(1).strip()
            captures.append(previous)
        return ", ".join(captures) if captures else UNKNOWN_ANSWER
