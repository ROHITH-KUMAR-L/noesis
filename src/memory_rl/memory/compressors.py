"""Fixed, non-learned compressors for agent tool outputs (plan.md §3 "Compression").

They make no meaning-preservation claim: a compressed observation may lose the
fact the agent needs later, and task success is what measures that.
"""

from __future__ import annotations

from dataclasses import dataclass

from memory_rl.memory.operations import RuleBasedCompressor


@dataclass(frozen=True, slots=True)
class LineTruncationCompressor:
    """Filler-word removal, then keep the first ``words_per_line`` words per line.

    Lists of short records keep their identifiers; long prose lines lose their
    tails. Empty lines are dropped.
    """

    words_per_line: int = 8
    name: str = "line-truncation-v1"

    def __post_init__(self) -> None:
        if self.words_per_line < 1:
            raise ValueError("words_per_line must be >= 1")

    def compress(self, text: str) -> str | None:
        lines = []
        for raw in text.splitlines():
            cleaned = RuleBasedCompressor().compress(raw) or ""
            words = cleaned.split()
            if not words:
                continue
            kept = words[: self.words_per_line]
            lines.append(" ".join(kept) + (" …" if len(words) > self.words_per_line else ""))
        return "\n".join(lines)
