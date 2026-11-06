"""LLMLingua-2 baseline (optional extra ``[llmlingua]``, plan.md §7 Phase 5).

Compress tool outputs older than M turns once with LLMLingua-2 at rate r. The
package (and a ~500 MB model download) is not installed by default; this module
fails loudly instead of silently falling back to another compressor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memory_rl.envs.policy_api import ControllerState, Decision
from memory_rl.memory import MemoryType, OperationKind


def llmlingua_available() -> bool:
    try:
        import llmlingua  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class LLMLinguaCompressor:
    rate: float = 0.5
    model: str = "microsoft/llmlingua-2-xlm-roberta-large-meetingbank"
    name: str = "llmlingua-2"
    _impl: Any = field(default=None, repr=False)

    def compress(self, text: str) -> str | None:
        if self._impl is None:
            try:
                from llmlingua import PromptCompressor
            except ImportError as exc:
                raise RuntimeError(
                    "LLMLingua baseline needs `uv pip install -e '.[llmlingua]'`"
                ) from exc
            self._impl = PromptCompressor(model_name=self.model, use_llmlingua2=True)
        result = self._impl.compress_prompt(text, rate=self.rate)
        return str(result["compressed_prompt"])


@dataclass
class LLMLinguaPolicy:
    m: int = 2
    rate: float = 0.5
    name: str = "llmlingua"
    compressor: LLMLinguaCompressor = field(default_factory=LLMLinguaCompressor)

    def reset(self, seed: int) -> None:
        self.compressor.rate = self.rate

    def decide(self, state: ControllerState) -> Decision:
        actions = {
            item.memory_id: OperationKind.COMPRESS
            for item in state.candidates
            if item.memory_type is MemoryType.TOOL_RESULT
            and state.turn - item.turn_index > self.m
            and OperationKind.COMPRESS in item.valid_actions
        }
        return Decision(edit=bool(actions), actions=actions)
