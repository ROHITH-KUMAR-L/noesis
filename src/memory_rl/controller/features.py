"""Controller observation schema (plan.md §6.6).

Fixed shapes: ``MAX_ITEMS`` item slots (padded) × ``ITEM_FEATURES`` plus
``GLOBAL_FEATURES``. All values are finite and bounded (roughly [0, 1.5]).
Inputs are inference-available only (``ControllerState``): no gold labels.
Bump ``SCHEMA_VERSION`` on any change; checkpoints record it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from memory_rl.envs.policy_api import ControllerState
from memory_rl.memory import MemoryStatus, MemoryType, OperationKind

SCHEMA_VERSION = "keyhunt-ctrl-v1"
MAX_ITEMS = 64
ACTIONS: tuple[OperationKind, ...] = (
    OperationKind.KEEP,
    OperationKind.COMPRESS,
    OperationKind.PRUNE,
)

ITEM_FEATURE_NAMES: tuple[str, ...] = (
    "age",                  # (turn - item turn) / max_turns
    "position",             # index / (n_items - 1)
    "log_tokens",           # log1p(tokens) / log1p(4096)
    "is_assistant",
    "is_tool_result",
    "is_compressed",
    "compression_gain",     # tokens saved by COMPRESS / item tokens
    "instruction_overlap",  # lexical Jaccard(item words, instruction words)
    "suffix_share",         # tokens from this item to the end / prompt tokens (re-bill cost)
    "token_share",          # this item's rendered tokens / prompt tokens
    "can_compress",
    "is_candidate",
)  # fmt: skip

GLOBAL_FEATURE_NAMES: tuple[str, ...] = (
    "turn_progress",        # turn / max_turns
    "context_fill",         # prompt tokens / context limit
    "last_cached_fraction",
    "cache_read_multiplier",
    "cache_write_multiplier_half",
    "breakpoint_mode",
    "item_fill",            # items / MAX_ITEMS
    "prefix_share",         # protected tokens / prompt tokens
    "budget_norm",          # Phase 7 (0 when unconstrained)
    "budget_remaining",
    "cap_headroom",
)  # fmt: skip

ITEM_FEATURES = len(ITEM_FEATURE_NAMES)
GLOBAL_FEATURES = len(GLOBAL_FEATURE_NAMES)

_WORD = re.compile(r"[a-z0-9]+")
_LOG_SCALE = math.log1p(4096)


@dataclass(frozen=True, slots=True)
class Observation:
    items: NDArray[np.float32]  # [MAX_ITEMS, ITEM_FEATURES]
    item_present: NDArray[np.bool_]  # [MAX_ITEMS]
    action_mask: NDArray[np.bool_]  # [MAX_ITEMS, 3]; all False for non-candidates
    global_features: NDArray[np.float32]  # [GLOBAL_FEATURES]
    item_ids: tuple[str, ...]  # slot -> memory id (diagnostics; never a feature)

    @property
    def candidate(self) -> NDArray[np.bool_]:
        return np.asarray(self.action_mask.any(axis=1))


def _words(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def build_observation(
    state: ControllerState,
    *,
    budget_norm: float = 0.0,
    budget_remaining: float = 0.0,
    cap_headroom: float = 0.0,
) -> Observation:
    items = state.items
    if len(items) > MAX_ITEMS:
        raise ValueError(f"{len(items)} active items exceed MAX_ITEMS={MAX_ITEMS}")
    feats = np.zeros((MAX_ITEMS, ITEM_FEATURES), dtype=np.float32)
    present = np.zeros(MAX_ITEMS, dtype=bool)
    mask = np.zeros((MAX_ITEMS, len(ACTIONS)), dtype=bool)
    prompt = max(state.prompt_tokens, 1)
    max_turns = max(state.max_turns, 1)
    instruction = _words(state.instruction)
    n = len(items)
    for slot, item in enumerate(items):
        words = _words(item.text)
        union = words | instruction
        overlap = len(words & instruction) / len(union) if union else 0.0
        feats[slot] = (
            min((state.turn - item.turn_index) / max_turns, 1.5),
            slot / (n - 1) if n > 1 else 0.0,
            min(math.log1p(item.token_count) / _LOG_SCALE, 1.5),
            float(item.memory_type is MemoryType.ASSISTANT),
            float(item.memory_type is MemoryType.TOOL_RESULT),
            float(item.status is MemoryStatus.COMPRESSED),
            item.compression_gain / max(item.token_count, 1),
            overlap,
            (state.prompt_tokens - item.span_start) / prompt,
            (item.span_end - item.span_start) / prompt,
            float(OperationKind.COMPRESS in item.valid_actions),
            float(item.is_candidate),
        )
        present[slot] = True
        for a, op in enumerate(ACTIONS):
            mask[slot, a] = op in item.valid_actions
    global_features = np.array(
        (
            min(state.turn / max_turns, 1.5),
            min(state.prompt_tokens / max(state.context_limit, 1), 1.5),
            state.last_cached_fraction,
            min(state.cache_read_multiplier, 1.5),
            min(state.cache_write_multiplier / 2.0, 1.5),
            float(state.cache_mode == "breakpoint"),
            n / MAX_ITEMS,
            state.prefix_tokens / prompt,
            min(budget_norm, 1.5),
            min(max(budget_remaining, 0.0), 1.5),
            min(max(cap_headroom, 0.0), 1.5),
        ),
        dtype=np.float32,
    )
    if not (np.isfinite(feats).all() and np.isfinite(global_features).all()):
        raise ValueError("non-finite controller features")
    return Observation(
        items=feats,
        item_present=present,
        action_mask=mask,
        global_features=global_features,
        item_ids=tuple(item.memory_id for item in items),
    )
