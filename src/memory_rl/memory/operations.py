"""KEEP / COMPRESS / PRUNE with structured results (plan.md §6.3).

Operations live here, not in the policy: a policy selects an ``OperationKind``
and the engine decides whether that action is legal. Failures are reported as
failures, never as success-shaped results.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import ClassVar, Protocol

from memory_rl.memory.models import (
    MemoryItem,
    MemoryStatus,
    MemoryType,
    OperationKind,
    OperationResult,
)
from memory_rl.memory.tokenizer import Tokenizer


class Compressor(Protocol):
    """Fixed, non-learned compressor. Returning the input means "no compression"."""

    @property
    def name(self) -> str: ...

    def compress(self, text: str) -> str | None: ...


_EDGE_PUNCTUATION = re.compile(r"^\W+|\W+$")


def _bare_word(word: str) -> str:
    return _EDGE_PUNCTUATION.sub("", word).lower()


@dataclass(frozen=True, slots=True)
class RuleBasedCompressor:
    """Deterministic whitespace + filler-word removal.

    It never rewrites numbers, identifiers, or in-word punctuation, and it makes
    no semantic-preservation claim: answer accuracy is measured separately.
    """

    name: str = "rule-based-v1"
    filler_words: ClassVar[frozenset[str]] = frozenset(
        {
            "actually",
            "basically",
            "honestly",
            "just",
            "kindly",
            "literally",
            "please",
            "really",
            "simply",
            "very",
        }
    )

    def compress(self, text: str) -> str | None:
        normalized = " ".join(text.split())
        if not normalized:
            return normalized
        kept = [
            word
            for word in normalized.split(" ")
            if _bare_word(word) not in self.filler_words
        ]
        return " ".join(kept)


@dataclass(frozen=True, slots=True)
class IdentityCompressor:
    """Never shortens text; used to prove unavailable compression is reported."""

    name: str = "identity"

    def compress(self, text: str) -> str | None:
        return text


def _active_tokens(item: MemoryItem) -> int:
    return 0 if item.status is MemoryStatus.PRUNED else item.token_count


def _failure(item: MemoryItem, operation: OperationKind, reason: str) -> OperationResult:
    active = _active_tokens(item)
    return OperationResult(
        memory_id=item.memory_id,
        operation=operation,
        success=False,
        changed=False,
        reason=reason,
        previous_status=item.status,
        new_status=item.status,
        previous_token_count=active,
        new_token_count=active,
        token_delta=0,
        item=item,
    )


def _compression_candidate(
    item: MemoryItem, compressor: Compressor, tokenizer: Tokenizer
) -> tuple[str | None, str]:
    """Return ``(candidate, reason)``; ``candidate`` is None when unavailable."""
    if item.status is MemoryStatus.PRUNED:
        return None, "item is pruned; COMPRESS is not a valid action"
    if item.memory_type is MemoryType.SYSTEM:
        return None, "system instructions are protected; COMPRESS is not a valid action"
    candidate = compressor.compress(item.text)
    if candidate is None:
        return None, "compressor returned no output"
    if not candidate.strip():
        return None, "compressed text is empty"
    if candidate == item.text:
        return None, "compressed text is unchanged"
    new_count = tokenizer.count(candidate)
    if new_count >= item.token_count:
        return None, f"compressed text is not shorter ({new_count} >= {item.token_count} tokens)"
    return candidate, "ok"


def can_compress(item: MemoryItem, compressor: Compressor, tokenizer: Tokenizer) -> bool:
    """True only when the fixed compressor yields a shorter valid representation."""
    candidate, _ = _compression_candidate(item, compressor, tokenizer)
    return candidate is not None


def keep_item(item: MemoryItem) -> OperationResult:
    if item.status is MemoryStatus.PRUNED:
        return _failure(item, OperationKind.KEEP, "item is pruned; KEEP is not a valid action")
    active = _active_tokens(item)
    return OperationResult(
        memory_id=item.memory_id,
        operation=OperationKind.KEEP,
        success=True,
        changed=False,
        reason="kept unchanged",
        previous_status=item.status,
        new_status=item.status,
        previous_token_count=active,
        new_token_count=active,
        token_delta=0,
        item=item,
    )


def compress_item(
    item: MemoryItem, compressor: Compressor, tokenizer: Tokenizer
) -> OperationResult:
    candidate, reason = _compression_candidate(item, compressor, tokenizer)
    if candidate is None:
        return _failure(item, OperationKind.COMPRESS, reason)
    new_count = tokenizer.count(candidate)
    updated = replace(
        item,
        text=candidate,
        token_count=new_count,
        status=MemoryStatus.COMPRESSED,
        compression_count=item.compression_count + 1,
    )
    return OperationResult(
        memory_id=item.memory_id,
        operation=OperationKind.COMPRESS,
        success=True,
        changed=True,
        reason=f"compressed with {compressor.name}",
        previous_status=item.status,
        new_status=updated.status,
        previous_token_count=item.token_count,
        new_token_count=new_count,
        token_delta=new_count - item.token_count,
        item=updated,
    )


def prune_item(item: MemoryItem) -> OperationResult:
    if item.status is MemoryStatus.PRUNED:
        return _failure(item, OperationKind.PRUNE, "item is already pruned")
    if item.memory_type is MemoryType.SYSTEM:
        return _failure(item, OperationKind.PRUNE, "system instructions are protected")
    updated = replace(item, status=MemoryStatus.PRUNED)
    return OperationResult(
        memory_id=item.memory_id,
        operation=OperationKind.PRUNE,
        success=True,
        changed=True,
        reason="pruned; original retained in the archive",
        previous_status=item.status,
        new_status=MemoryStatus.PRUNED,
        previous_token_count=item.token_count,
        new_token_count=0,
        token_delta=-item.token_count,
        item=updated,
    )


def valid_actions(
    item: MemoryItem, compressor: Compressor, tokenizer: Tokenizer
) -> tuple[OperationKind, ...]:
    """Action mask in canonical order (KEEP, COMPRESS, PRUNE).

    Pruned items and protected system content are never candidates. If this
    returns ``()`` the caller must terminate with an explicit reason instead of
    looping (plan.md §7 Phase 4).
    """
    if item.status is MemoryStatus.PRUNED or item.memory_type is MemoryType.SYSTEM:
        return ()
    actions = [OperationKind.KEEP]
    if can_compress(item, compressor, tokenizer):
        actions.append(OperationKind.COMPRESS)
    actions.append(OperationKind.PRUNE)
    return tuple(actions)
