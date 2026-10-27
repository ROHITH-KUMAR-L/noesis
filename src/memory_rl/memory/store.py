"""Active memory, an immutable archive, and a provenance log (plan.md §6.1, §6.3).

The store owns state transitions; policies only choose an ``OperationKind``.
Pruned items stay retrievable here for diagnostics but are excluded from
``active_items()``, which is the only input the prompt serializer accepts.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from memory_rl.memory.models import (
    MemoryItem,
    MemoryStatus,
    MemoryType,
    OperationKind,
    OperationResult,
)
from memory_rl.memory.operations import (
    Compressor,
    compress_item,
    keep_item,
    prune_item,
    valid_actions,
)
from memory_rl.memory.tokenizer import Tokenizer


@dataclass(frozen=True, slots=True)
class ProvenanceRecord:
    """One applied (or rejected) operation.

    Only hashes and token counts are stored, so pruned content never leaks into
    logs while each transformation stays auditable.
    """

    index: int
    memory_id: str
    operation: OperationKind
    success: bool
    reason: str
    previous_status: MemoryStatus
    new_status: MemoryStatus
    previous_token_count: int
    new_token_count: int
    previous_text_sha256: str
    new_text_sha256: str


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class MemoryStore:
    """Ordered memory items with an archive and provenance."""

    def __init__(
        self,
        items: Iterable[MemoryItem] = (),
        *,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._items: dict[str, MemoryItem] = {}
        self._order: list[str] = []
        self._log: list[ProvenanceRecord] = []
        self._tokenizer = tokenizer
        for item in items:
            self.add(item)

    def add(self, item: MemoryItem) -> None:
        if item.memory_id in self._items:
            raise ValueError(f"duplicate memory_id {item.memory_id!r}")
        if item.memory_type is MemoryType.SYSTEM:
            raise ValueError(
                "system instructions are protected and cannot be stored as memory items"
            )
        self._items[item.memory_id] = item
        self._order.append(item.memory_id)

    def get(self, memory_id: str) -> MemoryItem:
        try:
            return self._items[memory_id]
        except KeyError as exc:
            raise KeyError(f"unknown memory_id {memory_id!r}") from exc

    def has(self, memory_id: str) -> bool:
        return memory_id in self._items

    def active_items(self) -> tuple[MemoryItem, ...]:
        """Active items in insertion order; pruned items are excluded."""
        return tuple(
            self._items[memory_id]
            for memory_id in self._order
            if self._items[memory_id].status is not MemoryStatus.PRUNED
        )

    def archived_items(self) -> tuple[MemoryItem, ...]:
        """Every item, including pruned ones, in insertion order."""
        return tuple(self._items[memory_id] for memory_id in self._order)

    @property
    def active_token_count(self) -> int:
        return sum(item.token_count for item in self.active_items())

    @property
    def size(self) -> int:
        return len(self._order)

    @property
    def provenance(self) -> tuple[ProvenanceRecord, ...]:
        return tuple(self._log)

    def valid_actions(self, memory_id: str, compressor: Compressor) -> tuple[OperationKind, ...]:
        """Action mask for one item (canonical KEEP, COMPRESS, PRUNE order)."""
        return valid_actions(self.get(memory_id), compressor, self._require_tokenizer())

    def apply(
        self,
        operation: OperationKind,
        memory_id: str,
        *,
        compressor: Compressor | None = None,
    ) -> OperationResult:
        """Apply one operation, logging the attempt and updating state on success."""
        item = self.get(memory_id)
        if operation is OperationKind.KEEP:
            result = keep_item(item)
        elif operation is OperationKind.PRUNE:
            result = prune_item(item)
        elif operation is OperationKind.COMPRESS:
            if compressor is None:
                raise ValueError("COMPRESS requires a compressor")
            result = compress_item(item, compressor, self._require_tokenizer())
        else:
            raise ValueError(f"unknown operation {operation!r}")

        self._record(item, result)
        if result.success:
            self._items[memory_id] = result.item
        return result

    def _require_tokenizer(self) -> Tokenizer:
        if self._tokenizer is None:
            raise ValueError(
                "this MemoryStore was created without a tokenizer; compression is unavailable"
            )
        return self._tokenizer

    def _record(self, previous: MemoryItem, result: OperationResult) -> None:
        self._log.append(
            ProvenanceRecord(
                index=len(self._log),
                memory_id=result.memory_id,
                operation=result.operation,
                success=result.success,
                reason=result.reason,
                previous_status=previous.status,
                new_status=result.new_status,
                previous_token_count=result.previous_token_count,
                new_token_count=result.new_token_count,
                previous_text_sha256=_sha256(previous.text),
                new_text_sha256=_sha256(result.item.text),
            )
        )
