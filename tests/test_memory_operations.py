"""Phase 1 required tests: memory operations (plan.md §7, Phase 1).

Every test here maps to a bullet in the plan's "Required tests" list for the
memory engine. Counts use ``TestTokenizer`` (one token per code point) so exact
expected numbers are hand-checkable.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from memory_rl.memory import (
    IdentityCompressor,
    MemoryItem,
    MemoryStatus,
    MemoryStore,
    MemoryType,
    OperationKind,
    PromptSerializer,
    RuleBasedCompressor,
    TestTokenizer,
    valid_actions,
)

TOKENIZER = TestTokenizer()
SYSTEM = "Answer the task using only the memory below."
QUESTION = "Where does Alice live?"


def item(
    text: str,
    memory_id: str,
    memory_type: MemoryType = MemoryType.USER,
    turn_index: int = 0,
) -> MemoryItem:
    return MemoryItem.create(
        memory_id=memory_id,
        text=text,
        tokenizer=TOKENIZER,
        turn_index=turn_index,
        memory_type=memory_type,
    )


# --- operation contracts ------------------------------------------------------


def test_keep_preserves_text_id_status_and_token_count() -> None:
    original = item("The budget is 5000 tokens.", "m1")
    store = MemoryStore([original], tokenizer=TOKENIZER)

    result = store.apply(OperationKind.KEEP, "m1")

    assert result.success
    assert not result.changed
    assert result.token_delta == 0
    assert result.new_token_count == original.token_count

    kept = store.get("m1")
    assert kept.text == original.text
    assert kept.memory_id == "m1"
    assert kept.status is MemoryStatus.ACTIVE
    assert kept.token_count == original.token_count == len(original.text)


def test_successful_compression_changes_active_text_and_keeps_original() -> None:
    original = item("Please kindly note that the invoice ID is INV-2049.", "m1")
    store = MemoryStore([original], tokenizer=TOKENIZER)

    result = store.apply(OperationKind.COMPRESS, "m1", compressor=RuleBasedCompressor())

    assert result.success
    assert result.changed
    assert result.token_delta < 0

    compressed = store.get("m1")
    assert compressed.text == "note that the invoice ID is INV-2049."
    assert compressed.original_text == original.text
    assert compressed.status is MemoryStatus.COMPRESSED
    assert compressed.compression_count == 1
    assert compressed.token_count == len(compressed.text)
    assert "INV-2049" in compressed.text  # numbers/identifiers survive this compressor


@dataclass(frozen=True, slots=True)
class _LongerCompressor:
    name: str = "longer"

    def compress(self, text: str) -> str | None:
        return f"{text} with extra padding words"


@dataclass(frozen=True, slots=True)
class _NoneCompressor:
    name: str = "none"

    def compress(self, text: str) -> str | None:
        return None


@dataclass(frozen=True, slots=True)
class _EmptyCompressor:
    name: str = "empty"

    def compress(self, text: str) -> str | None:
        return "   "


@dataclass(frozen=True, slots=True)
class _ConstantCountTokenizer:
    """Counts every text as the same length, so nothing is ever "shorter"."""

    name: str = "test:constant"
    version: str = "1"

    def count(self, text: str) -> int:
        return 7


@pytest.mark.parametrize(
    "compressor",
    [IdentityCompressor(), _LongerCompressor(), _NoneCompressor(), _EmptyCompressor()],
    ids=["unchanged", "longer", "no-output", "empty"],
)
def test_failed_compression_leaves_item_untouched(compressor: object) -> None:
    original = item("Short fact: gate 4 opens at 06:00.", "m1")
    store = MemoryStore([original], tokenizer=TOKENIZER)

    result = store.apply(OperationKind.COMPRESS, "m1", compressor=compressor)  # type: ignore[arg-type]

    assert not result.success
    assert not result.changed
    assert result.token_delta == 0
    assert store.get("m1") == original
    assert store.get("m1").status is MemoryStatus.ACTIVE


def test_compression_shorter_in_characters_but_not_in_tokens_is_rejected() -> None:
    tokenizer = _ConstantCountTokenizer()
    original = MemoryItem.create(memory_id="m1", text="Please wait here.", tokenizer=tokenizer)
    store = MemoryStore([original], tokenizer=tokenizer)

    result = store.apply(OperationKind.COMPRESS, "m1", compressor=RuleBasedCompressor())

    assert not result.success
    assert "not shorter" in result.reason
    assert store.get("m1").text == original.text


def test_prune_removes_item_from_prompt_but_keeps_archive_record() -> None:
    kept = item("Alice lives in Pune.", "m1")
    dropped = item("Unrelated chatter about the weather.", "m2")
    store = MemoryStore([kept, dropped], tokenizer=TOKENIZER)
    serializer = PromptSerializer(TOKENIZER, SYSTEM)
    before = serializer.serialize(store.active_items(), QUESTION)

    result = store.apply(OperationKind.PRUNE, "m2")

    assert result.success
    assert result.changed
    assert result.new_token_count == 0
    assert result.token_delta == -dropped.token_count

    after = serializer.serialize(store.active_items(), QUESTION)
    assert "Unrelated chatter" not in after.text
    assert "m2" not in after.included_ids
    assert after.total_prompt_tokens < before.total_prompt_tokens

    archived = store.get("m2")
    assert archived.status is MemoryStatus.PRUNED
    assert archived.text == dropped.text
    assert archived.original_text == dropped.text
    assert "Unrelated chatter" in archived.original_text


def test_repeated_prune_is_rejected_and_masked() -> None:
    store = MemoryStore([item("Some fact.", "m1")], tokenizer=TOKENIZER)
    assert store.apply(OperationKind.PRUNE, "m1").success

    second = store.apply(OperationKind.PRUNE, "m1")

    assert not second.success
    assert not second.changed
    assert second.token_delta == 0
    assert store.valid_actions("m1", RuleBasedCompressor()) == ()


def test_repeated_compression_only_succeeds_while_shorter() -> None:
    store = MemoryStore([item("Please kindly really note the gate.", "m1")], tokenizer=TOKENIZER)
    compressor = RuleBasedCompressor()

    first = store.apply(OperationKind.COMPRESS, "m1", compressor=compressor)
    second = store.apply(OperationKind.COMPRESS, "m1", compressor=compressor)

    assert first.success
    assert not second.success
    assert store.get("m1").compression_count == 1
    assert store.get("m1").text == "note the gate."


def test_active_token_total_equals_sum_of_active_item_counts() -> None:
    store = MemoryStore(
        [
            item("please alpha beta", "m1"),
            item("gamma delta epsilon", "m2"),
            item("zeta", "m3"),
        ],
        tokenizer=TOKENIZER,
    )

    def assert_totals_match() -> None:
        assert store.active_token_count == sum(i.token_count for i in store.active_items())

    assert_totals_match()
    store.apply(OperationKind.COMPRESS, "m1", compressor=RuleBasedCompressor())
    assert_totals_match()
    store.apply(OperationKind.PRUNE, "m2")
    assert_totals_match()
    assert store.active_token_count == len("alpha beta") + len("zeta")


# --- protected content and leakage --------------------------------------------


def test_system_instructions_are_never_candidates() -> None:
    system_item = item(SYSTEM, "sys", memory_type=MemoryType.SYSTEM)

    assert valid_actions(system_item, RuleBasedCompressor(), TOKENIZER) == ()
    with pytest.raises(ValueError, match="protected"):
        MemoryStore().add(system_item)

    # The serializer owns instructions and the question; they never enter the store.
    prompt = PromptSerializer(TOKENIZER, SYSTEM).serialize((), QUESTION)
    assert SYSTEM in prompt.text
    assert QUESTION in prompt.text
    assert prompt.included_ids == ()


def test_memory_item_rejects_simulator_only_metadata() -> None:
    with pytest.raises(ValueError, match="simulator-only"):
        MemoryItem.create(
            memory_id="m1",
            text="Alice is critical here.",
            tokenizer=TOKENIZER,
            metadata={"is_critical": True},
        )


def test_action_mask_never_includes_unavailable_compression() -> None:
    store = MemoryStore([item("Plain fact with no filler.", "m1")], tokenizer=TOKENIZER)

    assert store.valid_actions("m1", RuleBasedCompressor()) == (
        OperationKind.KEEP,
        OperationKind.PRUNE,
    )
    assert store.valid_actions("m1", IdentityCompressor()) == (
        OperationKind.KEEP,
        OperationKind.PRUNE,
    )


def test_unknown_memory_id_raises() -> None:
    store = MemoryStore(tokenizer=TOKENIZER)
    with pytest.raises(KeyError, match="unknown memory_id"):
        store.get("missing")


# --- provenance ---------------------------------------------------------------


def test_provenance_logs_every_attempt_without_raw_pruned_text() -> None:
    store = MemoryStore(
        [
            item("Please keep the invoice INV-1.", "m1"),
            item("weather chatter", "m2"),
        ],
        tokenizer=TOKENIZER,
    )
    compressor = RuleBasedCompressor()

    store.apply(OperationKind.COMPRESS, "m1", compressor=compressor)
    store.apply(OperationKind.PRUNE, "m2")
    store.apply(OperationKind.PRUNE, "m2")  # rejected attempts are logged too

    log = store.provenance
    assert [record.operation for record in log] == [
        OperationKind.COMPRESS,
        OperationKind.PRUNE,
        OperationKind.PRUNE,
    ]
    assert [record.success for record in log] == [True, True, False]
    assert log[0].new_token_count < log[0].previous_token_count
    assert log[1].new_token_count == 0
    assert log[1].previous_token_count == len("weather chatter")
    # Raw text is deliberately absent from the log; only hashes are kept.
    assert "weather chatter" not in repr(log)


# --- Phase 1 exit criterion ---------------------------------------------------


def test_phase1_exit_criterion_end_to_end() -> None:
    """Load a history, run each operation, serialize, verify exact counts."""
    history = [
        item("Alice moved to Pune in March.", "m1", MemoryType.USER, 0),
        item("Tool result: Alice's account ID is ACCT-77.", "m2", MemoryType.TOOL_RESULT, 1),
        item("Please note the review is scheduled for Friday.", "m3", MemoryType.ASSISTANT, 2),
        item("Unrelated small talk about lunch.", "m4", MemoryType.USER, 3),
    ]
    store = MemoryStore(history, tokenizer=TOKENIZER)
    compressor = RuleBasedCompressor()

    # m1 has no removable filler, so COMPRESS is correctly masked out.
    assert store.valid_actions("m1", compressor) == (OperationKind.KEEP, OperationKind.PRUNE)
    assert store.valid_actions("m3", compressor) == (
        OperationKind.KEEP,
        OperationKind.COMPRESS,
        OperationKind.PRUNE,
    )

    keep = store.apply(OperationKind.KEEP, "m1")
    compress = store.apply(OperationKind.COMPRESS, "m3", compressor=compressor)
    prune = store.apply(OperationKind.PRUNE, "m4")
    assert keep.success and compress.success and prune.success

    prompt = PromptSerializer(TOKENIZER, SYSTEM).serialize(store.active_items(), QUESTION)

    expected_memory = (
        "[user] Alice moved to Pune in March.\n"
        "[tool_result] Tool result: Alice's account ID is ACCT-77.\n"
        "[assistant] note the review is scheduled for Friday."
    )
    expected_text = f"{SYSTEM}\n\nMEMORY:\n{expected_memory}\n\nTASK:\n{QUESTION}"

    assert prompt.text == expected_text
    assert prompt.memory_text == expected_memory
    assert prompt.included_ids == ("m1", "m2", "m3")
    assert prompt.memory_tokens == len(expected_memory)
    assert prompt.total_prompt_tokens == len(expected_text)
    assert prompt.fixed_prompt_tokens == len(f"{SYSTEM}\n\nMEMORY:\n\nTASK:\n{QUESTION}")

    # Formatting overhead is counted: memory block > raw item tokens.
    assert prompt.memory_tokens > sum(i.token_count for i in store.active_items())
    # Fixed overhead (instructions + headers + question) is counted in the total.
    assert prompt.total_prompt_tokens > prompt.memory_tokens

    # Pruned content stays archived but never reaches the prompt.
    assert store.get("m4").status is MemoryStatus.PRUNED
    assert "Unrelated small talk" not in prompt.text
    assert store.active_token_count == (
        len("Alice moved to Pune in March.")
        + len("Tool result: Alice's account ID is ACCT-77.")
        + len("note the review is scheduled for Friday.")
    )
