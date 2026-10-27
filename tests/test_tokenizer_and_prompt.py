"""Phase 1 required tests: tokenizer and prompt serialization (plan.md §6.2)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from memory_rl.memory import (
    MemoryItem,
    MemoryStatus,
    MemoryType,
    PromptSerializer,
    TestTokenizer,
    TiktokenTokenizer,
    TokenizerError,
    build_tokenizer,
)
from memory_rl.memory import tokenizer as tokenizer_module

TOKENIZER = TestTokenizer()
SYSTEM = "Use memory only."


def _items() -> list[MemoryItem]:
    return [
        MemoryItem.create(
            memory_id="m1", text="Alice lives in Pune.", tokenizer=TOKENIZER, turn_index=0
        ),
        MemoryItem.create(
            memory_id="m2",
            text="Ticket 42 is closed.",
            tokenizer=TOKENIZER,
            turn_index=1,
            memory_type=MemoryType.TOOL_RESULT,
        ),
    ]


# --- tokenizer ----------------------------------------------------------------


def test_test_tokenizer_is_deterministic_for_unicode_and_empty() -> None:
    samples = ["", "a", "héllo wörld", "日本語のテキスト", "3.14 → -2", "  \n\t "]
    for text in samples:
        assert TOKENIZER.count(text) == TOKENIZER.count(text) == len(text)
    assert TOKENIZER.count("") == 0
    assert TOKENIZER.count("日本語") == 3


def test_build_tokenizer_requires_an_explicit_known_spec() -> None:
    assert isinstance(build_tokenizer("test:char"), TestTokenizer)
    assert isinstance(build_tokenizer("tiktoken:cl100k_base"), TiktokenTokenizer)

    with pytest.raises(ValueError, match="unknown tokenizer spec"):
        build_tokenizer("wordcount")
    with pytest.raises(ValueError, match="encoding name"):
        build_tokenizer("tiktoken:")


def test_tiktoken_fails_explicitly_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def unavailable(encoding_name: str) -> object:
        raise TokenizerError(f"cannot load {encoding_name}")

    monkeypatch.setattr(tokenizer_module, "_load_tiktoken", unavailable)

    with pytest.raises(TokenizerError):
        TiktokenTokenizer().count("hello")


def test_tiktoken_counts_are_deterministic_including_unicode() -> None:
    tokenizer = TiktokenTokenizer()
    text = "Invoice INV-2049 is due 2026-03-01; ünïcode ✓"

    assert tokenizer.count(text) == tokenizer.count(text) > 0
    assert tokenizer.count("") == 0
    assert tokenizer.name == "tiktoken:cl100k_base"


# --- prompt serializer --------------------------------------------------------


def test_serializer_counts_fixed_memory_and_total_exactly() -> None:
    serializer = PromptSerializer(TOKENIZER, SYSTEM)
    prompt = serializer.serialize(_items(), "Where does Alice live?")

    assert prompt.memory_text == "[user] Alice lives in Pune.\n[tool_result] Ticket 42 is closed."
    assert prompt.text == (
        f"{SYSTEM}\n\nMEMORY:\n{prompt.memory_text}\n\nTASK:\nWhere does Alice live?"
    )
    assert prompt.memory_tokens == len(prompt.memory_text)
    assert prompt.fixed_prompt_tokens == len(prompt.fixed_text)
    assert prompt.total_prompt_tokens == len(prompt.text)
    # Prompt formatting costs tokens beyond the raw item text.
    assert prompt.memory_tokens > sum(item.token_count for item in _items())


def test_serializer_without_memory_reports_zero_memory_tokens() -> None:
    serializer = PromptSerializer(TOKENIZER, SYSTEM)

    prompt = serializer.serialize((), "Any question?")

    assert prompt.memory_text == ""
    assert prompt.memory_tokens == 0
    assert prompt.total_prompt_tokens == prompt.fixed_prompt_tokens
    assert prompt.included_ids == ()


def test_serializer_rejects_pruned_items() -> None:
    pruned = replace(_items()[0], status=MemoryStatus.PRUNED)

    with pytest.raises(ValueError, match="pruned"):
        PromptSerializer(TOKENIZER, SYSTEM).serialize([pruned], "Q?")


def test_serializer_rejects_duplicates_and_empty_questions() -> None:
    serializer = PromptSerializer(TOKENIZER, SYSTEM)
    items = _items()

    with pytest.raises(ValueError, match="duplicate"):
        serializer.serialize([items[0], items[0]], "Q?")
    with pytest.raises(ValueError, match="question"):
        serializer.serialize(items, "   ")


def test_serializer_is_deterministic_and_order_preserving() -> None:
    serializer = PromptSerializer(TOKENIZER, SYSTEM)

    first = serializer.serialize(_items(), "Q?")
    second = serializer.serialize(_items(), "Q?")

    assert first == second
    assert first.included_ids == ("m1", "m2")

    reversed_prompt = serializer.serialize(list(reversed(_items())), "Q?")
    assert reversed_prompt.included_ids == ("m2", "m1")
    assert reversed_prompt.text != first.text


def test_serializer_requires_instructions_and_template_placeholders() -> None:
    with pytest.raises(ValueError, match="system_instructions"):
        PromptSerializer(TOKENIZER, "   ")
    with pytest.raises(ValueError, match="item_template"):
        PromptSerializer(TOKENIZER, SYSTEM, item_template="[{text}]")


def test_serializer_counts_are_stable_with_a_model_tokenizer() -> None:
    serializer = PromptSerializer(TiktokenTokenizer(), SYSTEM)

    prompt = serializer.serialize(_items(), "Where does Alice live?")
    again = serializer.serialize(_items(), "Where does Alice live?")

    assert prompt == again
    assert prompt.total_prompt_tokens == serializer.tokenizer.count(prompt.text)
    assert prompt.memory_tokens == serializer.tokenizer.count(prompt.memory_text)
    assert prompt.total_prompt_tokens > prompt.memory_tokens > 0
