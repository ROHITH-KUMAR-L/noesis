"""One tokenizer interface for every method (plan.md §6.2).

No silent fallbacks: an unknown spec or an unavailable backend raises
``TokenizerError``/``ValueError`` instead of degrading to characters or words.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol

DEFAULT_TOKENIZER_SPEC = "tiktoken:cl100k_base"


class TokenizerError(RuntimeError):
    """Raised when a configured tokenizer cannot be loaded or used."""


class Tokenizer(Protocol):
    """Minimal contract every method must share for comparable token counts."""

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    def count(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class TestTokenizer:
    """Deterministic test-only tokenizer: one token per Unicode code point.

    It is explicitly named ``test:char`` so experiment metadata shows it was a
    test tokenizer, never a stand-in for a model tokenizer.
    """

    name: str = "test:char"
    version: str = "1"
    is_test: bool = True
    # Keep pytest from trying to collect this class as a test suite.
    __test__: ClassVar[bool] = False

    def count(self, text: str) -> int:
        return len(text)


def _load_tiktoken(encoding_name: str) -> Any:
    """Load a tiktoken encoding or fail loudly (no character/word fallback)."""
    try:
        import tiktoken
    except ImportError as exc:
        raise TokenizerError(
            "tiktoken is not installed; install it or select an explicit test tokenizer"
        ) from exc
    try:
        return tiktoken.get_encoding(encoding_name)
    except Exception as exc:  # noqa: BLE001 - surfaced as a tokenizer failure
        raise TokenizerError(f"cannot load tiktoken encoding {encoding_name!r}: {exc}") from exc


class TiktokenTokenizer:
    """Tokenizer matching the answer model where possible; loaded lazily."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding_name = encoding_name
        self._encoding: Any | None = None

    @property
    def name(self) -> str:
        return f"tiktoken:{self._encoding_name}"

    @property
    def version(self) -> str:
        try:
            import tiktoken
        except ImportError as exc:
            raise TokenizerError("tiktoken is not installed") from exc
        return str(getattr(tiktoken, "__version__", "unknown"))

    def _encoding_or_raise(self) -> Any:
        if self._encoding is None:
            self._encoding = _load_tiktoken(self._encoding_name)
        return self._encoding

    def count(self, text: str) -> int:
        return len(self._encoding_or_raise().encode(text, disallowed_special=()))


def build_tokenizer(spec: str = DEFAULT_TOKENIZER_SPEC) -> Tokenizer:
    """Resolve an explicit tokenizer spec; never falls back to another tokenizer."""
    if spec == "test:char":
        return TestTokenizer()
    if spec.startswith("tiktoken:"):
        encoding_name = spec.split(":", 1)[1]
        if not encoding_name:
            raise ValueError(
                "tiktoken spec must include an encoding name, e.g. tiktoken:cl100k_base"
            )
        return TiktokenTokenizer(encoding_name)
    raise ValueError(
        f"unknown tokenizer spec {spec!r}; expected 'test:char' or 'tiktoken:<encoding>'"
    )
