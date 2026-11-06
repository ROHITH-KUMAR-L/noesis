"""Frozen-LLM backends (plan.md §7 Phase 3, item 1).

``OpenAICompatibleBackend`` talks to any ``/chat/completions`` endpoint (OpenAI,
vLLM, Ollama, OpenRouter, ...). Credentials come only from the environment or a
local ``.env`` file (gitignored):

    NOESIS_LLM_BASE_URL=https://api.example.com/v1
    NOESIS_LLM_API_KEY=sk-...
    NOESIS_LLM_MODEL=model-name
    NOESIS_LLM_MAX_CALLS=200          # hard spending cap per process

Decoding is temperature 0. Paid calls require an explicit opt-in by the caller
(``--allow-paid`` on the CLIs); the call cap is enforced here.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from memory_rl.envs.agent_prompt import ChatMessage

DEFAULT_DOTENV = Path(__file__).resolve().parents[3] / ".env"


class LLMError(RuntimeError):
    pass


class LLMCallCapExceeded(LLMError):
    pass


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    latency_s: float
    output_tokens: int | None = None  # provider-reported, if any
    usage: dict[str, Any] = field(default_factory=dict)  # raw provider usage


class LLMBackend(Protocol):
    @property
    def name(self) -> str: ...

    def generate(self, messages: Sequence[ChatMessage], *, max_tokens: int) -> LLMResponse: ...


def load_dotenv(path: str | Path = DEFAULT_DOTENV) -> dict[str, str]:
    """Minimal ``KEY=VALUE`` parser. Existing environment variables win."""
    path = Path(path)
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
        os.environ.setdefault(key, value)
    return values


@dataclass(frozen=True, slots=True)
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    max_calls: int = 200
    timeout_s: float = 60.0
    max_retries: int = 2

    @classmethod
    def from_env(cls, dotenv: str | Path | None = DEFAULT_DOTENV) -> LLMConfig:
        if dotenv is not None:
            load_dotenv(dotenv)
        missing = [
            name
            for name in ("NOESIS_LLM_BASE_URL", "NOESIS_LLM_API_KEY", "NOESIS_LLM_MODEL")
            if not os.environ.get(name)
        ]
        if missing:
            raise LLMError(
                f"missing {missing}; copy .env.example to .env and fill in the endpoint"
            )
        return cls(
            base_url=os.environ["NOESIS_LLM_BASE_URL"].rstrip("/"),
            api_key=os.environ["NOESIS_LLM_API_KEY"],
            model=os.environ["NOESIS_LLM_MODEL"],
            max_calls=int(os.environ.get("NOESIS_LLM_MAX_CALLS", "200")),
        )

    def redacted(self) -> dict[str, Any]:
        return {"base_url": self.base_url, "model": self.model, "max_calls": self.max_calls}


class OpenAICompatibleBackend:
    """Blocking chat-completions client with retries and a call cap."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self.calls = 0

    @property
    def name(self) -> str:
        return f"openai-compatible:{self.config.model}"

    def generate(self, messages: Sequence[ChatMessage], *, max_tokens: int) -> LLMResponse:
        if self.calls >= self.config.max_calls:
            raise LLMCallCapExceeded(
                f"call cap reached ({self.config.max_calls}); raise NOESIS_LLM_MAX_CALLS"
            )
        payload = {
            "model": self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            f"{self.config.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
                "User-Agent": "noesis-memory-rl/0.1",
            },
            method="POST",
        )
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            self.calls += 1
            start = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_s) as response:
                    body = json.loads(response.read().decode("utf-8"))
                latency = time.monotonic() - start
                return self._parse(body, latency)
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code < 500 and exc.code != 429:
                    detail = exc.read().decode("utf-8", "replace")[:500]
                    raise LLMError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = exc
            if attempt < self.config.max_retries:
                time.sleep(2**attempt)
        raise LLMError(f"LLM request failed after retries: {last_error}")

    @staticmethod
    def _parse(body: dict[str, Any], latency: float) -> LLMResponse:
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"unexpected response shape: {str(body)[:300]}") from exc
        usage = body.get("usage") or {}
        output = usage.get("completion_tokens")
        return LLMResponse(
            text=text,
            latency_s=latency,
            output_tokens=int(output) if output is not None else None,
            usage=usage,
        )
