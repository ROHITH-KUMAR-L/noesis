"""Deterministic mock LLM for the KeyHunt toy env (tests, CI, training on CPU).

It is a scripted ReAct agent that reads only the messages it is sent — exactly
what a real frozen LLM would see. If a key it needs was pruned (or compressed
away), it cannot "remember" it and must re-search and re-open, paying for the
extra calls; with a step limit it may fail. That is what makes memory edits
measurable without a real model. Latency is simulated, not measured.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from memory_rl.envs.agent_prompt import ChatMessage
from memory_rl.envs.toy import key_note
from memory_rl.llm.backend import LLMResponse

_TASK = re.compile(r"Task: Submit the access keys for projects: ([^.\n]+)\.")
_KEY = re.compile(r"access key(?::| is) (K-\d{4})")
_NOTE = re.compile(r"shows the key for ([a-z]+): (K-\d{4})")
_RECORD = re.compile(r"Record (REC-\d{4}) \(project ([a-z]+), access credentials\)")


class ScriptedKeyHuntLLM:
    name = "mock:scripted-keyhunt-v1"

    def __init__(self, base_latency_s: float = 0.8, per_output_token_s: float = 0.02) -> None:
        self.base_latency_s = base_latency_s
        self.per_output_token_s = per_output_token_s
        self.calls = 0

    def generate(self, messages: Sequence[ChatMessage], *, max_tokens: int) -> LLMResponse:
        self.calls += 1
        text = self._act(messages)
        words = len(text.split())
        return LLMResponse(
            text=text, latency_s=self.base_latency_s + self.per_output_token_s * words * 1.3
        )

    @staticmethod
    def _act(messages: Sequence[ChatMessage]) -> str:
        if len(messages) < 2:
            return "Thought: I have no task.\nAction: submit[]"
        tasks = _TASK.findall(messages[1].content)
        if not tasks:
            return "Thought: I cannot find the task.\nAction: submit[]"
        projects = [p.strip() for p in tasks[-1].split(",") if p.strip()]
        history = [m.content for m in messages[2:]]

        noted: dict[str, str] = {}
        for message in messages[2:]:
            if message.role == "assistant":
                for project, key in _NOTE.findall(message.content):
                    noted[project] = key
        new_notes: list[str] = []
        keys = dict(noted)
        for message in messages[2:]:
            if message.role != "user":
                continue
            header = _RECORD.search(message.content)
            match = _KEY.search(message.content)
            if header and match and header.group(2) in projects:
                project = header.group(2)
                if project not in keys:
                    keys[project] = match.group(1)
                    new_notes.append(key_note(header.group(1), project, match.group(1)))
        note = " ".join(new_notes) + " " if new_notes else ""

        pending = [p for p in projects if p not in keys]
        if not pending:
            found = ", ".join(keys[p] for p in projects)
            return (
                f"Thought: {note}I have every key I need. I will submit.\n"
                f"Action: submit[{found}]"
            )

        project = pending[0]
        listing = re.compile(rf"(REC-\d{{4}}) \| {re.escape(project)} \| access credentials")
        for content in reversed(history):
            match = listing.search(content)
            if match:
                record = match.group(1)
                return (
                    f"Thought: {note}{record} holds the credentials for {project}. "
                    "I will open it.\n"
                    f"Action: open[{record}]"
                )
        return (
            f"Thought: {note}I need the key for {project}. I will search for its records.\n"
            f"Action: search[{project}]"
        )
