"""KeyHunt: a small deterministic tool environment (plan.md §7 Phase 3, item 4).

The agent must submit the access keys of K projects. Tools:

* ``search[project]`` — a long, noisy listing of records (the right record is
  marked "access credentials"); useless once that record has been opened.
* ``open[REC-id]`` — the record text. The credentials record holds the key,
  either on its own ``access key:`` line or inside a sentence (so a line
  compressor may or may not keep it).
* ``submit[K-1111, K-2222, ...]`` — ends the episode; success iff the submitted
  keys equal the required keys (order-insensitive).

Early observations (the opened credential records) are needed at the very
end; search listings are distractors once used. Observation lengths vary.
Everything is a pure function of the task id, so every method sees the same
tasks. Gold keys live only in ``KeyHuntTask`` and never reach observations
except through the record the agent itself opens.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from memory_rl.envs.base import AgentObservation, SplitName

PROJECTS = (
    "alpha", "beacon", "cobalt", "delta", "ember", "falcon", "garnet", "harbor",
    "indigo", "juniper", "kestrel", "lumen", "meridian", "nimbus", "onyx", "pylon",
    "quartz", "raven", "sierra", "tundra",
)  # fmt: skip

_TOPICS = (
    "budget report", "meeting minutes", "vendor contract", "incident review",
    "roadmap draft", "hiring plan", "security audit", "release notes",
    "design review", "travel policy",
)  # fmt: skip

_FILLER = (
    "The document was last revised by the operations team after the quarterly sync.",
    "Several reviewers left comments that were resolved in a later revision.",
    "It is cross-referenced from the shared drive index and the team wiki.",
    "Formatting follows the standard template, with appendices attached separately.",
    "Distribution is limited to the project group and its sponsors.",
    "An earlier draft of this file was archived and is no longer maintained.",
    "The summary section was rewritten to be shorter and easier to scan.",
    "Owners are asked to confirm the contents during the monthly review.",
)

SPLIT_SIZES: dict[str, int] = {"train": 400, "valid": 60, "test": 100}

SYSTEM_PROMPT = (
    "You are an agent that solves tasks by calling tools. Each turn, reply with one "
    "line 'Thought: <short reasoning>' and one line 'Action: <tool>[<argument>]'. "
    "Tools: search[<project>] lists records for a project; open[<record id>] shows a "
    "record; submit[<key>, <key>, ...] submits the final answer and ends the task. "
    "Use only information you have seen in this conversation."
)

#: Few-shot example projects are disjoint from task projects (no key confusion).
EXAMPLE_PROJECTS = ("zephyr", "wren", "vortex", "xenon")
EXAMPLE_TASK_ID = "keyhunt-example-0"


@dataclass(frozen=True, slots=True)
class _Record:
    record_id: str
    project: str
    topic: str
    text: str


@dataclass(frozen=True, slots=True)
class KeyHuntTask:
    """Simulator-side task. ``keys`` and ``credential_ids`` are gold labels."""

    task_id: str
    projects: tuple[str, ...]
    keys: tuple[str, ...]
    credential_ids: tuple[str, ...]
    records: tuple[_Record, ...]
    listings: tuple[tuple[str, str], ...]  # (project, listing text)

    @property
    def instruction(self) -> str:
        return f"Task: Submit the access keys for projects: {', '.join(self.projects)}."


def _rng_for(task_id: str) -> random.Random:
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _record_id(rng: random.Random, used: set[str]) -> str:
    while True:
        rid = f"REC-{rng.randint(1000, 9999)}"
        if rid not in used:
            used.add(rid)
            return rid


def _prose(rng: random.Random, sentences: int) -> str:
    return " ".join(rng.choice(_FILLER) for _ in range(sentences))


def generate_task(
    task_id: str,
    *,
    min_keys: int = 3,
    max_keys: int = 6,
    pool: Sequence[str] = PROJECTS,
) -> KeyHuntTask:
    rng = _rng_for(task_id)
    k = rng.randint(min_keys, max_keys)
    projects = tuple(rng.sample(list(pool), k))
    others = [p for p in pool if p not in projects]
    used_ids: set[str] = set()
    used_keys: set[str] = set()
    records: list[_Record] = []
    keys: list[str] = []
    credential_ids: list[str] = []
    listings: list[tuple[str, str]] = []

    for project in projects:
        while True:
            key = f"K-{rng.randint(1000, 9999)}"
            if key not in used_keys:
                used_keys.add(key)
                break
        keys.append(key)
        cred_id = _record_id(rng, used_ids)
        credential_ids.append(cred_id)
        header = f"Record {cred_id} (project {project}, access credentials)"
        if rng.random() < 0.5:
            body = f"access key: {key}\n{_prose(rng, rng.randint(6, 11))}"
        else:
            body = (
                f"{_prose(rng, rng.randint(3, 6))} For this project the current access key "
                f"is {key}, rotated at the last audit. {_prose(rng, rng.randint(3, 6))}"
            )
        records.append(_Record(cred_id, project, "access credentials", f"{header}\n{body}"))

        lines = [f"- {cred_id} | {project} | access credentials | restricted"]
        for _ in range(rng.randint(6, 10)):
            rid = _record_id(rng, used_ids)
            topic = rng.choice(_TOPICS)
            text = f"Record {rid} (project {project}, {topic})\n{_prose(rng, rng.randint(6, 12))}"
            records.append(_Record(rid, project, topic, text))
            lines.append(f"- {rid} | {project} | {topic} | {_prose(rng, 2).lower()}")
        for _ in range(rng.randint(4, 7)):  # cross-listed noise from other projects
            rid = _record_id(rng, used_ids)
            other = rng.choice(others)
            topic = rng.choice(_TOPICS)
            text = f"Record {rid} (project {other}, {topic})\n{_prose(rng, rng.randint(2, 5))}"
            records.append(_Record(rid, other, topic, text))
            note = _prose(rng, 1).lower()
            lines.append(f"- {rid} | {other} | {topic} | see also {project}. {note}")
        rng.shuffle(lines)
        listing = f"Records for {project}:\n" + "\n".join(lines)
        listings.append((project, listing))

    return KeyHuntTask(
        task_id=task_id,
        projects=projects,
        keys=tuple(keys),
        credential_ids=tuple(credential_ids),
        records=tuple(records),
        listings=tuple(listings),
    )


def key_note(record_id: str, project: str, key: str) -> str:
    """How the agent writes a key it has read into its own thought (scratchpad)."""
    return f"{record_id} shows the key for {project}: {key}."


def render_example(task: KeyHuntTask) -> str:
    """A complete, correct trajectory for ``task`` in the ReAct format."""
    lines = [f"Example (a different task):\n{task.instruction}"]
    note = ""
    for project, cred_id in zip(task.projects, task.credential_ids, strict=True):
        listing = next(text for p, text in task.listings if p == project)
        record = next(r for r in task.records if r.record_id == cred_id)
        lines += [
            f"Thought: {note}I need the key for {project}. I will search for its records.",
            f"Action: search[{project}]",
            f"Observation: {listing}",
            f"Thought: {cred_id} holds the credentials for {project}. I will open it.",
            f"Action: open[{cred_id}]",
            f"Observation: {record.text}",
        ]
        key = task.keys[task.projects.index(project)]
        note = key_note(cred_id, project, key) + " "
    lines += [
        f"Thought: {note}I have every key I need. I will submit.",
        f"Action: submit[{', '.join(task.keys)}]",
        "Observation: Submission accepted.",
    ]
    return "\n".join(lines)


_ACTION = re.compile(r"Action:\s*(search|open|submit)\[(.*?)\]", re.IGNORECASE | re.DOTALL)


def parse_action(text: str) -> tuple[str, str] | None:
    """Last ``Action: tool[arg]`` in the LLM output, or None if absent."""
    matches = _ACTION.findall(text)
    if not matches:
        return None
    tool, arg = matches[-1]
    return tool.lower(), arg.strip()


class KeyHuntEnv:
    """Deterministic toy tool environment implementing ``AgentEnv``."""

    name = "toy-keyhunt"

    def __init__(self, *, min_keys: int = 3, max_keys: int = 6, step_slack: int = 4) -> None:
        if not 1 <= min_keys <= max_keys:
            raise ValueError("need 1 <= min_keys <= max_keys")
        self.min_keys = min_keys
        self.max_keys = max_keys
        self.step_slack = step_slack
        self._task: KeyHuntTask | None = None
        self._few_shot: str | None = None
        self._steps = 0
        self._done = False
        self.max_steps = 2 * max_keys + 1 + step_slack

    def task_ids(self, split: SplitName) -> Sequence[str]:
        if split not in SPLIT_SIZES:
            raise ValueError(f"unknown split {split!r}")
        return tuple(f"keyhunt-{split}-{i:04d}" for i in range(SPLIT_SIZES[split]))

    def system_prompt(self) -> str:
        return SYSTEM_PROMPT

    def few_shot(self) -> str:
        if self._few_shot is None:
            example = generate_task(EXAMPLE_TASK_ID, min_keys=2, max_keys=2, pool=EXAMPLE_PROJECTS)
            self._few_shot = render_example(example)
        return self._few_shot

    @property
    def task(self) -> KeyHuntTask:
        if self._task is None:
            raise RuntimeError("call reset() first")
        return self._task

    def reset(self, task_id: str, seed: int) -> AgentObservation:
        del seed  # tasks are a pure function of task_id
        self._task = generate_task(task_id, min_keys=self.min_keys, max_keys=self.max_keys)
        self._steps = 0
        self._done = False
        self.max_steps = 2 * len(self._task.projects) + 1 + self.step_slack
        return AgentObservation(text="", instruction=self._task.instruction)

    def step(self, action_text: str) -> tuple[AgentObservation, float, bool, dict[str, Any]]:
        task = self.task
        if self._done:
            raise RuntimeError("episode already finished")
        self._steps += 1
        parsed = parse_action(action_text)
        info: dict[str, Any] = {"valid_action": parsed is not None}
        reward, done = 0.0, False
        if parsed is None:
            text = "Invalid action. Reply with 'Action: search[...]', open[...] or submit[...]."
        else:
            tool, arg = parsed
            info["tool"] = tool
            if tool == "search":
                listing = next((t for p, t in task.listings if p == arg.lower()), None)
                text = listing or f"No records found for {arg!r}."
            elif tool == "open":
                record = next((r for r in task.records if r.record_id == arg.upper()), None)
                text = record.text if record else f"Record {arg!r} does not exist."
            else:
                submitted = {k.strip().upper() for k in arg.split(",") if k.strip()}
                success = submitted == set(task.keys)
                reward, done = (1.0 if success else 0.0), True
                info["success"] = success
                text = "Submission accepted." if success else "Submission rejected."
        if not done and self._steps >= self.max_steps:
            done = True
            info["step_limit"] = True
        self._done = done
        return AgentObservation(text=text), reward, done, info
