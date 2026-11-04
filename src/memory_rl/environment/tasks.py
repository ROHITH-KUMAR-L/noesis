"""Synthetic tasks with known ground truth (plan.md §7 Phase 2).

Every template belongs to exactly one split, so validation/test examples come
from templates that training never sees (split-by-template, not split-by-row).

Simulator-only labels — the reference answer and the required-fact substrings —
live in ``TaskLabels``. They are never serialized into a prompt and never reach
an observation.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from memory_rl.memory import MemoryType


class TaskFamily(StrEnum):
    FACT_RECALL = "fact_recall"
    MULTI_CONSTRAINT = "multi_constraint"
    MULTI_STEP_DEPENDENCY = "multi_step_dependency"
    DISTRACTOR_HEAVY = "distractor_heavy"


class Split(StrEnum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class HistoryTurn:
    """One message of the source history, before it becomes a memory item."""

    memory_id: str
    text: str
    memory_type: MemoryType
    turn_index: int


@dataclass(frozen=True, slots=True)
class RequiredFact:
    """Simulator-only: exact substring that must survive in active memory."""

    memory_id: str
    text: str


@dataclass(frozen=True, slots=True)
class TaskLabels:
    """Ground truth. Never an inference-time input."""

    reference_answer: str
    required_facts: tuple[RequiredFact, ...]

    @property
    def critical_item_ids(self) -> tuple[str, ...]:
        return tuple(fact.memory_id for fact in self.required_facts)


@dataclass(frozen=True, slots=True)
class SyntheticTask:
    task_id: str
    family: TaskFamily
    template_id: str
    seed: int
    history: tuple[HistoryTurn, ...]
    question: str
    labels: TaskLabels


# --- value pools (varied, never repeated across a task) -----------------------

_NAMES = ("Alice", "Bruno", "Chen", "Dara", "Elif", "Farid", "Gita", "Hugo", "Ines", "Jonas")
_DATES = ("2026-03-04", "2026-04-19", "2026-05-30", "2026-06-11", "2026-07-22")
_AMOUNTS = ("1,240 USD", "87.50 USD", "9,999 USD", "430 USD", "2,015.75 USD")
_IDS = ("INV-2049", "ACCT-77", "TCK-318", "ORD-5521", "REF-9080")
_TIMES = ("06:00", "09:30", "13:15", "17:45", "21:00")
_GATES = ("Gate 4", "Gate 7", "Gate 12", "Gate 2", "Gate 9")
_EMAILS = ("ops@example.com", "billing@example.org", "team@example.net")

_DISTRACTORS = (
    "The cafeteria serves lunch until 14:00.",
    "A team meeting moved to Thursday.",
    "The printer on floor 3 is offline.",
    "Parking permits renew next month.",
    "The design review notes are in the shared drive.",
    "Someone watered the plants in the lobby.",
    "The build pipeline passed on the last commit.",
    "Coffee supplies arrive on Tuesday.",
)


@dataclass(frozen=True, slots=True)
class _Entry:
    text: str
    memory_type: MemoryType
    required_text: str | None = None


@dataclass(frozen=True, slots=True)
class _Scenario:
    entries: tuple[_Entry, ...]
    question: str
    answer: str


def _distractors(rng: random.Random, count: int) -> list[_Entry]:
    return [_Entry(rng.choice(_DISTRACTORS), MemoryType.USER) for _ in range(count)]


def _interleave(rng: random.Random, facts: Sequence[_Entry], n_distractors: int) -> list[_Entry]:
    entries = _distractors(rng, n_distractors)
    for fact in facts:
        entries.insert(rng.randint(0, len(entries)), fact)
    return entries


def _scenario(entries: Sequence[_Entry], question: str, answer: str) -> _Scenario:
    return _Scenario(tuple(entries), question, answer)


# --- family: exact fact recall ------------------------------------------------


def _t_fact_name(rng: random.Random) -> _Scenario:
    name = rng.choice(_NAMES)
    fact = _Entry(f"My name is {name}.", MemoryType.USER, required_text=name)
    entries = _interleave(rng, [fact], rng.randint(2, 5))
    return _scenario(entries, "What is the user's name?", name)


def _t_fact_date(rng: random.Random) -> _Scenario:
    date = rng.choice(_DATES)
    fact = _Entry(f"The deployment window is {date}.", MemoryType.FACT, required_text=date)
    entries = _interleave(rng, [fact], rng.randint(2, 6))
    return _scenario(entries, "When is the deployment window?", date)


def _t_fact_amount(rng: random.Random) -> _Scenario:
    amount = rng.choice(_AMOUNTS)
    fact = _Entry(f"The invoice total is {amount}.", MemoryType.FACT, required_text=amount)
    entries = _interleave(rng, [fact], rng.randint(3, 6))
    return _scenario(entries, "What is the invoice total?", amount)


def _t_fact_ticket(rng: random.Random) -> _Scenario:
    ticket = rng.choice(_IDS)
    fact = _Entry(
        f"Ticket {ticket} is assigned to the team.", MemoryType.FACT, required_text=ticket
    )
    entries = _interleave(rng, [fact], rng.randint(2, 5))
    return _scenario(entries, "Which ticket is assigned to the team?", ticket)


# --- family: multiple simultaneous constraints --------------------------------


def _t_gate_time(rng: random.Random) -> _Scenario:
    gate = rng.choice(_GATES)
    time = rng.choice(_TIMES)
    facts = [
        _Entry(f"The maintenance gate is {gate}.", MemoryType.FACT, required_text=gate),
        _Entry(f"The maintenance starts at {time}.", MemoryType.FACT, required_text=time),
    ]
    entries = _interleave(rng, facts, rng.randint(2, 5))
    return _scenario(entries, "Which gate and what time does maintenance start?", f"{gate}, {time}")


def _t_customer_id(rng: random.Random) -> _Scenario:
    name = rng.choice(_NAMES)
    customer_id = rng.choice(_IDS)
    facts = [
        _Entry(f"The customer is {name}.", MemoryType.FACT, required_text=name),
        _Entry(f"The customer ID is {customer_id}.", MemoryType.FACT, required_text=customer_id),
    ]
    entries = _interleave(rng, facts, rng.randint(2, 5))
    return _scenario(entries, "Who is the customer and what is their ID?", f"{name}, {customer_id}")


def _t_window_owner(rng: random.Random) -> _Scenario:
    name = rng.choice(_NAMES)
    date = rng.choice(_DATES)
    facts = [
        _Entry(f"The maintenance window owner is {name}.", MemoryType.FACT, required_text=name),
        _Entry(f"The maintenance window is {date}.", MemoryType.FACT, required_text=date),
    ]
    entries = _interleave(rng, facts, rng.randint(3, 6))
    return _scenario(
        entries, "Who owns the maintenance window and when is it?", f"{name}, {date}"
    )


# --- family: multi-step dependency (a tool result is needed later) ------------


def _t_invoice_amount(rng: random.Random) -> _Scenario:
    invoice = rng.choice(_IDS)
    amount = rng.choice(_AMOUNTS)
    entries = _distractors(rng, rng.randint(1, 3))
    entries.append(_Entry(f"Tool: fetched invoice {invoice}.", MemoryType.TOOL_RESULT))
    entries.extend(_distractors(rng, rng.randint(1, 3)))
    entries.append(
        _Entry(f"Invoice {invoice} has amount {amount}.", MemoryType.FACT, required_text=amount)
    )
    return _scenario(entries, "What is the amount on the fetched invoice?", amount)


def _t_order_email(rng: random.Random) -> _Scenario:
    order = rng.choice(_IDS)
    email = rng.choice(_EMAILS)
    entries = _distractors(rng, rng.randint(1, 2))
    entries.append(_Entry(f"Tool: placed order {order}.", MemoryType.TOOL_RESULT))
    entries.extend(_distractors(rng, rng.randint(1, 3)))
    entries.append(
        _Entry(
            f"The contact email for {order} is {email}.",
            MemoryType.FACT,
            required_text=email,
        )
    )
    return _scenario(entries, "What is the contact email for the placed order?", email)


def _t_ticket_owner(rng: random.Random) -> _Scenario:
    ticket = rng.choice(_IDS)
    name = rng.choice(_NAMES)
    entries = _distractors(rng, rng.randint(1, 3))
    entries.append(_Entry(f"Tool: fetched ticket {ticket}.", MemoryType.TOOL_RESULT))
    entries.extend(_distractors(rng, rng.randint(1, 2)))
    entries.append(
        _Entry(f"Ticket {ticket} is owned by {name}.", MemoryType.FACT, required_text=name)
    )
    return _scenario(entries, "Who owns the fetched ticket?", name)


# --- family: distractor-heavy histories ---------------------------------------


def _distractor_task(rng: random.Random, position: str) -> _Scenario:
    code = str(rng.randint(1000, 9999))
    fact = _Entry(f"The vault code is {code}.", MemoryType.FACT, required_text=code)
    entries = _distractors(rng, rng.randint(8, 12))
    if position == "early":
        index = rng.randint(0, 1)
    elif position == "middle":
        index = len(entries) // 2
    else:
        index = len(entries)
    entries.insert(index, fact)
    return _scenario(entries, "What is the vault code?", code)


def _t_distractor_early(rng: random.Random) -> _Scenario:
    return _distractor_task(rng, "early")


def _t_distractor_middle(rng: random.Random) -> _Scenario:
    return _distractor_task(rng, "middle")


def _t_distractor_late(rng: random.Random) -> _Scenario:
    return _distractor_task(rng, "late")


@dataclass(frozen=True, slots=True)
class TaskTemplate:
    template_id: str
    family: TaskFamily
    split: Split
    build: Callable[[random.Random], _Scenario]


TEMPLATES: tuple[TaskTemplate, ...] = (
    TaskTemplate("fact_recall/name", TaskFamily.FACT_RECALL, Split.TRAIN, _t_fact_name),
    TaskTemplate("fact_recall/ticket", TaskFamily.FACT_RECALL, Split.TRAIN, _t_fact_ticket),
    TaskTemplate("fact_recall/date", TaskFamily.FACT_RECALL, Split.VALIDATION, _t_fact_date),
    TaskTemplate("fact_recall/amount", TaskFamily.FACT_RECALL, Split.TEST, _t_fact_amount),
    TaskTemplate(
        "multi_constraint/gate_time",
        TaskFamily.MULTI_CONSTRAINT,
        Split.TRAIN,
        _t_gate_time,
    ),
    TaskTemplate(
        "multi_constraint/customer_id",
        TaskFamily.MULTI_CONSTRAINT,
        Split.VALIDATION,
        _t_customer_id,
    ),
    TaskTemplate(
        "multi_constraint/window_owner",
        TaskFamily.MULTI_CONSTRAINT,
        Split.TEST,
        _t_window_owner,
    ),
    TaskTemplate(
        "dependency/invoice_amount",
        TaskFamily.MULTI_STEP_DEPENDENCY,
        Split.TRAIN,
        _t_invoice_amount,
    ),
    TaskTemplate(
        "dependency/order_email",
        TaskFamily.MULTI_STEP_DEPENDENCY,
        Split.VALIDATION,
        _t_order_email,
    ),
    TaskTemplate(
        "dependency/ticket_owner",
        TaskFamily.MULTI_STEP_DEPENDENCY,
        Split.TEST,
        _t_ticket_owner,
    ),
    TaskTemplate(
        "distractor/early", TaskFamily.DISTRACTOR_HEAVY, Split.TRAIN, _t_distractor_early
    ),
    TaskTemplate(
        "distractor/middle",
        TaskFamily.DISTRACTOR_HEAVY,
        Split.VALIDATION,
        _t_distractor_middle,
    ),
    TaskTemplate(
        "distractor/late", TaskFamily.DISTRACTOR_HEAVY, Split.TEST, _t_distractor_late
    ),
)


def templates_for(
    *, family: TaskFamily | None = None, split: Split | None = None
) -> tuple[TaskTemplate, ...]:
    """Templates matching the filters, in registry order."""
    return tuple(
        template
        for template in TEMPLATES
        if (family is None or template.family is family)
        and (split is None or template.split is split)
    )


def _build_task(template: TaskTemplate, seed: int, index: int) -> SyntheticTask:
    rng = random.Random(f"{template.template_id}:{seed}:{index}")
    scenario = template.build(rng)
    history = tuple(
        HistoryTurn(
            memory_id=f"m{position:02d}",
            text=entry.text,
            memory_type=entry.memory_type,
            turn_index=position,
        )
        for position, entry in enumerate(scenario.entries)
    )
    required = tuple(
        RequiredFact(memory_id=f"m{position:02d}", text=entry.required_text)
        for position, entry in enumerate(scenario.entries)
        if entry.required_text is not None
    )
    return SyntheticTask(
        task_id=f"{template.template_id}/{seed}/{index}",
        family=template.family,
        template_id=template.template_id,
        seed=seed,
        history=history,
        question=scenario.question,
        labels=TaskLabels(reference_answer=scenario.answer, required_facts=required),
    )


def generate_dataset(
    *,
    family: TaskFamily | None = None,
    split: Split = Split.TRAIN,
    count: int = 1,
    seed: int = 0,
) -> tuple[SyntheticTask, ...]:
    """Deterministically generate ``count`` tasks, cycling through templates.

    A template's split never changes, so regenerating the same (split, seed,
    count) always yields the same tasks.
    """
    if count < 0:
        raise ValueError("count must be >= 0")
    templates = templates_for(family=family, split=split)
    if not templates:
        raise ValueError(f"no templates for family={family} split={split}")
    return tuple(
        _build_task(templates[index % len(templates)], seed, index) for index in range(count)
    )
