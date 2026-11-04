"""Reward components (plan.md §7 Phase 2/3).

Phase 2 needs finite rewards to verify the environment; the weighted research
reward — token cost, critical-loss, budget-violation, and operation-cost terms —
is Phase 3 work and is deliberately **not** implemented here yet. Every component
is logged separately so reward hacking stays diagnosable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewardBreakdown:
    """Separately logged reward terms; ``total`` is what the agent receives."""

    task_success: float
    token_cost: float
    required_information_loss: float
    budget_violation: float
    operation_cost: float

    @property
    def total(self) -> float:
        return (
            self.task_success
            - self.token_cost
            - self.required_information_loss
            - self.budget_violation
            - self.operation_cost
        )


class Phase2SuccessReward:
    """Placeholder for Phase 2: terminal task success only, all penalties zero.

    Phase 3 replaces this with the configurable weighted reward and re-runs the
    environment verification; do not report Phase 2 returns as research results.
    """

    name = "phase2-success-only"

    def compute(
        self,
        *,
        correct: bool,
        memory_tokens: int,
        budget_tokens: int,
        critical_facts_lost: int,
        budget_compliant: bool,
    ) -> RewardBreakdown:
        return RewardBreakdown(
            task_success=1.0 if correct else 0.0,
            token_cost=0.0,
            required_information_loss=0.0,
            budget_violation=0.0,
            operation_cost=0.0,
        )
