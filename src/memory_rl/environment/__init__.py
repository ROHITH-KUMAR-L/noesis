"""Synthetic environment public API (Phase 2)."""

from memory_rl.environment.features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    empty_observation,
    extract_features,
    question_keywords,
)
from memory_rl.environment.memory_env import (
    ACTION_ORDER,
    DEFAULT_SYSTEM_INSTRUCTIONS,
    EnvConfig,
    MemoryEnv,
)
from memory_rl.environment.reward import Phase2SuccessReward, RewardBreakdown
from memory_rl.environment.tasks import (
    TEMPLATES,
    HistoryTurn,
    RequiredFact,
    Split,
    SyntheticTask,
    TaskFamily,
    TaskLabels,
    TaskTemplate,
    generate_dataset,
    templates_for,
)

__all__ = [
    "ACTION_ORDER",
    "DEFAULT_SYSTEM_INSTRUCTIONS",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "TEMPLATES",
    "EnvConfig",
    "HistoryTurn",
    "MemoryEnv",
    "Phase2SuccessReward",
    "RequiredFact",
    "RewardBreakdown",
    "Split",
    "SyntheticTask",
    "TaskFamily",
    "TaskLabels",
    "TaskTemplate",
    "empty_observation",
    "extract_features",
    "generate_dataset",
    "question_keywords",
    "templates_for",
]
