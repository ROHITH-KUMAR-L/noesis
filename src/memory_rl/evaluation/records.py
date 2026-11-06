"""Per-episode JSONL records (plan.md §6.8).

Prompts are never stored as text in records; only counts and hashes.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

RECORD_FIELDS: tuple[str, ...] = (
    "method", "variant", "knob", "env", "split", "task_id", "seed", "llm", "tokenizer",
    "pricing_profile", "success", "score", "turns", "llm_calls", "terminal_reason",
    "prompt_tokens_total", "cached_tokens_total", "written_tokens_total",
    "uncached_tokens_total", "output_tokens_total", "input_cost", "cache_read_cost",
    "cache_write_cost", "output_cost", "total_cost", "token_cost", "edits", "forced_edits",
    "actions_keep", "actions_compress", "actions_prune", "tokens_removed", "edit_turns",
    "latency_s", "budget", "budget_compliant", "git_commit", "config_hash",
)  # fmt: skip

_NUMERIC = (
    "turns", "llm_calls", "prompt_tokens_total", "cached_tokens_total", "written_tokens_total",
    "uncached_tokens_total", "output_tokens_total", "input_cost", "cache_read_cost",
    "cache_write_cost", "output_cost", "total_cost", "token_cost", "edits", "forced_edits",
    "actions_keep", "actions_compress", "actions_prune", "tokens_removed", "latency_s",
)  # fmt: skip


class RecordError(ValueError):
    pass


@lru_cache(maxsize=1)
def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def config_hash(config: Mapping[str, Any]) -> str:
    blob = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def validate_record(record: Mapping[str, Any]) -> None:
    missing = [key for key in RECORD_FIELDS if key not in record]
    if missing:
        raise RecordError(f"record is missing fields: {missing}")
    for key in _NUMERIC:
        value = record[key]
        if not isinstance(value, int | float) or isinstance(value, bool):
            raise RecordError(f"{key} must be numeric, got {value!r}")
        if not math.isfinite(value) or value < 0:
            raise RecordError(f"{key} must be finite and >= 0, got {value!r}")
    if not isinstance(record["success"], bool):
        raise RecordError("success must be a bool")
    if not isinstance(record["edit_turns"], list):
        raise RecordError("edit_turns must be a list")


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]], *, overwrite: bool) -> int:
    """Write records; refuses to overwrite an existing file unless asked."""
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass overwrite=True (--overwrite) to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            validate_record(record)
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
