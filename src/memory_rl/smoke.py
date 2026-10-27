"""Smoke checks: the fastest signal that this installation is usable.

Run with ``python -m memory_rl.smoke`` (or the ``noesis-smoke`` script). Exits
nonzero when a fatal check fails, so it works as a CI step and as the Docker
entrypoint. Dependency probes are informational in Phase 0: they become fatal
only when the phase that needs them lands.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from dataclasses import dataclass

#: Installed-dependency probes: (module, why it matters, fatal?).
_PROBES: tuple[tuple[str, str, bool], ...] = (
    ("tiktoken", "Phase 1 token accounting", True),
    ("numpy", "Phase 4 observation features", False),
    ("torch", "Phase 6 PPO training", False),
)


@dataclass(frozen=True)
class Check:
    """Result of a single smoke check."""

    name: str
    ok: bool
    detail: str
    fatal: bool = True


def _check_python() -> Check:
    version = ".".join(str(part) for part in sys.version_info[:3])
    ok = sys.version_info >= (3, 11)
    return Check(
        name="python>=3.11",
        ok=ok,
        detail=f"{version} ({sys.executable})",
        fatal=True,
    )


def _check_package() -> Check:
    module = importlib.import_module("memory_rl")
    version = getattr(module, "__version__", "unknown")
    location = getattr(module, "__file__", "<namespace>")
    return Check(name="import memory_rl", ok=True, detail=f"{version} at {location}")


def _check_probe(module_name: str, why: str, fatal: bool) -> Check:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        return Check(
            name=module_name,
            ok=False,
            detail=f"MISSING ({why}): {exc}",
            fatal=fatal,
        )
    version = getattr(module, "__version__", "unknown")
    return Check(name=module_name, ok=True, detail=f"{version} ({why})", fatal=fatal)


def run_checks() -> list[Check]:
    """Run every smoke check. Kept separate from I/O so tests can drive it."""
    checks = [_check_python(), _check_package()]
    for module_name, why, fatal in _PROBES:
        checks.append(_check_probe(module_name, why, fatal))
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="memory_rl.smoke", description=__doc__)
    parser.parse_args(list(argv) if argv is not None else None)

    checks = run_checks()
    width = max(len(check.name) for check in checks)
    failed = False

    for check in checks:
        status = "ok  " if check.ok else ("FAIL" if check.fatal else "warn")
        print(f"[{status}] {check.name:<{width}}  {check.detail}")
        if check.fatal and not check.ok:
            failed = True

    passed = sum(check.ok for check in checks)
    verdict = "FAILED" if failed else "PASSED"
    print(f"{verdict}: {passed}/{len(checks)} checks ok")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
