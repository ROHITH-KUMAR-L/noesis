"""Phase 0 smoke tests: the package must import and the smoke command must work."""

from __future__ import annotations

import subprocess
import sys

import pytest

from memory_rl import smoke


def test_package_imports_and_declares_version() -> None:
    import memory_rl

    assert memory_rl.__version__
    assert memory_rl.__file__ is not None
    assert "memory_rl" in memory_rl.__file__


def test_smoke_check_reports_python_and_package() -> None:
    checks = smoke.run_checks()
    names = {check.name for check in checks}

    assert "python>=3.11" in names
    assert "import memory_rl" in names
    # Fatal checks must never be silently marked non-fatal.
    fatal_names = {"python>=3.11", "import memory_rl"}
    assert all(check.fatal for check in checks if check.name in fatal_names)


def test_smoke_command_exits_zero_on_success() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "memory_rl.smoke"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASSED" in result.stdout


def test_smoke_command_exits_nonzero_when_a_fatal_check_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failing run must fail loudly (plan.md Phase 0 exit criterion)."""
    broken = smoke.Check(name="python>=3.11", ok=False, detail="simulated", fatal=True)
    monkeypatch.setattr(smoke, "_check_python", lambda: broken)

    assert smoke.main([]) == 1

    result = subprocess.run(
        [sys.executable, "-m", "memory_rl.smoke", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
