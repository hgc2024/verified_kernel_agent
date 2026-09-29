"""Agent tools: thin wrappers around the ``kharness`` CLI.

The agent never executes arbitrary shell. Every tool call becomes a fixed ``kharness``
subcommand whose single JSON object on stdout is returned as a dict.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_KHARNESS_BIN = "harness/target/release/kharness"


def kharness_bin() -> Path:
    return Path(os.environ.get("VKA_KHARNESS_BIN", DEFAULT_KHARNESS_BIN))


def read_task(task_path: Path) -> dict[str, Any]:
    # TODO(phase-5): load the task TOML for the planner.
    raise NotImplementedError("TODO(phase-5): read_task")


def compile_candidate(task_path: Path, candidate: Path) -> dict[str, Any]:
    # TODO(phase-5): `kharness compile`.
    raise NotImplementedError("TODO(phase-5): compile_candidate")


def verify_candidate(task_path: Path, candidate: Path, trials: int = 20) -> dict[str, Any]:
    # TODO(phase-5): `kharness verify`.
    raise NotImplementedError("TODO(phase-5): verify_candidate")


def profile_candidate(task_path: Path, candidate: Path) -> dict[str, Any]:
    # TODO(phase-5): `kharness profile`.
    raise NotImplementedError("TODO(phase-5): profile_candidate")


def run_candidate(task_path: Path, candidate: Path) -> dict[str, Any]:
    # TODO(phase-5): `kharness run`.
    raise NotImplementedError("TODO(phase-5): run_candidate")
