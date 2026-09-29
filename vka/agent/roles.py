"""Prompt templates for the planner, coder, and reviewer roles (README: Agent > Roles and loop)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class Role(StrEnum):
    # Reads the task, baseline, and last profile; proposes an optimization plan.
    PLANNER = "planner"
    # Writes a full candidate .cu implementing the C ABI.
    CODER = "coder"
    # Explains the failure or bottleneck from harness JSON (+ ncu metrics); proposes a next step.
    REVIEWER = "reviewer"


def render_prompt(role: Role, context: dict[str, Any]) -> list[dict[str, str]]:
    """Return chat messages for ``role`` given the task and attempt-history context."""
    # TODO(phase-5): write the templates.
    raise NotImplementedError("TODO(phase-5): render_prompt")
