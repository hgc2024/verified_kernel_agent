"""planner -> coder -> kharness -> reviewer loop, repeated until the budget is exhausted.

Every attempt is written to ``results/trajectories/*.jsonl`` for later analysis or fine-tuning.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Mode(StrEnum):
    ONE_SHOT = "one_shot"
    ITERATIVE = "iterative"


@dataclass
class Budget:
    max_attempts: int = 5
    max_tokens: int | None = None
    max_wall_time_s: float | None = None


@dataclass
class AgentConfig:
    """Loop configuration, including the ablation switches listed in README: Agent > Ablations."""

    budget: Budget = field(default_factory=Budget)
    mode: Mode = Mode.ITERATIVE
    use_profiler_feedback: bool = True
    multi_agent: bool = True  # False: a single agent plays all roles.


def run_agent(task_id: str, config: AgentConfig) -> list[dict[str, Any]]:
    """Run the loop on one task; returns the per-attempt trajectory records."""
    # TODO(phase-5): implement the loop.
    raise NotImplementedError("TODO(phase-5): run_agent")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="task id, e.g. level1.gemm_fp32")
    parser.add_argument("--attempts", type=int, default=Budget.max_attempts)
    args = parser.parse_args(argv)
    run_agent(args.task, AgentConfig(budget=Budget(max_attempts=args.attempts)))


if __name__ == "__main__":
    main()
