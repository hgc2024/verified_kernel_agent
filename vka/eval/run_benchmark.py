"""Run an agent (or the baselines) over a task directory, writing ``results/runs/*.jsonl``."""

from __future__ import annotations

import argparse
from pathlib import Path


def run_benchmark(agent: str, tasks_dir: Path, attempts: int) -> Path:
    """Returns the path of the JSONL file written."""
    # TODO(phase-6): iterate tasks, run the agent, record one line per attempt.
    raise NotImplementedError("TODO(phase-6): run_benchmark")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True, help="agent config name, or 'baseline'")
    parser.add_argument("--tasks", type=Path, default=Path("tasks/level1"))
    parser.add_argument("--attempts", type=int, default=5)
    args = parser.parse_args(argv)
    run_benchmark(args.agent, args.tasks, args.attempts)


if __name__ == "__main__":
    main()
