"""Unimplemented stages fail loudly instead of returning fake data."""

from __future__ import annotations

from pathlib import Path

import pytest

from vka import reference, reward
from vka.agent import loop, memory, providers, roles, tools
from vka.eval import make_tables, metrics, run_benchmark

TASK = Path("tasks/level1/gemm_fp32.toml")
CAND = Path("cand.cu")

STUBS = {
    "reward": lambda: reward.reward(
        compiled=True,
        verified=True,
        all_guards_passed=True,
        baseline_median_ms=1.0,
        candidate_median_ms=1.0,
    ),
    "reference.reservoir_step": lambda: reference.reservoir_step(),
    "tools.read_task": lambda: tools.read_task(TASK),
    "tools.compile_candidate": lambda: tools.compile_candidate(TASK, CAND),
    "tools.verify_candidate": lambda: tools.verify_candidate(TASK, CAND),
    "tools.profile_candidate": lambda: tools.profile_candidate(TASK, CAND),
    "tools.run_candidate": lambda: tools.run_candidate(TASK, CAND),
    "roles.render_prompt": lambda: roles.render_prompt(roles.Role.PLANNER, {}),
    "memory.append": lambda: memory.AttemptMemory(Path("m.jsonl")).append(
        memory.Attempt(task_id="t", index=0, plan="", code_sha256="")
    ),
    "memory.load": lambda: memory.AttemptMemory(Path("m.jsonl")).load(),
    "memory.summary": lambda: memory.AttemptMemory(Path("m.jsonl")).summary(),
    "providers.complete": lambda: providers.ChatClient(
        providers.ProviderConfig(base_url="http://localhost", model="m")
    ).complete([]),
    "loop.run_agent": lambda: loop.run_agent("level1.gemm_fp32", loop.AgentConfig()),
    "metrics.pass_at_k": lambda: metrics.pass_at_k([], 1),
    "metrics.fast_p": lambda: metrics.fast_p([], 1.0),
    "metrics.cost_per_success": lambda: metrics.cost_per_success([]),
    "metrics.guard_trigger_rates": lambda: metrics.guard_trigger_rates([]),
    "run_benchmark": lambda: run_benchmark.run_benchmark("baseline", Path("tasks/level1"), 1),
    "make_tables.render_tables": lambda: make_tables.render_tables(Path("results")),
}


@pytest.mark.parametrize("name", sorted(STUBS))
def test_stub_raises_not_implemented(name: str) -> None:
    with pytest.raises(NotImplementedError, match=r"TODO\(phase-\d\)"):
        STUBS[name]()
