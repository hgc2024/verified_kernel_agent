"""Correctness-gated reward (README: Reward).

Correctness gates everything: a candidate that fails to compile, verify, or pass every blocking
guard earns exactly zero, however fast it is. Otherwise the reward is the clipped log speedup
over the baseline, so equal speed scores 0 and slowdowns are negative but bounded.

    reward = clip(log(baseline_median / candidate_median), -1, +2)
             - lambda_tokens * tokens_used - lambda_attempts * attempts
"""

from __future__ import annotations

REWARD_MIN = -1.0
REWARD_MAX = 2.0


def reward(
    *,
    compiled: bool,
    verified: bool,
    all_guards_passed: bool,
    baseline_median_ms: float | None,
    candidate_median_ms: float | None,
    tokens_used: int = 0,
    attempts: int = 0,
    lambda_tokens: float = 0.0,
    lambda_attempts: float = 0.0,
) -> float:
    # TODO(phase-6): implement and unit-test failure -> 0, equal -> 0, faster > 0, slower < 0.
    raise NotImplementedError("TODO(phase-6): reward")
