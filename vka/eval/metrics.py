"""Benchmark metrics: pass@k, fast_p, cost per success, guard trigger rates.

See README: Benchmark and metrics.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

FAST_P_THRESHOLDS = (1.0, 1.2, 2.0)

Record = dict[str, Any]


def pass_at_k(records: Iterable[Record], k: int) -> float:
    """Fraction of tasks with >= 1 fully verified candidate within the first ``k`` attempts."""
    # TODO(phase-6)
    raise NotImplementedError("TODO(phase-6): pass_at_k")


def fast_p(records: Iterable[Record], p: float) -> float:
    """Fraction of tasks with a verified candidate at least ``p`` times faster than baseline."""
    # TODO(phase-6)
    raise NotImplementedError("TODO(phase-6): fast_p")


def cost_per_success(records: Iterable[Record]) -> float:
    """Tokens spent per verified candidate."""
    # TODO(phase-6)
    raise NotImplementedError("TODO(phase-6): cost_per_success")


def guard_trigger_rates(records: Iterable[Record]) -> dict[str, float]:
    """Per guard: fraction of attempts on which it caught something."""
    # TODO(phase-6)
    raise NotImplementedError("TODO(phase-6): guard_trigger_rates")
