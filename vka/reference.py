"""PyTorch reference implementations, one per task.

A task's ``reference = "vka.reference:<name>"`` resolves to a function here, which produces the
expected outputs candidates are graded against. References favour obvious correctness over speed
and compute in float64 before casting back, so the reference's own rounding error stays well below
the task tolerance.

``torch`` is imported lazily so the CPU-only test suite can import this module without it.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

LAYERNORM_EPS = 1e-5


def resolve(spec: str) -> Callable[..., Any]:
    """Resolve a ``"module:function"`` reference string from a task TOML."""
    module_name, sep, attr = spec.partition(":")
    if not sep or not module_name or not attr:
        raise ValueError(f"reference {spec!r} must have the form 'module:function'")
    fn = getattr(importlib.import_module(module_name), attr)
    if not callable(fn):
        raise TypeError(f"reference {spec!r} is not callable")
    return fn


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """C = A @ B for row-major FP32 matrices."""
    return (a.double() @ b.double()).to(a.dtype)


def softmax(x: torch.Tensor) -> torch.Tensor:
    """Row-wise softmax over the last dimension."""
    import torch

    return torch.softmax(x.double(), dim=-1).to(x.dtype)


def layernorm(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """LayerNorm over the last dimension with biased variance and eps = LAYERNORM_EPS."""
    import torch

    y = torch.nn.functional.layer_norm(
        x.double(), (x.shape[-1],), gamma.double(), beta.double(), LAYERNORM_EPS
    )
    return y.to(x.dtype)


def reservoir_step(*args: Any, **kwargs: Any) -> Any:
    """Memristive reservoir state update, ported from hgc2024/physical_reservoir."""
    # TODO(phase-4): match the source repository's equations exactly; do not invent dynamics.
    raise NotImplementedError("TODO(phase-4): reservoir_step reference")
