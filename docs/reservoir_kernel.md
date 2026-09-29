# Reservoir kernel

Status: planned (phase 4). `kernels/reservoir_step.cu` and `vka.reference.reservoir_step` are
stubs.

## Source of truth

The dynamics come from [`hgc2024/physical_reservoir`](https://github.com/hgc2024/physical_reservoir),
a JAX implementation of a volatile memristive reservoir with bounded state variables, recurrent
dynamics, and fixed-step Euler/RK4 integration.

This document deliberately does not restate the equations yet. They get copied here, with
references to the exact source files and functions, when the port is done. Anything ambiguous in
the source gets a `TODO` and a marked stub rather than an invented choice.

## Plan

1. Euler step, then RK4.
2. Fuse the recurrent matvec, input projection, nonlinearity, bounding/clipping, and integration
   into a single kernel per time step where possible.
3. Parity test over N steps against a NumPy/JAX-equivalent reference, with a documented
   tolerance.

## Fusion choices

TODO(phase-4)

## Parity tolerance

TODO(phase-4)
