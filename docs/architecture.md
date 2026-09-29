# Architecture

Status: skeleton (phase 0). Sections marked TODO are filled in as the matching phase lands.

## Components

```
 vka.agent (Python)              kharness (Rust)                  vka_runner (C++/CUDA)
 ------------------              ---------------                  ---------------------
 planner                         compile  (nvcc, sandboxed)       load ONE candidate .so
   -> coder  --- candidate.cu -> verify   (fuzzed shapes/seeds) -> generate inputs, NaN-poison outputs
   -> reviewer                   guards   (anti-reward-hacking)    call vka_launch(...)
      ^                          bench    (median, bootstrap CI)   cudaEvent timing, L2 flush
      |                          profile  (ncu, optional)          JSON on stdout
      +------ JSON report <----------------------------------------+
```

- **Candidate**: a `.cu` file exporting the C ABI in `kernels/include/vka/kernels.h`.
- **vka_runner**: executes exactly one candidate per process, so a crash or hang in the
  candidate cannot corrupt the harness or other candidates.
- **kharness**: the grader. It is a separate program in a separate language from the agent, and
  it treats the candidate as untrusted.
- **vka.agent**: calls `kharness` only through fixed subcommands (`vka/agent/tools.py`); it never
  runs arbitrary shell.

## Data flow

1. The task TOML (`tasks/`) defines tensors, shape symbols, fuzz ranges, tolerances, and the
   reference function.
2. `kharness compile` builds the candidate into a shared library in a temp directory.
3. `kharness verify` drives the runner over the nominal shape plus fuzzed shapes with fresh
   OS-seeded inputs and compares against the reference.
4. `kharness bench` times candidate and baseline under the same methodology.
5. The combined JSON report feeds the reviewer and the reward (`vka/reward.py`).

TODO(phase-2): describe how reference outputs reach the runner (Python reference vs. C++
reference) once that is decided.
