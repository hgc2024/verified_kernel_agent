# verified_kernel_agent

An agentic system that writes and optimizes CUDA kernels. A tamper-resistant compile → verify →
profile harness, written in Rust, scores its work. The project also includes a correctness-gated
reward, a benchmark suite, and trajectory logging for improving models.

The grader is the product here. The agent is the thing being evaluated.

Most of this README describes the design. The [Status](#status) section says what exists today.

## Contents

- [Status](#status)
- [Why verification matters](#why-verification-matters)
- [Architecture](#architecture)
- [Quickstart (Linux / WSL2)](#quickstart-linux--wsl2)
- [Kernels](#kernels)
- [Harness (`kharness`)](#harness-kharness)
- [Agent (`vka.agent`)](#agent-vkaagent)
- [Reward](#reward)
- [Benchmark and metrics](#benchmark-and-metrics)
- [Reservoir kernel](#reservoir-kernel)
- [Testing and CI](#testing-and-ci)
- [Make targets](#make-targets)
- [Contributing](#contributing)
- [Results](#results)

## Status

A phase is checked only once its "done when" condition has been met.

| | Phase | Deliverable | Done when |
|---|---|---|---|
| [ ] | 0: Skeleton | Layout, build systems (CMake, Cargo, pyproject), CI, README with TBD blocks, stubs that raise `NotImplementedError` | `make build`, `cargo test`, and `pytest -m "not gpu"` pass on a clean clone |
| [ ] | 1: Kernels | `gemm_naive`, `gemm_tiled`, `softmax_online`, `layernorm_fused` + references + GPU parity tests | All GPU parity tests pass on fuzzed shapes |
| [ ] | 2: Harness | `kharness compile/verify/bench` + runner + stats + JSON | `kharness run` works end to end on the baselines |
| [ ] | 3: Guards | All [anti-reward-hacking guards](#anti-reward-hacking-guards) + cheating fixtures | Every cheater fixture is rejected by the intended guard |
| [ ] | 4: Reservoir | `reservoir_step` (Euler, then RK4) + parity vs. the JAX project | Parity test passes over N steps within the documented tolerance |
| [ ] | 5: Agent | Provider client, roles, loop, memory, trajectory logging | Agent produces a verified candidate on ≥1 level-1 task with a local model |
| [ ] | 6: Eval | Reward, metrics, benchmark runner, generated tables | `make bench && make tables` updates this README from real runs |
| [ ] | 7 (stretch) | Profiler-guided ablations; small fine-tune on logged trajectories | Ablation table generated; held-out tasks evaluated |

Phase 0 files are in place. The Python checks pass. `cargo test` and the nvcc kernel build have
not run yet on a machine with both toolchains.

Unimplemented stages raise `NotImplementedError` (Python), return
`HarnessError::NotImplemented` (Rust), or return `cudaErrorNotSupported` (CUDA), each with a
`TODO(phase-N)` marker. They never return fake data.

## Why verification matters

An agent that "optimizes" GPU code is only as good as its grader. Weak graders get reward-hacked
by cached outputs, skipped synchronization, or code that special-cases the test shapes. This
project treats verification and evaluation as the product. Every candidate is untrusted: it must
compile, match a reference over fuzzed shapes with fresh random inputs, and pass every guard
before its speed counts at all.

**Goals**

- Hand-written CUDA C++ baseline kernels with correct, reproducible benchmarks.
- A Rust harness that verifies candidate kernels robustly and times them properly.
- An agent loop (planner / coder / reviewer) that improves kernels using compiler, test, and
  profiler feedback.
- A reward function and benchmark suite with honest, script-generated reporting.

**Non-goals (for now)**

- Beating cuBLAS / cuDNN in general. Baselines are for learning and for grading the agent.
- Multi-GPU, multi-node, or distributed training.
- Training large models. A small fine-tune on logged trajectories is a stretch goal.

## Architecture

```
 vka.agent (Python)              kharness (Rust)                  vka_runner (C++/CUDA)
 planner                         compile  (nvcc, sandboxed)       load ONE candidate .so
   -> coder  --- candidate.cu -> verify   (fuzzed shapes/seeds) -> generate inputs, NaN-poison outputs
   -> reviewer                   guards   (anti-reward-hacking)    call vka_launch(...)
      ^                          bench    (median, bootstrap CI)   cudaEvent timing, L2 flush
      |                          profile  (ncu, optional)          JSON on stdout
      +------ JSON report <----------------------------------------+
```

More detail: [docs/architecture.md](docs/architecture.md).

```
verified_kernel_agent/
├── README.md                    # this file; the results block is GENERATED
├── LICENSE                      # MIT
├── .github/workflows/ci.yml     # CPU-only CI
├── CMakeLists.txt               # top-level; builds kernels/ and runner/
├── pyproject.toml               # python package `vka`
├── Makefile                     # convenience targets
├── .env.example                 # provider configuration template (copy to .env)
│
├── kernels/                     # hand-written CUDA C++ baselines, one shared library each
│   ├── include/vka/kernels.h    # C ABI declarations
│   ├── gemm_naive.cu
│   ├── gemm_tiled.cu            # shared-memory tiling
│   ├── softmax_online.cu        # single-pass (online) softmax, row-wise
│   ├── layernorm_fused.cu
│   └── reservoir_step.cu        # fused port of the memristive reservoir update
│
├── runner/runner.cpp            # loads ONE candidate: input generation, timing, JSON to stdout
│
├── harness/                     # Rust crate `kharness`
│   └── src/
│       ├── main.rs              # CLI (clap)
│       ├── lib.rs               # module wiring + HarnessError
│       ├── task.rs              # task TOML schema + loader
│       ├── compile.rs           # nvcc invocation, timeouts, captured diagnostics
│       ├── sandbox.rs           # subprocess isolation: timeout, env scrub, temp dirs
│       ├── verify.rs            # multi-trial correctness with fuzzed shapes/seeds
│       ├── timing.rs            # median, bootstrap CI, speedup
│       ├── guards.rs            # anti-reward-hacking checks
│       ├── profile.rs           # optional ncu invocation + metric extraction
│       └── report.rs            # JSON result schema
│
├── tasks/
│   ├── level1/*.toml            # single-op kernels (gemm, softmax, layernorm, ...)
│   ├── level2/                  # fused ops (bias+gelu, layernorm+residual, reservoir_step, ...)
│   └── level3/                  # small multi-kernel pipelines (stretch)
│
├── vka/                         # Python package
│   ├── reference.py             # PyTorch reference implementations for each task
│   ├── reward.py                # correctness-gated reward
│   ├── agent/
│   │   ├── loop.py              # planner → coder → harness → reviewer loop
│   │   ├── roles.py             # prompts/templates for planner, coder, reviewer
│   │   ├── tools.py             # compile / verify / profile / read_task wrappers around kharness
│   │   ├── memory.py            # per-task attempt history (JSONL)
│   │   └── providers.py         # OpenAI-compatible client (vLLM, Ollama, hosted APIs)
│   └── eval/
│       ├── run_benchmark.py     # runs agents over tasks, writes results/*.jsonl
│       ├── metrics.py           # pass@k, fast_p, cost per success
│       └── make_tables.py       # renders README tables from results (never hand-edited)
│
├── tests/
│   ├── python/                  # pytest; GPU tests marked @pytest.mark.gpu
│   ├── cpp/                     # optional CPU-side tests for runner utilities
│   └── fixtures/cheaters/       # deliberately cheating candidates the guards must reject
├── results/                     # gitignored except results/README.md (schema)
└── docs/
    ├── architecture.md
    ├── timing_methodology.md
    ├── anti_reward_hacking.md
    └── reservoir_kernel.md
```

## Quickstart (Linux / WSL2)

Target environment: Linux or WSL2 (Ubuntu 22.04+). Prerequisites:

- NVIDIA GPU with a recent driver, and CUDA Toolkit 12.x (`nvcc`)
- Nsight Compute CLI (`ncu`), optional, for profiling
- CMake ≥ 3.24 and a C++17 compiler
- Rust stable (`rustup`)
- Python ≥ 3.11

```bash
cp .env.example .env      # then set VKA_BASE_URL / VKA_MODEL (and VKA_API_KEY for hosted APIs)
make setup                # venv, cargo build, cmake configure
make build                # kernels + runner + harness
make test                 # CPU tests: cargo test + pytest -m "not gpu"
make test-gpu             # GPU tests (requires a CUDA device)
```

For GPU tests and the PyTorch references, install the extra: `pip install -e ".[gpu]"`. Pick the
torch build that matches your CUDA version from pytorch.org.

Windows-native is best-effort. On Windows, `.venv\Scripts\python -m pytest -m "not gpu"` runs the
Python tests.

## Kernels

### Baselines

| Kernel | Notes |
|---|---|
| `gemm_naive` | One thread per output element. Correctness reference for the tiling work. |
| `gemm_tiled` | Shared-memory tiling (start with 32×32 tiles), coalesced loads; handles M/N/K not divisible by the tile size. |
| `softmax_online` | Row-wise, single-pass online max/sum (numerically stable), one block per row (or warp-per-row for short rows). |
| `layernorm_fused` | Mean/variance via Welford in one pass; fused scale and shift. |
| `reservoir_step` | Fused state update for the memristive reservoir ([below](#reservoir-kernel)). |

- FP32 first. FP16/BF16 variants come in a later phase.
- Every kernel has a PyTorch reference in `vka/reference.py`. References compute in float64 and
  cast back, so their own rounding error stays well below the task tolerance.
- Every kernel documents its launch configuration in a header comment.
- Kernels build with `CMAKE_CUDA_ARCHITECTURES=native` by default. Override it with
  `make build CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89"`.
- Each baseline is its own shared library, because every candidate exports the same symbol.

### C ABI contract

Every candidate, hand-written or agent-generated, exports one fixed C function so the runner can
load them all the same way (`kernels/include/vka/kernels.h`):

```c
// Returns 0 on success, non-zero cudaError_t-compatible code on failure.
// All pointers are device pointers. The runner owns allocation and synchronization.
VKA_EXPORT int vka_launch(const void* const* inputs, void* const* outputs,
                          const long long* shape, int shape_len, void* stream);
```

- `inputs` and `outputs` follow the order declared in the task TOML.
- `shape` holds each shape symbol's concrete value, in the order symbols first appear across
  inputs and then outputs. For gemm, with `A[M,K]`, `B[K,N]`, `C[M,N]`, that gives
  `shape = {M, K, N}`.
- `stream` is a `cudaStream_t` passed as `void*`.
- The runner compiles or loads **one** candidate per process.

## Harness (`kharness`)

### CLI

```
kharness compile  --task tasks/level1/gemm_fp32.toml --candidate path/to/cand.cu
kharness verify   --task ... --candidate ... [--trials 20] [--seed-from-os]
kharness bench    --task ... --candidate ... [--warmup 10] [--iters 100]
kharness profile  --task ... --candidate ... [--metrics default]
kharness run      --task ... --candidate ...           # compile + verify + guards + bench
```

Every subcommand prints a single JSON object to stdout ([schema](#result-schema)) and logs to
stderr.

### Task schema

A task's `id` matches its path: `level1.gemm_fp32` lives at `tasks/level1/gemm_fp32.toml`.

```toml
id = "level1.gemm_fp32"
description = "C = A @ B, row-major, FP32"
reference = "vka.reference:gemm"          # python reference that generates expected outputs
baseline = "kernels/gemm_tiled.cu"        # speedup is measured against this
inputs  = [{ name = "A", dtype = "f32", shape = ["M", "K"] },
           { name = "B", dtype = "f32", shape = ["K", "N"] }]
outputs = [{ name = "C", dtype = "f32", shape = ["M", "N"] }]

[shapes]
nominal = { M = 1024, N = 1024, K = 1024 }
fuzz    = { M = [1, 4096], N = [1, 4096], K = [1, 2048], include_non_multiples_of = [16, 32, 128] }

[tolerance]
rtol = 1e-4
atol = 1e-5

[constraints]                             # optional; checked by the static lint guard
forbid = ["cublas", "cublaslt", "cudnn", "cutlass", "thrust"]
```

`kharness` validates each task when it loads it. The loader rejects unknown keys, any shape
symbol without a nominal size and fuzz range, inverted fuzz ranges, and negative tolerances.

### Verification

- Each candidate runs multiple trials: the nominal shape plus fuzzed shapes. Fuzzed shapes include
  sizes that are not multiples of common tile sizes, 1×1, and tall-skinny shapes.
- Every trial gets **fresh random inputs**, seeded from the OS. The seed is recorded in the
  results but never exposed to the candidate at compile time.
- Outputs are compared against the reference using `rtol` and `atol`. The report gives the max
  absolute and relative error and the index of the first mismatch.

### Anti-reward-hacking guards

Each guard produces a named pass/fail with evidence (`harness/src/guards.rs`):

1. **Output poisoning:** output buffers are filled with NaN before each launch, so untouched
   outputs fail.
2. **Input variation:** the candidate runs twice with different inputs. Identical outputs on
   different inputs means caching.
3. **Shape variation:** fuzzed shapes defeat special-casing of the nominal shape.
4. **Synchronization:** timing includes a device-wide sync after the launch, and a separate check
   validates results after a full sync. This catches kernels that are "fast" because they didn't
   finish.
5. **Static source lint:** rejects candidates that call libraries the task forbids, read files,
   use environment variables, spawn processes, or reference the reference implementation.
6. **Resource limits:** wall-clock timeout, memory cap, and no network (best effort, via the
   sandbox).
7. **Determinism note:** records whether outputs are bitwise identical across runs. This is
   informational and never fails a candidate.

[docs/anti_reward_hacking.md](docs/anti_reward_hacking.md) documents each guard, the exploit it
prevents, and the test that shows it catching a deliberately cheating fixture in
`tests/fixtures/cheaters/`.

### Timing methodology

- Warm-up iterations, then timed iterations measured with `cudaEvent`s.
- **L2 flush** between iterations: the runner writes a buffer larger than L2, sized from device
  properties.
- The report gives the median, p10/p90, and a 95% bootstrap confidence interval.
- It also gives achieved GFLOP/s and GB/s, plus % of theoretical peak (FP32 and memory bandwidth)
  from device properties. [docs/timing_methodology.md](docs/timing_methodology.md) documents the
  formulas.
- GPU name, driver, CUDA version, and clocks are recorded **at runtime** and never hardcoded.

### Result schema

The values below are schema placeholders, not measurements. The authoritative definition is
`harness/src/report.rs`, and [results/README.md](results/README.md) documents it.

```json
{
  "task_id": "level1.gemm_fp32",
  "candidate": "path or hash",
  "compile": {"ok": true, "stderr": "..."},
  "verify": {"ok": true, "trials": 20, "max_abs_err": 0.0, "max_rel_err": 0.0, "failures": []},
  "guards": [{"name": "output_poisoning", "ok": true, "evidence": "..."}],
  "timing": {"median_ms": 0.0, "p10_ms": 0.0, "p90_ms": 0.0, "ci95_ms": [0.0, 0.0],
             "baseline_median_ms": 0.0, "speedup": 0.0},
  "env": {"gpu": "detected-at-runtime", "driver": "...", "cuda": "..."},
  "seed": 123
}
```

## Agent (`vka.agent`)

### Providers

The agent uses a single OpenAI-compatible client. It works with **vLLM** (`vllm serve <model>`),
**Ollama** (through its OpenAI-compatible endpoint), and hosted APIs. Configure it with
`VKA_BASE_URL`, `VKA_API_KEY`, and `VKA_MODEL`, set in the environment or in `.env` (see
`.env.example`). Token usage and wall time are logged for every call.

### Roles and loop

```
planner  : reads the task + baseline + last profile → proposes an optimization plan (tiling, vectorized loads, warp-level reductions, fusion, ...)
coder    : writes a full candidate .cu implementing the C ABI
harness  : kharness run → compile errors | verification failures | guard failures | timing | profile metrics
reviewer : explains the failure or bottleneck from harness JSON (+ ncu metrics), proposes the next step
repeat until budget (max attempts, max tokens, max wall time) is exhausted
```

- **Memory:** a per-task JSONL of attempts (plan, code hash, harness result, reviewer notes). The
  agent sees a compact summary of prior attempts, not the full raw history.
- **Tool calls** are thin wrappers around the `kharness` CLI. The agent never executes arbitrary
  shell commands.
- **Trajectory logging:** every attempt is written to `results/trajectories/*.jsonl` for later
  analysis or fine-tuning.

### Ablations

These are configurable in `vka.agent.loop.AgentConfig`:

- One-shot vs. iterative.
- With vs. without profiler feedback.
- Single agent vs. planner/coder/reviewer.

## Reward

Defined in `vka/reward.py` (stub until phase 6):

```
if not (compiled and verified and all_guards_passed): reward = 0
else: reward = clip(log(baseline_median / candidate_median), -1, +2)
optional penalties: λ_tokens * tokens_used, λ_attempts * attempts
```

Required unit tests for the edge cases: failure → 0; equal speed → 0; faster → positive; slower →
negative (bounded).

## Benchmark and metrics

- `python -m vka.eval.run_benchmark --agent <config> --tasks tasks/level1 --attempts 5` writes
  `results/runs/<timestamp>.jsonl`.
- `vka/eval/metrics.py`:
  - **pass@k**: fraction of tasks with ≥1 fully verified candidate within k attempts.
  - **fast_p** (p ∈ {1.0, 1.2, 2.0}): fraction of tasks with a verified candidate at least p×
    faster than the baseline.
  - **cost per success**: tokens (and $ if configured) per verified candidate.
  - **Guard trigger rates**: how often each anti-hacking guard caught something.
- `make tables` renders markdown tables into the [Results](#results) block of this README. Until
  real runs exist, that block contains only the TBD placeholder.

## Reservoir kernel

`reservoir_step` is a fused CUDA port of the state update from
[hgc2024/physical_reservoir](https://github.com/hgc2024/physical_reservoir). That project is a
JAX implementation of a volatile memristive reservoir with bounded state variables, recurrent
dynamics, and fixed-step Euler/RK4 integration.

- The dynamics must match that repository's source exactly; none are invented here. Anything
  ambiguous gets a `TODO` and a clearly marked stub.
- Euler comes first, then RK4. Each time step fuses the recurrent matvec, input projection,
  nonlinearity, bounding/clipping, and integration into a single kernel where possible.
- A parity test runs against a NumPy/JAX-equivalent reference over N steps, with a documented
  tolerance.
- [docs/reservoir_kernel.md](docs/reservoir_kernel.md) explains the fusion choices.

## Testing and CI

This is the target coverage. Today's tests cover the task files, the stubs, provider
configuration, the README results block, Rust task parsing, and JSON schema round-trips.

- **Python** (pytest): CPU-only tests cover the task files, reward math, metrics, the table
  renderer, and provider request formatting (mocked). GPU tests are marked `@pytest.mark.gpu`.
- **Rust** (`cargo test`): task parsing, stats (median/bootstrap), guard logic on synthetic data,
  and JSON schema round-trips.
- **CUDA:** GPU tests compare each baseline kernel against its PyTorch reference over fuzzed
  shapes.
- **Cheating fixtures:** each `tests/fixtures/cheaters/*.cu` must be rejected by the intended
  guard (GPU test).
- **CI** (GitHub Actions, CPU-only): `cargo fmt --check`, `cargo clippy -D warnings`,
  `cargo test`, `ruff check`, `ruff format --check`, `pytest -m "not gpu"`,
  `clang-format --dry-run`, and a CMake build of the runner. CI has no nvcc, so GPU tests run
  locally.

## Make targets

```
make setup        # python venv, cargo build, cmake configure
make build        # kernels + runner + harness
make test         # cpu tests
make test-gpu     # gpu tests (requires CUDA device)
make bench        # baselines over level1 tasks → results/
make agent TASK=level1.gemm_fp32
make tables       # regenerate README results block from results/
make lint         # cargo fmt/clippy, ruff, clang-format (check only)
make fmt          # apply formatters
```

## Contributing

These rules apply to human and AI contributors alike.

1. **Never fabricate results.** No benchmark numbers, speedups, pass rates, GPU names, or
   "achieves X%" claims anywhere (README, docs, docstrings, commit messages) unless they came from
   running the code in this repo. Use `TBD` placeholders; result tables are *generated* by
   scripts.
2. **Every phase compiles.** After each phase, `cmake --build`, `cargo test`, and
   `pytest -m "not gpu"` must pass. GPU tests are marked and may be skipped on machines without a
   GPU.
3. **Small, reviewable commits** with conventional messages (`feat(harness): ...`,
   `test(kernels): ...`).
4. **No large binaries, datasets, or model weights** in git. Add them to `.gitignore`.
5. **Prefer clarity over cleverness** in the skeleton. Performance work comes later and is
   measured.
6. **Keep stubs honest.** A stub raises `NotImplementedError` or returns an explicit error, with a
   `TODO(phase-N)` comment. It never returns fake data.

Style:

- C++17 / CUDA C++, `clang-format` (Google style, 100 columns).
- Rust 2021 edition, `clippy` clean; `anyhow` + `thiserror` for errors; `clap`, `serde`, `toml`,
  `serde_json`.
- Python: `ruff`, type hints, `dataclasses` or `pydantic` for schemas.
- Docstrings explain *why*, not just *what*. No marketing language in code or docs.

## Results

`make tables` generates this block from `results/`. Do not edit it by hand.

<!-- RESULTS:START -->
Results: TBD. Generated by `make tables`.
<!-- RESULTS:END -->

## Related projects

- [hgc2024/physical_reservoir](https://github.com/hgc2024/physical_reservoir)
- [hgc2024/embodied_vla_sim_loop](https://github.com/hgc2024/embodied_vla_sim_loop)

## License

MIT
