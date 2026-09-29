# results/

Everything in this directory except this file is gitignored. It is written by scripts, never by
hand.

| Path | Written by | Contents |
|---|---|---|
| `runs/<timestamp>.jsonl` | `vka.eval.run_benchmark` | One line per attempt: a `kharness` run report plus agent metadata. |
| `trajectories/*.jsonl` | `vka.agent.loop` | Full per-attempt trajectories (plan, code hash, harness result, reviewer notes, token usage). |

## `kharness` run report

Every `kharness` subcommand prints one JSON object with this shape. The authoritative definition
is `harness/src/report.rs`; the values below are placeholders.

```json
{
  "task_id": "level1.gemm_fp32",
  "candidate": "path or hash",
  "compile": {"ok": true, "stderr": "..."},
  "verify": {"ok": true, "trials": 20, "max_abs_err": 0.0, "max_rel_err": 0.0, "failures": []},
  "guards": [{"name": "output_poisoning", "ok": true, "evidence": "..."}],
  "timing": {"median_ms": 0.0, "p10_ms": 0.0, "p90_ms": 0.0, "ci95_ms": [0.0, 0.0],
             "baseline_median_ms": 0.0, "speedup": 0.0},
  "profile": {"metrics": {}},
  "env": {"gpu": "detected-at-runtime", "driver": "...", "cuda": "..."},
  "seed": 123
}
```

- Sections for stages that did not run are omitted.
- `env` is always detected at runtime.
- `seed` is recorded for reproduction and is never visible to the candidate.
- When a subcommand fails inside the harness (as opposed to the candidate failing), the output
  is `{"task": ..., "candidate": ..., "error": "..."}` and the exit code is non-zero.

TODO(phase-5/6): document the `runs/` and `trajectories/` line schemas once they exist.
