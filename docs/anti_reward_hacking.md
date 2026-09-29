# Anti-reward-hacking guards

Status: planned (phase 3). Guard names match `harness/src/guards.rs`.

Each guard reports `{name, ok, evidence}`. Each one gets a deliberately cheating candidate in
`tests/fixtures/cheaters/`, and a test that shows the intended guard rejecting it.

| Guard | Exploit it prevents | Fixture | Status |
|---|---|---|---|
| `output_poisoning` | Leaving outputs untouched (e.g. relying on a previous correct run's buffer). Outputs are NaN-filled before every launch. | TODO | TODO(phase-3) |
| `input_variation` | Returning cached or precomputed outputs. Two runs with different inputs must not produce identical outputs. | TODO | TODO(phase-3) |
| `shape_variation` | Special-casing the nominal shape. Fuzzed shapes include non-multiples of common tile sizes, 1x1, and tall-skinny. | TODO | TODO(phase-3) |
| `synchronization` | Being "fast" because work was not finished (e.g. launching on another stream). Results are re-validated after a full device sync. | TODO | TODO(phase-3) |
| `static_lint` | Calling forbidden libraries (task `constraints.forbid`), reading files, reading environment variables, spawning processes, or referencing the reference implementation. | TODO | TODO(phase-3) |
| `resource_limits` | Hanging, exhausting memory, or reaching the network. Enforced best-effort by the sandbox. | TODO | TODO(phase-3) |
| `determinism` | Informational only: whether outputs are bitwise identical across runs. Never fails a candidate. | n/a | TODO(phase-3) |
