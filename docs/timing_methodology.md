# Timing methodology

Status: planned (phase 2). Nothing here is implemented yet.

## Procedure

1. Warm-up iterations (default 10), not recorded.
2. Timed iterations (default 100), each bracketed by `cudaEvent`s on the launch stream.
3. Between iterations, flush L2 by writing a buffer larger than the device's L2 (size read from
   `cudaDeviceProp::l2CacheSize` at runtime), so each iteration starts cold.
4. A device-wide sync after the launch ends each measurement. Results are validated separately
   after a full sync (see the synchronization guard in `anti_reward_hacking.md`).

## Statistics

- Median, p10, p90 of per-iteration times.
- 95% confidence interval of the median by bootstrap resampling.
- Speedup = baseline median / candidate median, with both measured in the same process session.

TODO(phase-2): number of bootstrap resamples and the RNG used.

## Throughput and % of peak

TODO(phase-2): document the exact formulas. The plan:

- Achieved GFLOP/s = task FLOP count / median time. The FLOP count is defined per task
  (e.g. `2*M*N*K` for GEMM).
- Achieved GB/s = minimum bytes moved (inputs read once, outputs written once) / median time.
- Peaks come from device properties at runtime (SM count, clock, FP32 lanes per SM; memory clock
  and bus width). None are hardcoded.

## Environment capture

GPU name, driver version, CUDA runtime version, and clocks are recorded at runtime in every
report's `env` section.
