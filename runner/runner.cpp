// vka_runner: loads ONE candidate shared library, generates inputs, calls the reference and the
// candidate's `vka_launch`, times it, and prints a single JSON object to stdout (logs to stderr).
//
// It runs as a subprocess of kharness so that a candidate that crashes, hangs, or corrupts the
// CUDA context takes down only this process, never the harness.
//
// Planned responsibilities (TODO(phase-2)):
//   - dlopen/LoadLibrary the candidate and resolve `vka_launch`.
//   - allocate inputs/outputs from the task description passed by kharness; NaN-poison outputs.
//   - warm-up, then timed iterations with cudaEvents and an L2 flush between iterations.
//   - record GPU name, driver, CUDA version, and clocks at runtime.
#include <cstdio>

#include "vka/kernels.h"

int main(int argc, char** argv) {
  (void)argc;
  (void)argv;
  std::fprintf(stderr, "vka_runner: not implemented yet (TODO(phase-2))\n");
  std::printf("{\"ok\": false, \"error\": \"vka_runner not implemented (TODO(phase-2))\"}\n");
  return 2;
}
