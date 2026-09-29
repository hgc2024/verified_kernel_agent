// gemm_tiled: C = A @ B, row-major, FP32, shared-memory tiling. Task: tasks/level1/gemm_fp32.toml.
// Shape symbols (see vka/kernels.h): shape = {M, K, N}.
//
// Plan: start with 32x32 tiles and coalesced global loads; M/N/K need not be multiples of the
// tile size, so edge tiles must be bounds-checked.
//
// Launch configuration: TODO(phase-1): document the tile/block shape and why it was chosen.
//
// Stub: returns cudaErrorNotSupported so callers see an explicit failure, never a silently
// untouched output buffer.
#include <cuda_runtime.h>

#include "vka/kernels.h"

extern "C" VKA_EXPORT int vka_launch(const void* const* inputs, void* const* outputs,
                                     const long long* shape, int shape_len, void* stream) {
  // TODO(phase-1): implement the tiled GEMM kernel and launch it on `stream`.
  (void)inputs;
  (void)outputs;
  (void)shape;
  (void)shape_len;
  (void)stream;
  return static_cast<int>(cudaErrorNotSupported);
}
