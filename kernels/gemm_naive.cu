// gemm_naive: C = A @ B, row-major, FP32. Task: tasks/level1/gemm_fp32.toml.
// Shape symbols (see vka/kernels.h): shape = {M, K, N}.
//
// Role: correctness reference for the tiled GEMM work, one thread per output element.
//
// Launch configuration: TODO(phase-1): document the block shape and why it was chosen.
//
// Stub: returns cudaErrorNotSupported so callers see an explicit failure, never a silently
// untouched output buffer.
#include <cuda_runtime.h>

#include "vka/kernels.h"

extern "C" VKA_EXPORT int vka_launch(const void* const* inputs, void* const* outputs,
                                     const long long* shape, int shape_len, void* stream) {
  // TODO(phase-1): implement the naive GEMM kernel and launch it on `stream`.
  (void)inputs;
  (void)outputs;
  (void)shape;
  (void)shape_len;
  (void)stream;
  return static_cast<int>(cudaErrorNotSupported);
}
