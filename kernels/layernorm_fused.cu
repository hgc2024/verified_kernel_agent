// layernorm_fused: Y = (X - mean) / sqrt(var + eps) * gamma + beta over the last dim, FP32.
// Task: tasks/level1/layernorm_fp32.toml. Shape symbols (see vka/kernels.h): shape = {R, C}.
//
// Plan: mean and variance via Welford in one pass, with the scale and shift fused into the
// same kernel.
//
// Launch configuration: TODO(phase-1): document the block mapping and why it was chosen.
//
// Stub: returns cudaErrorNotSupported so callers see an explicit failure, never a silently
// untouched output buffer.
#include <cuda_runtime.h>

#include "vka/kernels.h"

extern "C" VKA_EXPORT int vka_launch(const void* const* inputs, void* const* outputs,
                                     const long long* shape, int shape_len, void* stream) {
  // TODO(phase-1): implement the fused LayerNorm kernel and launch it on `stream`.
  (void)inputs;
  (void)outputs;
  (void)shape;
  (void)shape_len;
  (void)stream;
  return static_cast<int>(cudaErrorNotSupported);
}
