// softmax_online: Y = softmax(X, dim=-1), row-wise, row-major, FP32.
// Task: tasks/level1/softmax_fp32.toml. Shape symbols (see vka/kernels.h): shape = {R, C}.
//
// Plan: single-pass online max/sum (numerically stable), one block per row, or one warp per row
// for short rows.
//
// Launch configuration: TODO(phase-1): document the block/warp mapping and why it was chosen.
//
// Stub: returns cudaErrorNotSupported so callers see an explicit failure, never a silently
// untouched output buffer.
#include <cuda_runtime.h>

#include "vka/kernels.h"

extern "C" VKA_EXPORT int vka_launch(const void* const* inputs, void* const* outputs,
                                     const long long* shape, int shape_len, void* stream) {
  // TODO(phase-1): implement the online softmax kernel and launch it on `stream`.
  (void)inputs;
  (void)outputs;
  (void)shape;
  (void)shape_len;
  (void)stream;
  return static_cast<int>(cudaErrorNotSupported);
}
