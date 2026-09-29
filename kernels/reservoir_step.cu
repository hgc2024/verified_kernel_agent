// reservoir_step: fused state update for the memristive reservoir from hgc2024/physical_reservoir.
//
// The dynamics must match that repository's source exactly: they are not defined here yet on
// purpose. See docs/reservoir_kernel.md.
//
// Plan: Euler step first, then RK4, fusing the recurrent matvec, input projection, nonlinearity,
// bounding/clipping, and integration into one kernel per time step where possible.
//
// Launch configuration: TODO(phase-4): document once the equations are ported.
//
// Stub: returns cudaErrorNotSupported so callers see an explicit failure, never a silently
// untouched output buffer.
#include <cuda_runtime.h>

#include "vka/kernels.h"

extern "C" VKA_EXPORT int vka_launch(const void* const* inputs, void* const* outputs,
                                     const long long* shape, int shape_len, void* stream) {
  // TODO(phase-4): port the state update from hgc2024/physical_reservoir.
  (void)inputs;
  (void)outputs;
  (void)shape;
  (void)shape_len;
  (void)stream;
  return static_cast<int>(cudaErrorNotSupported);
}
