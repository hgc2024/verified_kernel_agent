// C ABI contract shared by every candidate kernel (hand-written or agent-generated).
//
// The runner loads exactly one candidate shared library per process and calls `vka_launch`.
// A fixed, C-only ABI lets the runner treat every candidate uniformly and keeps C++ name mangling
// and templates out of the loading path.
//
// Conventions:
//   - `inputs` and `outputs` are device pointers, in the order the task TOML declares them.
//   - `shape` holds the concrete value of each shape symbol, in order of first appearance across
//     the task's inputs and then outputs. Example, gemm with A[M,K], B[K,N], C[M,N]:
//     shape = {M, K, N}, shape_len = 3.
//   - `stream` is a `cudaStream_t` passed as `void*` so this header does not depend on CUDA.
//   - The runner owns allocation and synchronization. Candidates must not allocate device memory
//     for outputs, synchronize the device, or touch anything outside the given buffers.
#ifndef VKA_KERNELS_H_
#define VKA_KERNELS_H_

#if defined(_WIN32)
#define VKA_EXPORT __declspec(dllexport)
#else
#define VKA_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Returns 0 on success, non-zero cudaError_t-compatible code on failure.
// All pointers are device pointers. The runner owns allocation and synchronization.
VKA_EXPORT int vka_launch(const void* const* inputs, void* const* outputs, const long long* shape,
                          int shape_len, void* stream);

#ifdef __cplusplus
}
#endif

#endif  // VKA_KERNELS_H_
