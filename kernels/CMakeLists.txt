# Each baseline is its own shared library: every candidate exports the same `vka_launch` symbol,
# so they cannot share one library, and the runner loads exactly one of them per process.
set(VKA_BASELINE_KERNELS
    gemm_naive
    gemm_tiled
    softmax_online
    layernorm_fused
    reservoir_step
)

foreach(kernel IN LISTS VKA_BASELINE_KERNELS)
  add_library(vka_${kernel} SHARED ${kernel}.cu)
  target_link_libraries(vka_${kernel} PRIVATE vka_abi)
  set_target_properties(vka_${kernel} PROPERTIES
    CUDA_VISIBILITY_PRESET hidden
    CXX_VISIBILITY_PRESET hidden
  )
endforeach()
