# TODO(phase-2): link CUDA::cudart and the platform dynamic loader once the runner executes
# candidates. Until then it is plain C++ so it builds without nvcc.
add_executable(vka_runner runner.cpp)
target_link_libraries(vka_runner PRIVATE vka_abi)
