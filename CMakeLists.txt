cmake_minimum_required(VERSION 3.24)  # 3.24: CMAKE_CUDA_ARCHITECTURES=native

project(verified_kernel_agent LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VKA_BUILD_KERNELS "Build the CUDA baseline kernels (requires nvcc)" ON)

# The C ABI header is shared by kernels/, runner/, and agent-generated candidates.
add_library(vka_abi INTERFACE)
target_include_directories(vka_abi INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/kernels/include)

# CUDA is optional at configure time so the CPU-only CI can still build the runner.
include(CheckLanguage)
check_language(CUDA)
if(VKA_BUILD_KERNELS AND CMAKE_CUDA_COMPILER)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  add_subdirectory(kernels)
else()
  message(STATUS "vka: CUDA compiler not found or VKA_BUILD_KERNELS=OFF; skipping kernels/")
endif()

add_subdirectory(runner)
