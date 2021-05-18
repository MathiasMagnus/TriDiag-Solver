cmake_minimum_required(VERSION 3.8) # CUDA language support

project(TriDiagSolver LANGUAGES CXX)

option(TRIDIAG_CUDA "Build the original CUDA accelerated solver" OFF)
option(TRIDIAG_HIP "Build the ported HIP accelerated solver" OFF)
option(TRIDIAG_OPENCL "Build the ported OpenCL accelerated solver" ON)
option(USE_HIP_CPU "Prefer HIP-CPU runtime instead of HW acceleration" OFF)
option(BUILD_EXAMPLE "Build exmamples" ON)
option(BUILD_TESTING "Build unit tests" ON)
option(BUILD_BENCHMARK "Build benchmarks" ON)

if(BUILD_TESTING)
  include(CTest)
  find_package(GTest CONFIG REQUIRED)
endif()

if(BUILD_BENCHMARK)
  find_package(benchmark REQUIRED CONFIG)
endif()

if(TRIDIAG_CUDA)
  enable_language(CUDA)
  add_subdirectory(CUDA)
endif(TRIDIAG_CUDA)

if(TRIDIAG_HIP)
  if(USE_HIP_CPU)
    find_package(hip_cpu_rt REQUIRED)
  else()
    set(AMDGPU_TARGETS "gfx803;gfx900;gfx906;gfx908" CACHE STRING "List of specific machine types for library to target")
    find_package(hip REQUIRED CONFIG)
  endif()
  add_subdirectory(HIP)
endif(TRIDIAG_HIP)

if(TRIDIAG_OPENCL)
  find_package(OpenCL REQUIRED)
  add_subdirectory(OpenCL)
endif(TRIDIAG_OPENCL)