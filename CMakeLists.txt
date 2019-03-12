cmake_minimum_required(VERSION 3.8) # CUDA language support

project(TriDiagSolver LANGUAGES CXX)

option(TRIDIAG_CUDA "Build the original CUDA accelerated solver" OFF)
option(TRIDIAG_OPENCL "Build the ported OpenCL accelerated solver" ON)

# Strip default warning level from host compiler when using MSVC
if (MSVC)
  string(REGEX REPLACE "/W[0-9]" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif (MSVC)

if(TRIDIAG_CUDA)
  enable_language(CUDA)
  add_subdirectory(CUDA)
endif(TRIDIAG_CUDA)

if(TRIDIAG_OPENCL)
  find_package(OpenCL REQUIRED)
  add_subdirectory(OpenCL)
endif(TRIDIAG_OPENCL)

if(TRIDIAG_CUDA AND TRIDIAG_OPENCL)
  add_subdirectory(validate)
endif(TRIDIAG_CUDA AND TRIDIAG_OPENCL)