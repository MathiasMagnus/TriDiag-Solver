cmake_minimum_required(VERSION 3.8) # cxx_std_17

add_library(tridiag_hip_common INTERFACE)

target_link_libraries(tridiag_hip_common
  INTERFACE
    $<$<BOOL:${USE_HIP_CPU}>:hip_cpu_rt::hip_cpu_rt>
    $<$<NOT:$<BOOL:${USE_HIP_CPU}>>:hip::device>
)
target_compile_options(tridiag_hip_common
  INTERFACE
    $<$<AND:$<BOOL:${USE_HIP_CPU}>,$<CXX_COMPILER_ID:MSVC>>:/bigobj>
)

if(BUILD_EXAMPLE)
  set(Headers
    tridiag_solver.hpp
    spike_kernel.hpp
    cusparse_ops.hpp
  )
  set(Sources
    tridiag_example_hip.cpp
  )
  set(Kernels
    spike_kernel.hpp
  )
  add_executable(tridiag_example_hip
    ${Headers}
    ${Sources}
    ${Kernels}
  )
  target_link_libraries(tridiag_example_hip
    INTERFACE
      tridiag_hip_common
  )
  source_group ("Headers" FILES ${Headers})
  source_group ("Sources" FILES ${Sources})
  source_group ("Kernels" FILES ${Kernels})
endif()

if(BUILD_TESTING)
  set(Headers
    tridiag_solver.hpp
    spike_kernel.hpp
    cusparse_ops.hpp
  )
  set(Sources
    tridiag_test_hip.cpp
  )
  set(Kernels
    spike_kernel.hpp
  )
  add_executable(tridiag_test_hip
    ${Headers}
    ${Sources}
    ${Kernels}
  )
  target_link_libraries(tridiag_test_hip
    PRIVATE
      tridiag_hip_common
      GTest::gtest
      GTest::gtest_main
  )
  add_test(
    NAME solve_hip
    COMMAND tridiag_test_hip
  )
endif()

if(BUILD_BENCHMARK)
  set(Headers
    tridiag_solver.hpp
    spike_kernel.hpp
    cusparse_ops.hpp
  )
  set(Sources
    tridiag_benchmark_hip.cpp
  )
  set(Kernels
    spike_kernel.hpp
  )
  add_executable(tridiag_benchmark_hip
    ${Headers}
    ${Sources}
    ${Kernels}
  )
  target_link_libraries(tridiag_benchmark_hip
    PRIVATE
      tridiag_hip_common
      benchmark::benchmark
  )
endif()