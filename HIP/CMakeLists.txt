cmake_minimum_required(VERSION 3.8) # cxx_std_17

set(Target_Name TriDiagSolver-HIP)

set(Hdrs
  tridiag_solver.hpp
  spike_kernel.hpp
  cusparse_ops.hpp
)

set(Srcs main.cpp)

add_executable(${Target_Name} ${Hdrs} ${Srcs})

if(USE_HIP_CPU)
  target_link_libraries(${Target_Name}
    PRIVATE
      hip_cpu_rt::hip_cpu_rt
  )
  target_compile_options(${Target_Name}
    PRIVATE
      $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
  )
else()
  target_link_libraries(${Target_Name}
    hip::device
  )
endif()

source_group ("Headers" FILES ${Hdrs})
source_group ("Sources" FILES ${Srcs})