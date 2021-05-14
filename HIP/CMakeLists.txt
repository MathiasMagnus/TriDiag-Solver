cmake_minimum_required(VERSION 3.8) # cxx_std_17

set(Target_Name TriDiagSolver-HIP)

set(Hdrs spike_kernel.hpp
         cusparse_ops.hpp)

set(Srcs spike_host.hip.cpp
         main.cpp)

add_executable(${Target_Name} ${Hdrs} ${Srcs})

if(USE_HIP_CPU)
  target_link_libraries(${Target_Name}
    PUBLIC
      hip_cpu_rt::hip_cpu_rt
  )
  target_compile_options(${Target_Name}
    PUBLIC
      $<$<CXX_COMPILER_ID:MSVC>:/bigobj>
  )
else()
  target_link_libraries(${Target_Name}
    hip::device
  )
endif()

source_group ("Headers" FILES ${Hdrs})
source_group ("Sources" FILES ${Srcs})