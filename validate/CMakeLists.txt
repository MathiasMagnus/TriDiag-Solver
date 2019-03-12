cmake_minimum_required(VERSION 3.8) # CUDA language support

set(Target_Name Validate)

set(Hdrs Validate.hpp
         ../OpenCL/TriDiagSolver.hpp
         ../CUDA/spike_kernel.hxx
         ../CUDA/cusparse_ops.hxx)

set(Srcs Main.cu
         ../CUDA/spike_host.cu)

add_executable(${Target_Name} ${Hdrs} ${Srcs})

target_include_directories(${Target_Name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
                                                  ../OpenCL
                                                  ../CUDA)

target_link_libraries(${Target_Name} PRIVATE OpenCL::OpenCL)

#target_compile_options(${Target_Name} PRIVATE $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -pedantic>
#                                              $<$<CXX_COMPILER_ID:GNU>:-Wno-ignored-attributes>
#											  $<$<CXX_COMPILER_ID:MSVC>:/W4>)

set_target_properties(${Target_Name} PROPERTIES CXX_STANDARD 14
                                                CXX_STANDARD_REQUIRED ON
                                                CXX_EXTENSIONS OFF
												CUDA_STANDARD 14
                                                CUDA_STANDARD_REQUIRED ON
                                                CUDA_EXTENSIONS OFF)

source_group ("Headers" FILES ${Hdrs})
source_group ("Sources" FILES ${Srcs})