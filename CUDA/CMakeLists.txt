cmake_minimum_required(VERSION 3.8) # CUDA language support

set(Target_Name TriDiagSolver-CUDA)

set(Hdrs spike_kernel.hxx
         cusparse_ops.hxx)

set(Srcs spike_host.cu
         main.cu)

add_executable(${Target_Name} ${Hdrs} ${Srcs})

target_include_directories(${Target_Name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

#target_compile_options(${Target_Name} PRIVATE 
#                                              $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -pedantic>
#                                              $<$<CXX_COMPILER_ID:MSVC>:/W4>)

set_target_properties(${Target_Name} PROPERTIES CUDA_STANDARD 14
                                                CXX_STANDARD 14
                                                CXX_STANDARD_REQUIRED ON)
                                                #CXX_EXTENSIONS OFF)

source_group ("Headers" FILES ${Hdrs})
source_group ("Sources" FILES ${Srcs})