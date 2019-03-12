cmake_minimum_required(VERSION 3.7) # OpenCL::OpenCL imported target syntax

set(Target_Name TriDiagSolver-OpenCL)

set(Hdrs TriDiagSolver.hpp)

set(Srcs Main.cpp)

set(Krns TriDiagSolver.cl)

add_executable(${Target_Name} ${Hdrs} ${Srcs} ${Krns})

target_include_directories(${Target_Name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${Target_Name} PRIVATE OpenCL::OpenCL)

target_compile_options(${Target_Name} PRIVATE $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -pedantic>
                                              $<$<CXX_COMPILER_ID:GNU>:-Wno-ignored-attributes>
											  $<$<CXX_COMPILER_ID:MSVC>:/W4>)

set_target_properties(${Target_Name} PROPERTIES CXX_STANDARD 14
                                                CXX_STANDARD_REQUIRED ON
                                                CXX_EXTENSIONS OFF)

source_group ("Headers" FILES ${Hdrs})
source_group ("Sources" FILES ${Srcs})
source_group ("Kernels" FILES ${Krns})