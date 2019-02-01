cmake_minimum_required(VERSION 3.7) # OpenCL::OpenCL imported target syntax

project(OpenCL-TriDiagSolver LANGUAGES CXX)

if (MSVC)
  string(REGEX REPLACE "/W[0-9]" "/W4" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
endif (MSVC)

find_package(OpenCL REQUIRED)

set(Hdrs TriDiagSolver.hpp)

set(Srcs TriDiagSolver.cpp
         Main.cpp)

set(Krns TriDiagSolver.cl)

add_executable(${PROJECT_NAME} ${Hdrs} ${Srcs})

target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)

target_compile_options(${PROJECT_NAME} PRIVATE $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wall -Wextra -pedantic>)

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 14
                                                 CXX_STANDARD_REQUIRED ON
                                                 CXX_EXTENSIONS OFF)

source_group ("Headers" FILES ${Hdrs})
source_group ("Sources" FILES ${Srcs})
source_group ("Kernels" FILES ${Krns})

if(CMAKE_GENERATOR MATCHES "Visual Studio")
  set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
endif(CMAKE_GENERATOR MATCHES "Visual Studio")