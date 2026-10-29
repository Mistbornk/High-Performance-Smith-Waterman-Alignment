cmake_minimum_required(VERSION 3.18)
project(SmithWaterman VERSION 0.2.0 LANGUAGES CXX)

option(SW_ENABLE_STRIPED "Build the x86 SSE4.1 striped backend" ON)
option(SW_ENABLE_CUDA "Build the CUDA backend (needs the CUDA toolkit)" OFF)
option(SW_BUILD_EXAMPLES "Build the sw-align example" ON)
option(SW_BUILD_BENCHMARKS "Build the sw-bench benchmark driver" ON)

add_library(sw src/io.cpp src/scalar.cpp)
target_include_directories(sw PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_compile_features(sw PUBLIC cxx_std_17)
set_target_properties(sw PROPERTIES CXX_EXTENSIONS OFF)
find_package(Threads REQUIRED)
target_link_libraries(sw PUBLIC Threads::Threads)

if(SW_ENABLE_STRIPED)
  if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86)$"
     OR NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "Striped needs GCC/Clang on x86. Use -DSW_ENABLE_STRIPED=OFF.")
  endif()
  # Only striped.cpp sees libsimdpp, C++20 and SSE4.1.
  add_library(sw-striped OBJECT src/striped.cpp)
  target_include_directories(sw-striped PRIVATE ${PROJECT_SOURCE_DIR}/include)
  target_include_directories(sw-striped SYSTEM PRIVATE ${PROJECT_SOURCE_DIR}/third_party/libsimdpp/include)
  set_target_properties(sw-striped PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
  target_compile_options(sw-striped PRIVATE -msse4.1)
  target_compile_definitions(sw-striped PRIVATE SIMDPP_ARCH_X86_SSE4_1 SW_HAVE_STRIPED=1)
  target_sources(sw PRIVATE $<TARGET_OBJECTS:sw-striped>)
  target_compile_definitions(sw PUBLIC SW_HAVE_STRIPED=1)
endif()

if(SW_ENABLE_CUDA)
  # Must precede enable_language; CMake 3.22 has no "native". 86 = RTX 30xx.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(sw PRIVATE src/cuda.cu)
  # sw was created before CUDA was enabled, so set its CUDA properties explicitly.
  set_target_properties(sw PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF)
  target_link_libraries(sw PRIVATE CUDA::cudart)
  target_compile_definitions(sw PUBLIC SW_HAVE_CUDA=1)
endif()

if(SW_BUILD_EXAMPLES)
  add_executable(sw-align examples/align.cpp)
  target_link_libraries(sw-align PRIVATE sw)
endif()
if(SW_BUILD_BENCHMARKS)
  add_executable(sw-bench benchmarks/sw_bench.cpp)
  target_link_libraries(sw-bench PRIVATE sw)
endif()
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
