cmake_minimum_required(VERSION 3.18)
project(SmithWaterman VERSION 0.1.0 LANGUAGES CXX)

option(SW_ENABLE_STRIPED "Build the x86 SSE4.1 striped backend" ON)
option(SW_BUILD_EXAMPLES "Build the alignment CLI example" ON)

add_library(sw
  src/alignment.cpp
  src/backends/scalar/backend.cpp
  src/backends/cuda/backend.cpp)
add_library(sw::alignment ALIAS sw)
target_include_directories(sw PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_compile_features(sw PUBLIC cxx_std_17)
set_target_properties(sw PROPERTIES CXX_EXTENSIONS OFF)

if(SW_ENABLE_STRIPED)
  add_subdirectory(src/backends/striped)
else()
  target_sources(sw PRIVATE src/backends/striped/unavailable.cpp)
endif()

if(SW_BUILD_EXAMPLES)
  add_executable(sw-align examples/align.cpp)
  target_link_libraries(sw-align PRIVATE sw::alignment)
endif()
include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
