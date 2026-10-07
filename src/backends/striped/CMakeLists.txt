if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86)$"
   OR NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(FATAL_ERROR "Striped needs GCC/Clang on x86. Use -DSW_ENABLE_STRIPED=OFF for the API skeleton.")
endif()
# Only the kernel sees third-party headers, C++20 and SSE4.1 flags.
add_library(sw-striped-kernel OBJECT kernel.cpp)
target_include_directories(sw-striped-kernel PRIVATE
  ${PROJECT_SOURCE_DIR}/include)
target_include_directories(sw-striped-kernel SYSTEM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/libsimdpp/include)
set_target_properties(sw-striped-kernel PROPERTIES CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)
target_compile_options(sw-striped-kernel PRIVATE -msse4.1)
target_compile_definitions(sw-striped-kernel PRIVATE SIMDPP_ARCH_X86_SSE4_1)
target_sources(sw PRIVATE backend.cpp $<TARGET_OBJECTS:sw-striped-kernel>)
