# High-Performance Smith-Waterman Alignment

## Environment

- CMake 3.18 or newer.
- GCC or Clang with C++20 support for the striped SIMD backend.
- An x86 CPU with SSE4.1 support for the striped SIMD backend.
- A CMake-compatible build tool, such as GNU Make or Ninja.

The public API requires C++17. The striped kernel requires C++20.
The libsimdpp headers are included in `third_party/`; no separate installation
is required. The current build does not require the CUDA toolkit.

## Build

```sh
git clone https://github.com/Mistbornk/High-Performance-Smith-Waterman-Alignment.git
cd High-Performance-Smith-Waterman-Alignment
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
```

The default build produces the `sw` library, the `sw-align` example, and tests.

Optional CMake flags:

- `-DSW_BUILD_EXAMPLES=OFF`: disable the command-line example.
- `-DBUILD_TESTING=OFF`: disable tests.
- `-DSW_ENABLE_STRIPED=OFF`: build only the API skeleton on platforms without
  the striped backend; scalar and CUDA backends are not implemented.
