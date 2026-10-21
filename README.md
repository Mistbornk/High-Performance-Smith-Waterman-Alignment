# High-Performance Smith-Waterman Alignment

[![CI](https://github.com/Mistbornk/High-Performance-Smith-Waterman-Alignment/actions/workflows/ci.yml/badge.svg?event=pull_request)](https://github.com/Mistbornk/High-Performance-Smith-Waterman-Alignment/actions/workflows/ci.yml)

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

The default build produces the `sw` library, the `sw-align` example, the
`sw-bench` benchmark driver, and tests.

Optional CMake flags:

- `-DSW_BUILD_EXAMPLES=OFF`: disable the command-line example.
- `-DSW_BUILD_BENCHMARKS=OFF`: disable `sw-bench`.
- `-DBUILD_TESTING=OFF`: disable tests.
- `-DSW_ENABLE_STRIPED=OFF`: build without the striped backend on platforms without
  SSE4.1; the scalar backend is still available. CUDA is not implemented yet.

## Benchmark

```sh
python3 scripts/gen_dataset.py --out data/len512 --pairs 2000 --length 512
python3 scripts/run_benchmarks.py --bench build/sw-bench --dataset data/len512 \
  --threads 1,4,16 --out results/run1
```

See [docs/benchmark.md](docs/benchmark.md) for the IO, batch API and timing contract.
