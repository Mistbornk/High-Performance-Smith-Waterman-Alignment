# High-Performance Smith-Waterman Alignment

[![CI](https://github.com/Mistbornk/High-Performance-Smith-Waterman-Alignment/actions/workflows/ci.yml/badge.svg?event=pull_request)](https://github.com/Mistbornk/High-Performance-Smith-Waterman-Alignment/actions/workflows/ci.yml)

## Environment

- CMake 3.18 or newer.
- GCC or Clang with C++20 support for the striped SIMD backend.
- An x86 CPU with SSE4.1 support for the striped SIMD backend.
- A CMake-compatible build tool, such as GNU Make or Ninja.

The public API requires C++17. The striped kernel requires C++20.
The libsimdpp headers are included in `third_party/`; no separate installation
is required. The CUDA toolkit is only needed with `-DSW_ENABLE_CUDA=ON`.

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
  SSE4.1; the scalar backend is still available.
- `-DSW_ENABLE_CUDA=ON`: build the score-only CUDA backend (CUDA toolkit 12; defaults to
  `CMAKE_CUDA_ARCHITECTURES=86`, override for other GPUs).

## Layout

```text
include/sw/sw.hpp      encoding, Cigar, Scoring, Alignment, per-backend functions, parallel_for
include/sw/io.hpp      FASTA/FASTQ -> SequenceSet, Workload (zip | all_vs_all)
src/io.cpp             IO and Workload
src/scalar.cpp         sw::scalar::align / score   (naive Gotoh; the reference)
src/striped.cpp        sw::striped::align / score  (SSE4.1; only file built with C++20/libsimdpp)
src/striped/           striped SIMD kernel internals
src/cuda.cu            sw::cuda::Batch             (one warp per pair; score-only)
benchmarks/sw_bench.cpp  timing driver (one CSV row per run)
tests/                 align_test (all backends vs. an oracle), io_test, bench_flow_test.py
```

Every backend is a pair of free functions with the same signature:

```cpp
#include <sw/sw.hpp>
auto ref = sw::encode_sequence("ACGT"), query = sw::encode_sequence("ACTGT");
sw::Alignment a = sw::scalar::align(ref, query, {1, -4, -6, -1});  // score + CIGAR
std::int32_t s = sw::scalar::score(ref, query);                     // score only
```

Adding a backend means one source file plus one `case` in `sw-bench`.

## Benchmark

```sh
python3 scripts/gen_dataset.py --out data/len512 --pairs 2000 --length 512
python3 scripts/run_benchmarks.py --bench build/sw-bench --dataset data/len512 \
  --threads 1,4,16 --out results/run1
```

The CUDA backend runs with `--score-only`; `--regs N` (1, 2, 4, 8 or 16) sets how many query
columns each lane owns; by default tasks are grouped by query length and each group picks its own:

```sh
build/sw-bench --reference R.fa --query Q.fa --backend cuda --score-only --regs 8
```

See [docs/benchmark.md](docs/benchmark.md) for the IO and timing contract.
