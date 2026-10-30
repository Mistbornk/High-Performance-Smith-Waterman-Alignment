#pragma once
// Everything a caller needs: DNA encoding, CIGAR, scoring, the per-backend align/score
// functions and a small parallel_for used to run many pairs on CPU threads.
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(__CUDACC__)
#define SW_HOST_DEVICE __host__ __device__
#else
#define SW_HOST_DEVICE
#endif
// Defined to 1 by CMake when the SSE4.1 striped / CUDA backends are built.
#ifndef SW_HAVE_STRIPED
#define SW_HAVE_STRIPED 0
#endif
#ifndef SW_HAVE_CUDA
#define SW_HAVE_CUDA 0
#endif

namespace sw {
// ---- DNA encoding: A=0 C=1 G=2 T=3 N=4, one byte per base (host and device). ----
using Base = std::int8_t;
SW_HOST_DEVICE constexpr Base encode_base(char c) noexcept {
  switch (c) {
    case 'A': case 'a': return 0;
    case 'C': case 'c': return 1;
    case 'G': case 'g': return 2;
    case 'T': case 't': return 3;
    default: return 4;
  }
}
SW_HOST_DEVICE constexpr char decode_base(int c) noexcept {
  switch (c) {
    case 0: return 'A'; case 1: return 'C'; case 2: return 'G'; case 3: return 'T';
    default: return 'N';
  }
}
SW_HOST_DEVICE constexpr Base complement_base(Base c) noexcept {
  return c >= 0 && c < 4 ? static_cast<Base>(3 - c) : Base{4};
}

using Sequence = std::basic_string<Base>;
using SequenceView = std::basic_string_view<Base>;
// Strict: accepts A/C/G/T/N in either case.
inline Sequence encode_sequence(std::string_view text) {
  Sequence result;
  result.reserve(text.size());
  for (char c : text) {
    switch (c) {
      case 'A': case 'a': case 'C': case 'c': case 'G': case 'g':
      case 'T': case 't': case 'N': case 'n': break;
      default: throw std::invalid_argument("DNA must contain only A/C/G/T/N");
    }
    result.push_back(encode_base(c));
  }
  return result;
}

// ---- CIGAR: M consumes both inputs, I consumes query, D consumes reference. ----
struct CigarOperation {
  std::uint32_t size;
  char op;
  CigarOperation(std::uint32_t length, char operation) : size(length), op(operation) {}
};
class Cigar : public std::vector<CigarOperation> {
 public:
  Cigar() = default;
  explicit Cigar(std::string_view text) {
    std::uint32_t length = 0;
    bool digits = false;
    for (char c : text) {
      if (c >= '0' && c <= '9') {
        const auto digit = static_cast<std::uint32_t>(c - '0');
        if (length > (std::numeric_limits<std::uint32_t>::max() - digit) / 10)
          throw std::length_error("CIGAR length overflow");
        length = length * 10 + digit;
        digits = true;
      } else {
        if (!digits || !length || (c != 'M' && c != 'I' && c != 'D'))
          throw std::invalid_argument("Invalid alignment CIGAR");
        emplace_back(length, c);
        length = 0;
        digits = false;
      }
    }
    if (digits) throw std::invalid_argument("CIGAR is missing an operation");
  }
  std::string to_string() const {
    std::string result;
    for (const auto& item : *this) result += std::to_string(item.size) + item.op;
    return result;
  }
};

// ---- Global affine-gap alignment: gap(k) = gap_open + (k-1) * gap_extend. ----
// N is a fifth symbol (N/N matches). Every backend requires gap_open <= gap_extend <= 0.
struct Scoring {
  int match = 1;
  int mismatch = -4;
  int gap_open = -6;
  int gap_extend = -1;
};
struct Alignment {
  std::int32_t score = 0;
  Cigar cigar;  // Always covers both whole inputs.
};

// Naive Gotoh, one cell at a time with 32-bit scores. The reference implementation.
namespace scalar {
std::int32_t score(SequenceView reference, SequenceView query, Scoring scoring = {});
Alignment align(SequenceView reference, SequenceView query, Scoring scoring = {});
} // namespace scalar

#if SW_HAVE_STRIPED
// SSE4.1 striped SIMD with 8-bit differences; rejects scoring outside the byte range.
namespace striped {
std::int32_t score(SequenceView reference, SequenceView query, Scoring scoring = {});
Alignment align(SequenceView reference, SequenceView query, Scoring scoring = {});
} // namespace striped
#endif

class Workload;  // sw/io.hpp

#if SW_HAVE_CUDA
// One warp per pair on the GPU; score-only.
namespace cuda {
// Empty when a device can run the kernels, otherwise the reason it cannot.
const char* unavailable_reason() noexcept;
// Device copy of a workload: the constructor uploads it, score() can then run many
// times. Throws runtime_error when no usable device is present.
class Batch {
 public:
  explicit Batch(const Workload& workload);
  ~Batch();
  Batch(const Batch&) = delete;
  Batch& operator=(const Batch&) = delete;
  // Synchronous; out[i] is the global alignment score of workload task i. `regs` is the
  // number of query columns each lane owns (1, 2, 4, 8 or 16); 0 groups the
  // tasks by query length and picks it per group.
  void score(Scoring scoring, std::vector<std::int32_t>& out, int regs = 0);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace cuda
#endif

// Calls body(i) for every i in [0, n) on up to `threads` threads (the caller is one
// of them), handing out indices in chunks because pair sizes vary. The first
// exception stops the remaining work and is rethrown.
template <class Body> void parallel_for(std::size_t n, unsigned threads, Body body) {
  constexpr std::size_t chunk = 16;
  std::atomic<std::size_t> next{0};
  std::atomic<bool> stop{false};
  std::exception_ptr error;
  std::mutex error_mutex;
  auto run = [&] {
    try {
      while (!stop.load(std::memory_order_relaxed)) {
        const auto begin = next.fetch_add(chunk, std::memory_order_relaxed);
        if (begin >= n) return;
        for (auto i = begin, end = std::min(n, begin + chunk); i < end; ++i) body(i);
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(error_mutex);
      if (!error) error = std::current_exception();
      stop = true;
    }
  };
  const auto count = std::max<std::size_t>(1, std::min<std::size_t>(threads, (n + chunk - 1) / chunk));
  std::vector<std::thread> workers;
  for (std::size_t i = 1; i < count; ++i) workers.emplace_back(run);
  run();
  for (auto& worker : workers) worker.join();
  if (error) std::rethrow_exception(error);
}
} // namespace sw
#undef SW_HOST_DEVICE
