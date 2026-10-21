#pragma once
#include <sw/alignment.hpp>
#include <sw/workload.hpp>
#include <memory>
#include <vector>

namespace sw {
struct BatchOptions {
  // CPU worker threads (each owns one Aligner). Device backends require 1.
  unsigned threads = 1;
  // false requests score-only results (empty CIGAR), the mode GPU database search
  // kernels such as CUDASW4 run in. Backends that cannot skip traceback reject it
  // instead of silently doing the extra work, so benchmark rows stay comparable.
  bool traceback = true;
};

// Aligns every task of a Workload; results[i] belongs to workload.task(i).
// Synchronous: returns only after all work (threads, device kernels, copies) is done.
// One instance per caller; concurrent align() calls on the same instance are unsupported.
class BatchAligner {
 public:
  virtual ~BatchAligner() = default;
  virtual Backend backend() const noexcept = 0;
  virtual BatchOptions options() const noexcept = 0;
  // `results` is resized to workload.size(); its capacity is reused across calls.
  virtual void align(const Workload& workload, std::vector<AlignmentResult>& results) = 0;
};

// CPU backends get a thread-parallel driver over per-thread Aligners; CUDA gets its
// own batch implementation. Throws for unavailable backends; never falls back.
std::unique_ptr<BatchAligner> make_batch_aligner(Backend backend, Scoring scoring = {},
                                                 BatchOptions options = {});
} // namespace sw
