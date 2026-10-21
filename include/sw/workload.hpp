#pragma once
#include <sw/io.hpp>
#include <cstdint>

namespace sw {
// zip:        task i = (reference i, query i); both sets must have equal size.
// all_vs_all: every query against every reference, query-major like a database
//             search (CUDASW4): task i = (reference i % R, query i / R).
// Tasks are computed, not materialized, so a large all_vs_all costs no memory.
enum class Pairing { zip, all_vs_all };
const char* pairing_name(Pairing pairing) noexcept;

struct AlignmentTask {
  std::uint32_t reference;
  std::uint32_t query;
};

class Workload {
 public:
  Workload(io::SequenceSet references, io::SequenceSet queries, Pairing pairing);
  const io::SequenceSet& references() const noexcept { return references_; }
  const io::SequenceSet& queries() const noexcept { return queries_; }
  Pairing pairing() const noexcept { return pairing_; }
  std::size_t size() const noexcept { return size_; }
  AlignmentTask task(std::size_t i) const noexcept {
    if (pairing_ == Pairing::zip) return {std::uint32_t(i), std::uint32_t(i)};
    const auto r = references_.size();
    return {std::uint32_t(i % r), std::uint32_t(i / r)};
  }
  SequenceView reference(std::size_t i) const noexcept { return references_[task(i).reference]; }
  SequenceView query(std::size_t i) const noexcept { return queries_[task(i).query]; }
  // Sum of |reference| * |query| over all tasks: the DP cell count used for GCUPS.
  std::uint64_t cells() const noexcept { return cells_; }

 private:
  io::SequenceSet references_, queries_;
  Pairing pairing_;
  std::size_t size_ = 0;
  std::uint64_t cells_ = 0;
};
} // namespace sw
