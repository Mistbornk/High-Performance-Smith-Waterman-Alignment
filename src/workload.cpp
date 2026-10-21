#include <sw/workload.hpp>
#include <limits>
#include <stdexcept>

namespace sw {
const char* pairing_name(Pairing pairing) noexcept {
  switch (pairing) {
    case Pairing::zip: return "zip";
    case Pairing::all_vs_all: return "all_vs_all";
  }
  return "unknown";
}

Workload::Workload(io::SequenceSet references, io::SequenceSet queries, Pairing pairing)
    : references_(std::move(references)), queries_(std::move(queries)), pairing_(pairing) {
  constexpr auto max = std::numeric_limits<std::uint64_t>::max();
  if (pairing_ == Pairing::zip) {
    if (references_.size() != queries_.size())
      throw std::invalid_argument("zip pairing needs equal reference and query counts (" +
          std::to_string(references_.size()) + " vs " + std::to_string(queries_.size()) + ")");
    size_ = references_.size();
    for (std::size_t i = 0; i < size_; ++i) {
      const std::uint64_t n = references_[i].size(), m = queries_[i].size();
      if (m && n > (max - cells_) / m) throw std::length_error("Workload cell count overflow");
      cells_ += n * m;
    }
  } else if (pairing_ == Pairing::all_vs_all) {
    const std::uint64_t r = references_.size(), q = queries_.size();
    if (q && r > std::numeric_limits<std::size_t>::max() / q)
      throw std::length_error("all_vs_all task count overflow");
    size_ = static_cast<std::size_t>(r * q);
    const std::uint64_t n = references_.total_bases(), m = queries_.total_bases();
    if (m && n > max / m) throw std::length_error("Workload cell count overflow");
    cells_ = n * m;
  } else {
    throw std::invalid_argument("Unknown pairing");
  }
}
} // namespace sw
