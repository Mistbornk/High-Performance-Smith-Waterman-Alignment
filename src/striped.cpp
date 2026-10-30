// The only translation unit compiled with C++20, -msse4.1 and libsimdpp.
// Include the kernel in exactly one translation unit: it has non-inline definitions.
#include "striped/global_alignment.hpp"
#include <sw/sw.hpp>
#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace sw::striped {
namespace {
// Per-thread trace scratch, reused across calls.
thread_local simdpp::SIMDPP_ARCH_NAMESPACE::SIMDVec trace;

void validate(SequenceView ref, SequenceView query, Scoring p) {
  const auto open = -std::int64_t(p.gap_open), extend = -std::int64_t(p.gap_extend);
  const auto mismatch = -std::int64_t(p.mismatch);
  if (p.match <= 0 || mismatch < 0 || extend <= 0 || open < extend)
    throw std::invalid_argument("Requires match>0, mismatch<=0, gap_open<=gap_extend<0");
  if (p.match + 2 * open > 127 || mismatch > 2 * open)
    throw std::length_error("Scoring exceeds striped byte-difference limits");
  for (auto sequence : {ref, query})
    for (Base base : sequence)
      if (base < 0 || base > 4) throw std::invalid_argument("Encoded DNA must contain 0..4");
  const std::int64_t magnitude = std::max({std::int64_t(p.match), mismatch, open, extend});
  const auto limit = static_cast<std::size_t>(INT32_MAX / (4 * magnitude));
  if (ref.size() > limit || query.size() > limit - ref.size())
    throw std::length_error("Alignment exceeds striped index/score range");
}

GlobalAlignmentStriped<> make_kernel(Scoring p) {
  return {static_cast<std::uint8_t>(p.match), static_cast<std::uint8_t>(-p.mismatch),
          static_cast<std::uint8_t>(-p.gap_open), static_cast<std::uint8_t>(-p.gap_extend),
          static_cast<std::uint8_t>(-p.gap_open), static_cast<std::uint8_t>(-p.gap_extend)};
}
} // namespace

std::int32_t score(SequenceView ref, SequenceView query, Scoring p) {
  validate(ref, query, p);
  return make_kernel(p).score(ref, query);
}

Alignment align(SequenceView ref, SequenceView query, Scoring p) {
  validate(ref, query, p);
  const auto width = simdpp::SIMDPP_ARCH_NAMESPACE::SIMD_WIDTH;
  const auto segments = (query.size() + width - 1) / width;
  if (segments && ref.size() > trace.max_size() / segments)
    throw std::length_error("Traceback allocation size overflow");
  auto [score, cigar] = make_kernel(p).align(ref, query, trace);
  return {score, std::move(cigar)};
}
} // namespace sw::striped
