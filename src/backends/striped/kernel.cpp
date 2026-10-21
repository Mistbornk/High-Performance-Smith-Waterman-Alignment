#include "kernel.hpp"
// Include in exactly one translation unit: the kernel has non-inline definitions.
#include "detail/global_alignment.hpp"
#include <stdexcept>

namespace sw::detail {
namespace {
sw::GlobalAlignmentStriped<> make_kernel(Scoring p) {
  return {static_cast<std::uint8_t>(p.match), static_cast<std::uint8_t>(-p.mismatch),
          static_cast<std::uint8_t>(-p.gap_open), static_cast<std::uint8_t>(-p.gap_extend),
          static_cast<std::uint8_t>(-p.gap_open), static_cast<std::uint8_t>(-p.gap_extend)};
}
} // namespace
struct StripedTrace { simdpp::SIMDPP_ARCH_NAMESPACE::SIMDVec data; };
StripedTrace* create_trace() { return new StripedTrace; }
void destroy_trace(StripedTrace* trace) noexcept { delete trace; }
AlignmentResult execute(SequenceView ref, SequenceView query, Scoring p, StripedTrace& scratch) {
  const auto width = simdpp::SIMDPP_ARCH_NAMESPACE::SIMD_WIDTH;
  const auto segments = (query.size() + width - 1) / width;
  if (segments && ref.size() > scratch.data.max_size() / segments)
    throw std::length_error("Traceback allocation size overflow");
  auto [score, cigar] = make_kernel(p).align(ref, query, scratch.data);
  return {score, std::move(cigar), 0, ref.size(), 0, query.size()};
}
std::int32_t execute_score(SequenceView ref, SequenceView query, Scoring p) {
  return make_kernel(p).score(ref, query);
}
}  // namespace sw::detail
