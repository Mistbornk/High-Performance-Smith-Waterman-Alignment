#include "backend.hpp"
#include "kernel.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace sw::detail {
const char* striped_unavailable_reason() noexcept {
  return __builtin_cpu_supports("sse4.1") ? "" : "striped requires an SSE4.1 CPU";
}
namespace {
void validate_scoring(Scoring p) {
  const auto open = -std::int64_t(p.gap_open), extend = -std::int64_t(p.gap_extend);
  const auto mismatch = -std::int64_t(p.mismatch);
  if (p.match <= 0 || mismatch < 0 || extend <= 0 || open < extend)
    throw std::invalid_argument("Requires match>0, mismatch<=0, gap_open<=gap_extend<0");
  if (p.match + 2 * open > 127 || mismatch > 2 * open)
    throw std::length_error("Scoring exceeds striped byte-difference limits");
}
class StripedAligner final : public Aligner {
 public:
  explicit StripedAligner(Scoring scoring)
      : scoring_(scoring), trace_(create_trace(), destroy_trace) {}
  Backend backend() const noexcept override { return Backend::striped; }
  AlignmentResult align(SequenceView ref, SequenceView query) override {
    validate(ref, query);
    return execute(ref, query, scoring_, *trace_);
  }
  std::int32_t score(SequenceView ref, SequenceView query) override {
    validate(ref, query);
    return execute_score(ref, query, scoring_);
  }
 private:
  void validate(SequenceView ref, SequenceView query) const {
    for (auto sequence : {ref, query})
      for (Base base : sequence)
        if (base < 0 || base > 4) throw std::invalid_argument("Encoded DNA must contain 0..4");
    const auto p = scoring_;
    const std::int64_t magnitude = std::max({std::int64_t(p.match), -std::int64_t(p.mismatch),
        -std::int64_t(p.gap_open), -std::int64_t(p.gap_extend)});
    const auto limit = static_cast<std::size_t>(INT32_MAX / (4 * magnitude));
    if (ref.size() > limit || query.size() > limit - ref.size())
      throw std::length_error("Alignment exceeds striped index/score range");
  }
  Scoring scoring_;
  std::unique_ptr<StripedTrace, void(*)(StripedTrace*)> trace_;
};
} // namespace
std::unique_ptr<Aligner> make_striped(Scoring scoring) {
  // Baseline ISA code: hardware is checked before allocating SIMD scratch.
  if (const auto reason = striped_unavailable_reason(); *reason) throw std::runtime_error(reason);
  validate_scoring(scoring);
  return std::make_unique<StripedAligner>(scoring);
}
} // namespace sw::detail
