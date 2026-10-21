#pragma once
#include <sw/alignment.hpp>
namespace sw::detail {
struct StripedTrace;
StripedTrace* create_trace();
void destroy_trace(StripedTrace*) noexcept;
AlignmentResult execute(SequenceView, SequenceView, Scoring, StripedTrace&);
std::int32_t execute_score(SequenceView, SequenceView, Scoring);
} // namespace sw::detail
