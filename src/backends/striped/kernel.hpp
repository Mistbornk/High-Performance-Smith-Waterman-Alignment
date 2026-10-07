#pragma once
#include <sw/alignment.hpp>
namespace sw::detail {
struct StripedTrace;
StripedTrace* create_trace();
void destroy_trace(StripedTrace*) noexcept;
AlignmentResult execute(SequenceView, SequenceView, Scoring, StripedTrace&);
} // namespace sw::detail
