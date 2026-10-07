#pragma once
#include <sw/alignment.hpp>
namespace sw::detail {
const char* striped_unavailable_reason() noexcept;
const char* scalar_unavailable_reason() noexcept;
const char* cuda_unavailable_reason() noexcept;
std::unique_ptr<Aligner> make_striped(Scoring);
std::unique_ptr<Aligner> make_scalar(Scoring);
std::unique_ptr<Aligner> make_cuda(Scoring);
} // namespace sw::detail
