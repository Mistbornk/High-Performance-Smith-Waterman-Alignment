#pragma once
#include <sw/alignment.hpp>
#include <sw/batch.hpp>
namespace sw::detail {
const char* striped_unavailable_reason() noexcept;
const char* scalar_unavailable_reason() noexcept;
const char* cuda_unavailable_reason() noexcept;
std::unique_ptr<Aligner> make_striped(Scoring);
std::unique_ptr<Aligner> make_scalar(Scoring);
std::unique_ptr<Aligner> make_cuda(Scoring);
// CUDA batches whole workloads on the device instead of driving per-thread Aligners.
std::unique_ptr<BatchAligner> make_cuda_batch(Scoring, BatchOptions);
} // namespace sw::detail
