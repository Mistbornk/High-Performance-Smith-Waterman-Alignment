#include "backend.hpp"
#include <stdexcept>
namespace sw::detail {
const char* cuda_unavailable_reason() noexcept { return "cuda backend is not implemented"; }
std::unique_ptr<Aligner> make_cuda(Scoring) {
  throw std::runtime_error(cuda_unavailable_reason());
}
std::unique_ptr<BatchAligner> make_cuda_batch(Scoring, BatchOptions) {
  throw std::runtime_error(cuda_unavailable_reason());
}
} // namespace sw::detail
