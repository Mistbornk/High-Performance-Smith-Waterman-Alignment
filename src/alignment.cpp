#include "backend.hpp"
#include <stdexcept>
namespace sw {
const char* backend_name(Backend backend) noexcept {
  switch (backend) {
    case Backend::striped: return "striped";
    case Backend::scalar: return "scalar";
    case Backend::cuda: return "cuda";
  }
  return "unknown";
}
const char* unavailable_reason(Backend backend) noexcept {
  switch (backend) {
    case Backend::striped: return detail::striped_unavailable_reason();
    case Backend::scalar: return detail::scalar_unavailable_reason();
    case Backend::cuda: return detail::cuda_unavailable_reason();
  }
  return "Unknown alignment backend";
}
std::unique_ptr<Aligner> make_aligner(Backend backend, Scoring scoring) {
  const auto reason = unavailable_reason(backend);
  if (*reason) throw std::runtime_error(reason);
  switch (backend) {
    case Backend::striped: return detail::make_striped(scoring);
    case Backend::scalar: return detail::make_scalar(scoring);
    case Backend::cuda: return detail::make_cuda(scoring);
  }
  throw std::invalid_argument("Unknown alignment backend");
}
} // namespace sw
