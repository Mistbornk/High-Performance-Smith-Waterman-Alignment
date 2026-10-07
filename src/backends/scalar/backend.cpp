#include "backend.hpp"
#include <stdexcept>
namespace sw::detail {
const char* scalar_unavailable_reason() noexcept { return "scalar backend is not implemented"; }
std::unique_ptr<Aligner> make_scalar(Scoring) {
  throw std::runtime_error(scalar_unavailable_reason());
}
} // namespace sw::detail
