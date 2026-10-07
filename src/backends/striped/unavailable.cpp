#include "backend.hpp"
#include <stdexcept>
namespace sw::detail {
const char* striped_unavailable_reason() noexcept { return "striped backend was disabled at build time"; }
std::unique_ptr<Aligner> make_striped(Scoring) {
  throw std::runtime_error(striped_unavailable_reason());
}
} // namespace sw::detail
