#pragma once
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace sw {
struct CigarOperation {
  std::uint32_t size;
  char op;
  CigarOperation(std::uint32_t length, char operation) : size(length), op(operation) {}
};
// M consumes both inputs, I consumes query, D consumes reference.
class Cigar : public std::vector<CigarOperation> {
 public:
  Cigar() = default;
  explicit Cigar(std::string_view text) {
    std::uint32_t length = 0;
    bool digits = false;
    for (char c : text) {
      if (c >= '0' && c <= '9') {
        const auto digit = static_cast<std::uint32_t>(c - '0');
        if (length > (std::numeric_limits<std::uint32_t>::max() - digit) / 10)
          throw std::length_error("CIGAR length overflow");
        length = length * 10 + digit;
        digits = true;
      } else {
        if (!digits || !length || (c != 'M' && c != 'I' && c != 'D'))
          throw std::invalid_argument("Invalid alignment CIGAR");
        emplace_back(length, c);
        length = 0;
        digits = false;
      }
    }
    if (digits) throw std::invalid_argument("CIGAR is missing an operation");
  }
  std::string to_string() const {
    std::string result;
    for (const auto& item : *this) result += std::to_string(item.size) + item.op;
    return result;
  }
  explicit operator std::string() const { return to_string(); }
};
} // namespace sw
