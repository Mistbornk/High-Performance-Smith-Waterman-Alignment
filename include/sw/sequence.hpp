#pragma once
#include <sw/base.hpp>
#include <stdexcept>
#include <string>
#include <string_view>

namespace sw {
using Sequence = std::basic_string<Base>;
using SequenceView = std::basic_string_view<Base>;
inline EncodedSequenceView encoded_view(SequenceView sequence) noexcept {
  return {sequence.data(), sequence.size()};
}
// Public ASCII input is strict; accepts A/C/G/T/N in either case.
inline Sequence encode_sequence(std::string_view text) {
  Sequence result;
  result.reserve(text.size());
  for (char c : text) {
    switch (c) {
      case 'A': case 'a': case 'C': case 'c': case 'G': case 'g':
      case 'T': case 't': case 'N': case 'n': break;
      default: throw std::invalid_argument("DNA must contain only A/C/G/T/N");
    }
    result.push_back(encode_base(c));
  }
  return result;
}
} // namespace sw
