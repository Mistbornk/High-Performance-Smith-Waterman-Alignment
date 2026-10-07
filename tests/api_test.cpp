#include "check.hpp"
#include <sw/alignment.hpp>
#include <iostream>
#include <type_traits>
int main() {
  try {
    static_assert(std::is_trivially_copyable<sw::EncodedSequenceView>::value);
    static_assert(sw::encode_base('t') == 3 && sw::complement_base(4) == 4);
    const auto sequence = sw::encode_sequence("aCgTn");
    CHECK(sequence.size() == 5);
    for (int i = 0; i < 5; ++i) CHECK(sequence[i] == i);
    CHECK(sw::encoded_view(sequence).data == sequence.data());
    expect_throw<std::invalid_argument>([] { sw::encode_sequence("AX"); });
    CHECK(sw::Cigar("3M1I2D").to_string() == "3M1I2D");
    for (const auto bad : {"0M", "1X", "M", "1", "1M0", "1M2"})
      expect_throw<std::invalid_argument>([&] { sw::Cigar cigar(bad); });
    expect_throw<std::length_error>([] { sw::Cigar cigar("4294967296M"); });
    for (auto backend : {sw::Backend::scalar, sw::Backend::cuda}) {
      CHECK(*sw::unavailable_reason(backend));
      expect_throw<std::runtime_error>([&] { sw::make_aligner(backend); });
    }
    const auto unknown = static_cast<sw::Backend>(999);
    CHECK(std::string(sw::backend_name(unknown)) == "unknown");
    expect_throw<std::runtime_error>([&] { sw::make_aligner(unknown); });
    if (*sw::unavailable_reason(sw::Backend::striped))
      expect_throw<std::runtime_error>([] { sw::make_aligner(); });
    std::cout << "API, types and unavailable backend contracts passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
