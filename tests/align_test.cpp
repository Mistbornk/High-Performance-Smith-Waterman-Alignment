#include "test.hpp"
#include <atomic>
#include <future>
#include <iostream>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace {
std::string random_dna(std::mt19937& rng, std::size_t length) {
  static const std::string bases = "ACGTN";
  std::string text;
  for (std::size_t i = 0; i < length; ++i) text += bases[rng() % 5];
  return text;
}

void check_types() {
  static_assert(sw::encode_base('t') == 3 && sw::complement_base(4) == 4);
  static_assert(std::is_trivially_copyable<sw::Scoring>::value);
  const auto sequence = sw::encode_sequence("aCgTn");
  CHECK(sequence.size() == 5);
  for (int i = 0; i < 5; ++i) CHECK(sequence[i] == i);
  expect_throw<std::invalid_argument>([] { sw::encode_sequence("AX"); });
  CHECK(sw::Cigar("3M1I2D").to_string() == "3M1I2D");
  for (const auto bad : {"0M", "1X", "M", "1", "1M0", "1M2"})
    expect_throw<std::invalid_argument>([&] { sw::Cigar cigar(bad); });
  expect_throw<std::length_error>([] { sw::Cigar cigar("4294967296M"); });
}

// Behaviour every backend shares: fixed examples, input validation, random oracle checks.
void check_backend(const test::Backend& b, std::vector<sw::Scoring> scorings, unsigned seed) {
  struct Example { const char* ref; const char* query; int score; const char* cigar; };
  for (const auto x : {Example{"ACGT", "ACGT", 4, "4M"}, {"ACGT", "AGGT", -1, "4M"},
                       {"ACGT", "ACTGT", -2, "2M1I2M"}, {"ACTGT", "ACGT", -2, "2M1D2M"},
                       {"", "ACG", -8, "3I"}, {"ACG", "", -8, "3D"}, {"", "", 0, ""}}) {
    const auto result = b.align(sw::encode_sequence(x.ref), sw::encode_sequence(x.query), {});
    CHECK(result.score == x.score);
    CHECK(result.cigar.to_string() == x.cigar);
    test::check_alignment(b, x.ref, x.query, {});
  }
  CHECK(b.score(sw::encode_sequence("aCgTn"), sw::encode_sequence("ACGTN"), {}) == 5);
  CHECK(b.score(sw::encode_sequence("N"), sw::encode_sequence("A"), {}) == -4);
  expect_throw<std::invalid_argument>([&] { b.align(sw::Sequence{sw::Base{-1}}, {}, {}); });
  expect_throw<std::invalid_argument>([&] { b.score({}, sw::Sequence{sw::Base{5}}, {}); });
  expect_throw<std::invalid_argument>([&] { b.score({}, {}, {1, -4, -1, -6}); });
  expect_throw<std::length_error>([&] { b.score(sw::encode_sequence("A"), {}, {1, -4, INT32_MIN, -1}); });

  std::mt19937 rng(seed);
  for (const auto p : scorings) {
    for (const auto length : {0u, 1u, 2u, 15u, 16u, 17u, 31u, 32u, 33u, 63u, 65u, 129u}) {
      for (unsigned n = 0; n < 8; ++n) {
        const auto ref = random_dna(rng, length), query = random_dna(rng, length + n);
        test::check_alignment(b, ref, query, p);
        test::check_alignment(b, query, ref, p);
      }
    }
    test::check_alignment(b, std::string(20, 'A') + std::string(160, 'C') + std::string(20, 'G'),
                          std::string(20, 'A') + std::string(20, 'G'), p);
    test::check_alignment(b, std::string(513, 'A'), std::string(513, 'A'), p);
    test::check_alignment(b, "ACGT", "ACTGT", p);  // Scratch shrinks after long input.
  }
  // Per-thread scratch: concurrent callers must not interfere.
  std::vector<std::future<void>> workers;
  for (int i = 0; i < 4; ++i) workers.push_back(std::async(std::launch::async, [&b] {
    for (int j = 0; j < 20; ++j) test::check_alignment(b, "ACGTNACGT", "ACGTACGT", {});
  }));
  for (auto& worker : workers) worker.get();
}

void check_parallel_for() {
  for (unsigned threads : {1u, 3u, 64u}) {
    for (std::size_t n : {0u, 1u, 15u, 16u, 17u, 1000u}) {
      std::vector<std::atomic<int>> hits(n);
      sw::parallel_for(n, threads, [&](std::size_t i) { ++hits[i]; });
      for (auto& hit : hits) CHECK(hit == 1);
    }
  }
  std::atomic<int> calls{0};
  expect_throw<std::runtime_error>([&] {
    sw::parallel_for(10000, 4, [&](std::size_t i) {
      ++calls;
      if (i == 100) throw std::runtime_error("stop");
    });
  });
  CHECK(calls < 10000);  // The error stops the remaining chunks.
}
} // namespace

int main() {
  try {
    check_types();
    check_parallel_for();
    // Scalar uses 32-bit cells, so it also takes scoring the striped byte kernel rejects.
    const test::Backend scalar{"scalar", sw::scalar::score, sw::scalar::align};
    check_backend(scalar, {sw::Scoring{}, {3, -1, -4, -3}, {63, -64, -32, -1},
                           {200, -150, -260, -11}, {2, 0, -3, -3}}, 4242);
    expect_throw<std::invalid_argument>([] { sw::scalar::score({}, {}, {1, -4, -6, 1}); });
#if SW_HAVE_STRIPED
    const test::Backend striped{"striped", sw::striped::score, sw::striped::align};
    check_backend(striped, {sw::Scoring{}, {3, -1, -4, -3}, {10, -15, -30, -5}, {63, -64, -32, -1}}, 787);
    expect_throw<std::length_error>([] { sw::striped::score({}, {}, {200, -150, -260, -11}); });
    // Cross-check on longer random pairs where both accept the scoring.
    std::mt19937 rng(99);
    for (int k = 0; k < 200; ++k) {
      const auto ref = sw::encode_sequence(random_dna(rng, rng() % 300));
      const auto query = sw::encode_sequence(random_dna(rng, rng() % 300));
      CHECK(sw::striped::score(ref, query, {3, -1, -4, -3}) == sw::scalar::score(ref, query, {3, -1, -4, -3}));
    }
#endif
    std::cout << "Types, parallel_for, scalar" << (SW_HAVE_STRIPED ? " and striped" : "")
              << " backends passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
