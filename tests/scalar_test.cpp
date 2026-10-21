#include "check.hpp"
#include "support/check_alignment.hpp"
#include <iostream>
#include <random>
#include <string>

int main() {
  try {
    CHECK(!*sw::unavailable_reason(sw::Backend::scalar));
    auto aligner = sw::make_aligner(sw::Backend::scalar);
    CHECK(aligner->backend() == sw::Backend::scalar);
    struct Example { const char* ref; const char* query; int score; const char* cigar; };
    for (const auto x : {Example{"ACGT","ACGT",4,"4M"}, {"ACGT","AGGT",-1,"4M"},
                         {"ACGT","ACTGT",-2,"2M1I2M"}, {"ACTGT","ACGT",-2,"2M1D2M"},
                         {"","ACG",-8,"3I"}, {"ACG","",-8,"3D"}, {"","",0,""}}) {
      const auto result = aligner->align(x.ref, x.query);
      CHECK(result.score == x.score);
      CHECK(result.cigar.to_string() == x.cigar);
      CHECK(aligner->score(x.ref, x.query) == x.score);
      check_alignment(*aligner, x.ref, x.query, {});
    }
    CHECK(aligner->align("aCgTn", "ACGTN").score == 5);
    CHECK(aligner->score("N", "A") == -4);
    expect_throw<std::invalid_argument>([&] { aligner->align("AX", "AA"); });
    expect_throw<std::invalid_argument>([&] { aligner->score(sw::Sequence{sw::Base{5}}, sw::Sequence{}); });
    expect_throw<std::invalid_argument>([] { sw::make_aligner(sw::Backend::scalar, {1, -4, -1, -6}); });
    expect_throw<std::invalid_argument>([] { sw::make_aligner(sw::Backend::scalar, {1, -4, -6, 1}); });
    expect_throw<std::length_error>([] {
      sw::make_aligner(sw::Backend::scalar, {1, -4, INT32_MIN, -1})->score("A", "A");
    });

    std::mt19937 rng(4242);
    const std::string bases = "ACGTN";
    // Includes scoring the striped byte kernel rejects; scalar uses 32-bit cells.
    for (const auto p : {sw::Scoring{}, sw::Scoring{3, -1, -4, -3}, sw::Scoring{63, -64, -32, -1},
                         sw::Scoring{200, -150, -260, -11}, sw::Scoring{2, 0, -3, -3}}) {
      auto worker = sw::make_aligner(sw::Backend::scalar, p);
      for (const auto length : {0u, 1u, 2u, 15u, 16u, 17u, 33u, 65u, 129u}) {
        for (unsigned n = 0; n < 8; ++n) {
          std::string ref, query;
          for (unsigned i = 0; i < length; ++i) ref += bases[rng() % 5];
          for (unsigned i = 0; i < length + n; ++i) query += bases[rng() % 5];
          check_alignment(*worker, ref, query, p);
          check_alignment(*worker, query, ref, p);
        }
      }
      check_alignment(*worker, std::string(20, 'A') + std::string(160, 'C') + std::string(20, 'G'),
                      std::string(20, 'A') + std::string(20, 'G'), p);
      check_alignment(*worker, "ACGT", "ACTGT", p);  // Scratch shrinks after long input.
    }
    // Cross-check against striped where both accept the scoring.
    if (!*sw::unavailable_reason(sw::Backend::striped)) {
      auto striped = sw::make_aligner(sw::Backend::striped, {3, -1, -4, -3});
      auto scalar = sw::make_aligner(sw::Backend::scalar, {3, -1, -4, -3});
      for (int k = 0; k < 200; ++k) {
        std::string ref, query;
        for (unsigned i = rng() % 300; i; --i) ref += bases[rng() % 5];
        for (unsigned i = rng() % 300; i; --i) query += bases[rng() % 5];
        CHECK(striped->score(ref, query) == scalar->score(ref, query));
      }
    }
    std::cout << "Scalar oracle, CIGAR replay, score-only and validation passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
