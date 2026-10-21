#include "check.hpp"
#include "support/global_oracle.hpp"
#include <sw/batch.hpp>
#include <iostream>
#include <random>

namespace {
std::string decoded(sw::SequenceView sequence) {
  std::string text;
  for (auto base : sequence) text.push_back(sw::decode_base(base));
  return text;
}
sw::io::SequenceSet random_set(std::mt19937& rng, std::size_t count, unsigned max_length) {
  sw::io::SequenceSet set;
  for (std::size_t i = 0; i < count; ++i) {
    sw::Sequence sequence(rng() % (max_length + 1), 0);
    for (auto& base : sequence) base = static_cast<sw::Base>(rng() % 5);
    set.add("s" + std::to_string(i), sequence);
  }
  return set;
}
} // namespace

int main() {
  try {
    expect_throw<std::runtime_error>([] { sw::make_batch_aligner(sw::Backend::cuda); });
    expect_throw<std::invalid_argument>([] { sw::make_batch_aligner(sw::Backend::scalar, {}, {0}); });

    std::vector<sw::Backend> backends{sw::Backend::scalar};
    if (!*sw::unavailable_reason(sw::Backend::striped)) backends.push_back(sw::Backend::striped);

    std::mt19937 rng(2026);
    const sw::Scoring scoring{2, -3, -5, -2};
    for (auto pairing : {sw::Pairing::zip, sw::Pairing::all_vs_all}) {
      const std::size_t n = pairing == sw::Pairing::zip ? 300 : 20;
      sw::Workload workload(random_set(rng, n, 90), random_set(rng, n, 90), pairing);
      std::vector<std::int32_t> oracle;
      for (std::size_t i = 0; i < workload.size(); ++i)
        oracle.push_back(static_cast<std::int32_t>(test::global_score(
            decoded(workload.reference(i)), decoded(workload.query(i)), scoring)));

      for (auto backend : backends) {
        auto single = sw::make_aligner(backend, scoring);
        std::vector<sw::AlignmentResult> expected;
        for (std::size_t i = 0; i < workload.size(); ++i)
          expected.push_back(single->align(workload.reference(i), workload.query(i)));
        for (bool traceback : {true, false}) {
          for (unsigned threads : {1u, 3u, 8u, 64u}) {
            auto batch = sw::make_batch_aligner(backend, scoring, {threads, traceback});
            CHECK(batch->backend() == backend && batch->options().threads == threads);
            CHECK(batch->options().traceback == traceback);
            std::vector<sw::AlignmentResult> results(5);  // Stale content must be replaced.
            for (int repeat = 0; repeat < 2; ++repeat) {
              batch->align(workload, results);
              CHECK(results.size() == workload.size());
              for (std::size_t i = 0; i < results.size(); ++i) {
                CHECK(results[i].score == oracle[i]);
                CHECK(results[i].reference_end == workload.reference(i).size());
                CHECK(results[i].query_end == workload.query(i).size());
                if (traceback) CHECK(results[i].cigar.to_string() == expected[i].cigar.to_string());
                else CHECK(results[i].cigar.empty());
              }
            }
          }
        }
      }
    }
    auto batch = sw::make_batch_aligner(sw::Backend::scalar, {}, {4});
    std::vector<sw::AlignmentResult> results(3);
    batch->align(sw::Workload({}, {}, sw::Pairing::zip), results);
    CHECK(results.empty());
    std::cout << "Batch aligner ordering, threading, score-only and validation passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
