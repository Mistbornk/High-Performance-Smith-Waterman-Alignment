#include "test.hpp"
#include <sw/io.hpp>
#include <iostream>
#include <random>
#include <vector>

namespace {
sw::io::SequenceSet random_set(std::mt19937& rng, const std::vector<std::size_t>& lengths) {
  sw::io::SequenceSet set;
  for (std::size_t i = 0; i < lengths.size(); ++i) {
    sw::Sequence sequence(lengths[i], 0);
    for (auto& base : sequence) base = static_cast<sw::Base>(rng() % 5);
    set.add("s" + std::to_string(i), sequence);
  }
  return set;
}

// Every task's score must equal sw::scalar's, for every register tile width, on one reused Batch.
void check_workload(const sw::Workload& workload, sw::Scoring p) {
  sw::cuda::Batch batch(workload);
  std::vector<std::int32_t> scores(3);  // Stale content must be replaced.
  for (int regs : {0, 1, 2, 4, 8, 16}) {
    batch.score(p, scores, regs);
    CHECK(scores.size() == workload.size());
    for (std::size_t i = 0; i < workload.size(); ++i)
      CHECK(scores[i] == sw::scalar::score(workload.reference(i), workload.query(i), p));
  }
}
} // namespace

int main() {
  if (const auto reason = sw::cuda::unavailable_reason(); *reason) {
    std::cout << "cuda unavailable: " << reason << '\n';
    return 77;
  }
  try {
    std::mt19937 rng(2026);
    // Lengths around the stripe (32 * regs columns) and 32-row prefetch boundaries, plus empties.
    const std::vector<std::size_t> edges{0, 1, 2, 31, 32, 33, 63, 64, 65, 100, 257, 511, 512, 513};
    std::vector<std::size_t> refs, queries;
    for (auto n : edges)
      for (auto m : edges) refs.push_back(n), queries.push_back(m);
    for (const auto p : {sw::Scoring{}, sw::Scoring{3, -1, -4, -3}, sw::Scoring{2, -3, -5, -2},
                         sw::Scoring{200, -150, -260, -11}, sw::Scoring{2, 0, -3, -3},
                         sw::Scoring{1, -1, -1, -1}, sw::Scoring{5, -4, -20, 0},
                         sw::Scoring{1, -4, -680, -680}, sw::Scoring{500, -700, -170, -1}}) {
      check_workload(sw::Workload(random_set(rng, refs), random_set(rng, queries), sw::Pairing::zip), p);
    }
    // all_vs_all mapping, and more tasks than one launch's warps hold edge scratch for.
    std::vector<std::size_t> mixed;
    for (int i = 0; i < 40; ++i) mixed.push_back(rng() % 300);
    check_workload(sw::Workload(random_set(rng, mixed), random_set(rng, mixed), sw::Pairing::all_vs_all), {});
    // Query lengths in every bucket at once (regs == 0 splits them into launches), and
    // the same set with a forced regs, where order still maps scores back to tasks.
    {
      std::vector<std::size_t> lens, order;
      for (int i = 0; i < 300; ++i) lens.push_back(1 + (i * 37) % 700), order.push_back(rng() % 400);
      check_workload(sw::Workload(random_set(rng, order), random_set(rng, lens), sw::Pairing::zip), {});
    }
    // Long pairs: many stripes and many prefetch blocks.
    check_workload(sw::Workload(random_set(rng, {5000, 3000, 1}), random_set(rng, {4000, 3001, 7000}),
                                sw::Pairing::zip), {});
    // Similar sequences, where long diagonals and gaps dominate.
    const auto base = sw::encode_sequence(std::string(600, 'A') + std::string(300, 'C') + std::string(600, 'G'));
    sw::io::SequenceSet r, q;
    r.add("r", base);
    q.add("q", base.substr(0, 600) + base.substr(900));
    check_workload(sw::Workload(r, q, sw::Pairing::zip), {});
    check_workload(sw::Workload({}, {}, sw::Pairing::zip), {});

    sw::cuda::Batch batch(sw::Workload(r, q, sw::Pairing::zip));
    std::vector<std::int32_t> scores;
    expect_throw<std::invalid_argument>([&] { batch.score({1, -4, -1, -6}, scores); });
    expect_throw<std::length_error>([&] { batch.score({1, -4, INT32_MIN, -1}, scores); });
    expect_throw<std::invalid_argument>([&] { batch.score({}, scores, 3); });
    // Scoring outside the exact range of half precision is rejected, not computed wrongly.
    expect_throw<std::length_error>([&] { batch.score({1, -4, -683, -683}, scores); });    // 3 * 684 > 2048
    expect_throw<std::length_error>([&] { batch.score({700, -4, -1, -1}, scores); });      // 3 * 701 > 2048
    expect_throw<std::length_error>([&] { batch.score({1, -2100, -6, -1}, scores); });     // mismatch below -2048
    expect_throw<std::length_error>([&] { batch.score({600, -1600, -1, -1}, scores); });  // only match - mismatch > 2048
    std::cout << "CUDA scores match scalar on zip, all_vs_all, edges and long pairs\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
