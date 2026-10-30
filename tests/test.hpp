#pragma once
// Minimal test helpers plus an independent oracle shared by every backend test.
#include <sw/sw.hpp>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

inline void check(bool condition, const char* expression) {
  if (!condition) throw std::runtime_error(std::string("Check failed: ") + expression);
}
#define CHECK(expression) check(static_cast<bool>(expression), #expression)
template <class Exception, class Function> void expect_throw(Function function) {
  try { function(); } catch (const Exception&) { return; }
  throw std::runtime_error("Expected exception was not thrown");
}

namespace test {
// Independent 64-bit affine-gap DP, for test validation only (not a backend).
inline std::int64_t global_score(std::string_view ref, std::string_view query, sw::Scoring p) {
  const auto neg = std::numeric_limits<std::int64_t>::min() / 4;
  const auto gap = [p](std::size_t n) -> std::int64_t {
    return n ? p.gap_open + (static_cast<std::int64_t>(n) - 1) * p.gap_extend : 0;
  };
  if (ref.empty()) return gap(query.size());
  if (query.empty()) return gap(ref.size());
  std::vector<std::int64_t> previous(query.size() + 1), current(query.size() + 1);
  std::vector<std::int64_t> vertical(query.size() + 1, neg);
  for (std::size_t j = 0; j <= query.size(); ++j) previous[j] = gap(j);
  for (std::size_t i = 1; i <= ref.size(); ++i) {
    current[0] = gap(i);
    auto horizontal = neg;
    for (std::size_t j = 1; j <= query.size(); ++j) {
      vertical[j] = std::max(previous[j] + p.gap_open, vertical[j] + p.gap_extend);
      horizontal = std::max(current[j - 1] + p.gap_open, horizontal + p.gap_extend);
      current[j] = std::max({previous[j - 1] + (ref[i - 1] == query[j - 1] ? p.match : p.mismatch),
                             vertical[j], horizontal});
    }
    previous.swap(current);
  }
  return previous.back();
}

// One backend's entry points, so the same checks run against each of them.
struct Backend {
  const char* name;
  std::int32_t (*score)(sw::SequenceView, sw::SequenceView, sw::Scoring);
  sw::Alignment (*align)(sw::SequenceView, sw::SequenceView, sw::Scoring);
};

// Score vs. the oracle, full CIGAR replay, and score-only agreement.
inline void check_alignment(const Backend& backend, const std::string& ref, const std::string& query,
                            sw::Scoring p) {
  const auto r_seq = sw::encode_sequence(ref), q_seq = sw::encode_sequence(query);
  const auto result = backend.align(r_seq, q_seq, p);
  CHECK(result.score == global_score(ref, query, p));
  std::size_t r = 0, q = 0;
  std::int64_t score = 0;
  for (const auto& item : result.cigar) {
    CHECK(item.size > 0);
    CHECK(item.op == 'M' || item.op == 'I' || item.op == 'D');
    if (item.op != 'I') CHECK(item.size <= ref.size() - r);
    if (item.op != 'D') CHECK(item.size <= query.size() - q);
    if (item.op == 'M') {
      for (std::uint32_t i = 0; i < item.size; ++i)
        score += ref[r++] == query[q++] ? p.match : p.mismatch;
    } else {
      score += p.gap_open + (std::int64_t(item.size) - 1) * p.gap_extend;
      if (item.op == 'I') q += item.size; else r += item.size;
    }
  }
  CHECK(r == ref.size() && q == query.size());
  CHECK(score == result.score);
  CHECK(backend.score(r_seq, q_seq, p) == result.score);
}
} // namespace test
