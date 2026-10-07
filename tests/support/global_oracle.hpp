#pragma once
#include <sw/alignment.hpp>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>
namespace test {
// Independent 64-bit affine-gap DP, for test validation only (not a backend).
inline std::int64_t global_score(std::string_view ref, std::string_view query, sw::Scoring p) {
  const auto neg = std::numeric_limits<std::int64_t>::min() / 4;
  const auto gap = [p](std::size_t n) -> std::int64_t {
    return n ? p.gap_open + (static_cast<std::int64_t>(n) - 1) * p.gap_extend : 0;
  };
  if (ref.empty()) return gap(query.size());
  if (query.empty()) return gap(ref.size());
  std::vector<std::int64_t> previous(query.size()+1), current(query.size()+1);
  std::vector<std::int64_t> vertical(query.size()+1, neg);
  for (std::size_t j=0; j<=query.size(); ++j) previous[j] = gap(j);
  for (std::size_t i=1; i<=ref.size(); ++i) {
    current[0] = gap(i);
    auto horizontal = neg;
    for (std::size_t j=1; j<=query.size(); ++j) {
      vertical[j] = std::max(previous[j]+p.gap_open, vertical[j]+p.gap_extend);
      horizontal = std::max(current[j-1]+p.gap_open, horizontal+p.gap_extend);
      current[j] = std::max({previous[j-1]+(ref[i-1]==query[j-1] ? p.match : p.mismatch),
                             vertical[j], horizontal});
    }
    previous.swap(current);
  }
  return previous.back();
}
} // namespace test
