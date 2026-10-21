#pragma once
#include "../check.hpp"
#include "global_oracle.hpp"
#include <string>

// Score vs. the independent oracle, full CIGAR replay, encoded-input and score-only
// agreement. Shared by every backend test.
inline void check_alignment(sw::Aligner& aligner, const std::string& ref, const std::string& query, sw::Scoring p) {
  const auto result = aligner.align(ref, query);
  CHECK(result.reference_begin == 0 && result.reference_end == ref.size());
  CHECK(result.query_begin == 0 && result.query_end == query.size());
  CHECK(result.score == test::global_score(ref, query, p));
  std::size_t r=0, q=0;
  std::int64_t score=0;
  for (const auto& item : result.cigar) {
    CHECK(item.size > 0);
    CHECK(item.op == 'M' || item.op == 'I' || item.op == 'D');
    if (item.op != 'I') CHECK(item.size <= ref.size()-r);
    if (item.op != 'D') CHECK(item.size <= query.size()-q);
    if (item.op == 'M') {
      for (std::uint32_t i=0; i<item.size; ++i)
        score += ref[r++] == query[q++] ? p.match : p.mismatch;
    } else {
      score += p.gap_open + (std::int64_t(item.size)-1)*p.gap_extend;
      if (item.op == 'I') q += item.size; else r += item.size;
    }
  }
  CHECK(r == ref.size() && q == query.size());
  CHECK(score == result.score);
  const auto encoded = aligner.align(sw::encode_sequence(ref), sw::encode_sequence(query));
  CHECK(encoded.score == result.score);
  CHECK(encoded.cigar.to_string() == result.cigar.to_string());
  CHECK(aligner.score(ref, query) == result.score);
}
