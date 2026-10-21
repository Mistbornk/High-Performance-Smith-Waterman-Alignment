#include "backend.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

// Naive (Gotoh) global affine-gap DP, one cell at a time with 32-bit scores.
// Serves as the plain multi-threaded baseline next to striped SIMD and CUDA.
//   E(i,j) = max(S(i-1,j) + open, E(i-1,j) + extend)   deletion, consumes reference
//   F(i,j) = max(S(i,j-1) + open, F(i,j-1) + extend)   insertion, consumes query
//   S(i,j) = max(S(i-1,j-1) + sub(i,j), E(i,j), F(i,j))
// with S(i,0) = gap(i), S(0,j) = gap(j), gap(k) = open + (k-1)*extend.
namespace sw::detail {
const char* scalar_unavailable_reason() noexcept { return ""; }
namespace {
constexpr std::int32_t neg_inf = std::numeric_limits<std::int32_t>::min() / 2;
// Trace byte per cell: bits 0-1 source of S, bit 2 E extends, bit 3 F extends.
enum : std::uint8_t { from_diag = 0, from_e = 1, from_f = 2, e_extends = 4, f_extends = 8 };

class ScalarAligner final : public Aligner {
 public:
  explicit ScalarAligner(Scoring scoring) : p_(scoring) {}
  Backend backend() const noexcept override { return Backend::scalar; }

  std::int32_t score(SequenceView ref, SequenceView query) override {
    validate(ref, query);
    const auto n = ref.size(), m = query.size();
    if (!n) return gap(m);
    if (!m) return gap(n);
    row_.resize(m + 1);
    e_.assign(m + 1, neg_inf);
    for (std::size_t j = 0; j <= m; ++j) row_[j] = gap(j);
    for (std::size_t i = 1; i <= n; ++i) {
      auto diag = row_[0];
      row_[0] = gap(i);
      auto f = neg_inf;
      for (std::size_t j = 1; j <= m; ++j) {
        e_[j] = std::max(row_[j] + p_.gap_open, e_[j] + p_.gap_extend);
        f = std::max(row_[j - 1] + p_.gap_open, f + p_.gap_extend);
        const auto s = std::max({diag + substitution(ref[i - 1], query[j - 1]), e_[j], f});
        diag = row_[j];
        row_[j] = s;
      }
    }
    return row_[m];
  }

  AlignmentResult align(SequenceView ref, SequenceView query) override {
    validate(ref, query);
    const auto n = ref.size(), m = query.size();
    AlignmentResult result{0, {}, 0, n, 0, m};
    if (!n || !m) {
      result.score = gap(n + m);
      if (n) result.cigar.emplace_back(static_cast<std::uint32_t>(n), 'D');
      if (m) result.cigar.emplace_back(static_cast<std::uint32_t>(m), 'I');
      return result;
    }
    if (n > trace_.max_size() / m) throw std::length_error("Traceback allocation size overflow");
    trace_.resize(n * m);
    row_.resize(m + 1);
    e_.assign(m + 1, neg_inf);
    for (std::size_t j = 0; j <= m; ++j) row_[j] = gap(j);
    for (std::size_t i = 1; i <= n; ++i) {
      auto diag = row_[0];
      row_[0] = gap(i);
      auto f = neg_inf;
      auto* t = trace_.data() + (i - 1) * m;
      for (std::size_t j = 1; j <= m; ++j) {
        std::uint8_t flags = 0;
        // Ties prefer opening a new gap, and diagonal > insertion > deletion for S,
        // the same order the striped backtrace uses.
        const auto e_open = row_[j] + p_.gap_open, e_ext = e_[j] + p_.gap_extend;
        if (e_ext > e_open) flags |= e_extends;
        e_[j] = std::max(e_open, e_ext);
        const auto f_open = row_[j - 1] + p_.gap_open, f_ext = f + p_.gap_extend;
        if (f_ext > f_open) flags |= f_extends;
        f = std::max(f_open, f_ext);
        auto s = diag + substitution(ref[i - 1], query[j - 1]);
        if (f > s) s = f, flags |= from_f;
        if (e_[j] > s) s = e_[j], flags = (flags & ~3) | from_e;
        t[j - 1] = flags;
        diag = row_[j];
        row_[j] = s;
      }
    }
    result.score = row_[m];
    result.cigar = backtrace(n, m);
    return result;
  }

 private:
  std::int32_t gap(std::size_t k) const {
    return k ? p_.gap_open + static_cast<std::int32_t>(k - 1) * p_.gap_extend : 0;
  }
  std::int32_t substitution(Base a, Base b) const { return a == b ? p_.match : p_.mismatch; }

  void validate(SequenceView ref, SequenceView query) const {
    for (auto sequence : {ref, query})
      for (Base base : sequence)
        if (base < 0 || base > 4) throw std::invalid_argument("Encoded DNA must contain 0..4");
    // Every DP value is bounded by (n + m) * max|parameter|; keep it inside int32
    // with headroom for the neg_inf sentinel arithmetic.
    const std::int64_t magnitude = std::max({std::int64_t(p_.match), -std::int64_t(p_.mismatch),
        -std::int64_t(p_.gap_open), -std::int64_t(p_.gap_extend), std::int64_t(1)});
    const auto limit = static_cast<std::size_t>(INT32_MAX / (4 * magnitude));
    if (ref.size() > limit || query.size() > limit - ref.size())
      throw std::length_error("Alignment exceeds scalar score range");
  }

  Cigar backtrace(std::size_t i, std::size_t j) const {
    const auto m = j;
    Cigar cigar;
    auto push = [&cigar](char op) {
      if (!cigar.empty() && cigar.back().op == op) ++cigar.back().size;
      else cigar.emplace_back(1, op);
    };
    enum { in_s, in_e, in_f } state = in_s;
    while (i > 0 && j > 0) {
      const auto flags = trace_[(i - 1) * m + (j - 1)];
      if (state == in_s) {
        const auto source = flags & 3;
        if (source == from_diag) { push('M'); --i; --j; }
        else state = source == from_e ? in_e : in_f;
      } else if (state == in_e) {
        push('D');
        state = flags & e_extends ? in_e : in_s;
        --i;
      } else {
        push('I');
        state = flags & f_extends ? in_f : in_s;
        --j;
      }
    }
    for (; i > 0; --i) push('D');
    for (; j > 0; --j) push('I');
    std::reverse(cigar.begin(), cigar.end());
    return cigar;
  }

  Scoring p_;
  std::vector<std::int32_t> row_, e_;
  std::vector<std::uint8_t> trace_;
};
} // namespace

std::unique_ptr<Aligner> make_scalar(Scoring scoring) {
  // gap_open <= gap_extend keeps adjacent gap runs from scoring better when a CIGAR
  // merges them, so the CIGAR always replays to the DP score.
  if (scoring.gap_extend > 0 || scoring.gap_open > scoring.gap_extend)
    throw std::invalid_argument("Requires gap_open<=gap_extend<=0");
  return std::make_unique<ScalarAligner>(scoring);
}
} // namespace sw::detail
