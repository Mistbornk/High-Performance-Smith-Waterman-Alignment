#pragma once

#include <cassert>
#include <sw/sequence.hpp>
#include <array>
#include <vector>
#include <simdpp/simd.h>

namespace simdpp::SIMDPP_ARCH_NAMESPACE {
  constexpr uint8_t SIMD_WIDTH = SIMDPP_FAST_INT8_SIZE;
  using SIMDVec = std::vector<uint8v, aligned_allocator<uint8v, SIMD_WIDTH>>;

  constexpr uint8_t MATCH_OFFSET      = 0;
  constexpr uint8_t DELETE_OFFSET     = 1;
  constexpr uint8_t INSERT_OFFSET     = 2;
  constexpr uint8_t DELETE_EXT_OFFSET = 3;
  constexpr uint8_t INSERT_EXT_OFFSET = 4;

  constexpr uint8_t MATCH_MASK        = 1u << MATCH_OFFSET;
  constexpr uint8_t DELETE_MASK       = 1u << DELETE_OFFSET;
  constexpr uint8_t INSERT_MASK       = 1u << INSERT_OFFSET;
  constexpr uint8_t DELETE_EXT_MASK   = 1u << DELETE_EXT_OFFSET;
  constexpr uint8_t INSERT_EXT_MASK   = 1u << INSERT_EXT_OFFSET;

  uint8v
  move_r(const uint8v &v, uint8_t init = 0u) {
    auto ret = uint8v{};
    if constexpr (SIMD_WIDTH == 16) {
      ret = move16_r<1>(v);
    } else if constexpr (SIMD_WIDTH == 32) {
      auto tmp = extract<15>(v);
      ret = move16_r<1>(v);
      ret = insert<16>(ret, tmp);
    } else assert(false);
    ret = insert<0>(ret, init);
    return ret;
  }

  auto
  get_profile(
    sw::SequenceView que,
    const uint8_t M,
    const uint8_t X,
    const uint8_t IGO,
    const uint8_t IGE,
    const uint8_t DGO,
    const uint8_t DGE
  ) {
    auto n = que.size();
    auto segn = (n - 1 + SIMD_WIDTH) / SIMD_WIDTH;
    
    auto bias = IGO + IGE + DGO + DGE;
    auto s = SIMDVec(5 * segn, make_int(-X + bias));
    auto p = (int8_t*)s.data();
    for (int i = 0, ptr = 0; i < segn; i++) {
      for (int j = i; j < segn * SIMD_WIDTH; j += segn) {
        if (j < n)
          p[que[j] * segn * SIMD_WIDTH + ptr] = M + bias;
        ptr++;
      }
    }
    return s;
  }

  // Build a single row (length=segn) profile where "match" is allowed for
  // multiple query characters.
  //
  // allow_match[c] == true means query char code c is treated as match
  // for this reference row.
  inline SIMDVec
  get_profile_row_multi(
    sw::SequenceView que,
    const std::array<bool, 5>& allow_match,
    const uint8_t M,
    const uint8_t X,
    const uint8_t IGO,
    const uint8_t IGE,
    const uint8_t DGO,
    const uint8_t DGE
  ) {
    const size_t n = que.size();
    const size_t segn = (n - 1 + SIMD_WIDTH) / SIMD_WIDTH;

    const auto bias = IGO + IGE + DGO + DGE;
    SIMDVec row(segn, make_int(-X + bias));
    auto p = (int8_t*)row.data();

    for (size_t i = 0, ptr = 0; i < segn; ++i) {
      for (size_t j = i; j < segn * SIMD_WIDTH; j += segn) {
        if (j < n) {
          const uint8_t qc = que[j];
          if (qc < 5 && allow_match[qc]) {
            p[ptr] = static_cast<int8_t>(M + bias);
          }
        }
        ++ptr;
      }
    }
    return row;
  }
}

