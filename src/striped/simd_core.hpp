#pragma once

#include <cassert>
#include <sw/sw.hpp>
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

    // match = M + bias; mismatch = -X + bias
    auto bias = IGO + IGE + DGO + DGE;

    // A=0, C=1, G=2, T=3, N=4; initially all mismatch.
    // [ A block ][ C block ][ G block ][ T block ][ N block ]
    // Each block: segn vectors = segn * SIMD_WIDTH bytes.
    auto s = SIMDVec(5 * segn, make_int(-X + bias));
    auto p = (int8_t*)s.data();

    // Striped layout: n=32, SIMD_WIDTH=16, segn=2
    //                lane 0  lane 1  lane 2  ...  lane 15
    // vector i=0:      q0      q2      q4    ...    q30
    // vector i=1:      q1      q3      q5    ...    q31
    // ptr:          0..15 (vector 0), 16..31 (vector 1)
    for (int i = 0, ptr = 0; i < segn; ++i) {
      for (int j = i; j < segn * SIMD_WIDTH; j += segn) {
        // que[j]=G: update only G at ptr (b = bias).
        // block:    A      C      G      T      N
        // score:  -X+b   -X+b    M+b   -X+b   -X+b
        //                        ^
        // index = block start + byte offset
        if (j < n)
          p[que[j] * segn * SIMD_WIDTH + ptr] = M + bias;
        ++ptr; // Padding stays mismatch.
      }
    }

    // DP lookup: profile.data() + ref[row] * segn
    // query G + reference G -> match; other reference bases -> mismatch.
    return s;
  }

}

