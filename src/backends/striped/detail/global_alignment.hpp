#pragma once

#include <sw/sequence.hpp>
#include <sw/cigar.hpp>

#include <tuple>
#include <iomanip>
#include <iostream>
#include <vector>
#include <bitset>
#include <limits>

#include <simdpp/simd.h>
#include "simd_core.hpp"

namespace simdpp::SIMDPP_ARCH_NAMESPACE {
  auto
  global_backtrace(auto rn, auto sn, auto &d, auto ans_i, auto ans_j) {
    auto segn = (sn - 1 + SIMD_WIDTH) / SIMD_WIDTH;

    auto cigar = sw::Cigar{};
    auto push = [&cigar](int sz, char op) {
      if (cigar.size() and cigar.back().op == op)
        cigar.back().size += sz;
      else 
        cigar.emplace_back(sz, op);
    };

    if (rn - ans_i > 0) push(rn - ans_i, 'D');
    if (sn - ans_j > 0) push(sn - ans_j, 'I');

    auto ptr = (uint8_t*)d.data();
    int state = 0; // 0: S, 1: E, 2: F
    while (ans_i > 0 and ans_j > 0) {
      size_t idx = (ans_i - 1) * segn * SIMD_WIDTH + (ans_j - 1) % segn * SIMD_WIDTH + (ans_j - 1) / segn;
      if (state == 0) {
        if (ptr[idx] & MATCH_MASK) {
          ans_i--;
          ans_j--;
          push(1, 'M');
        } else if (ptr[idx] & INSERT_MASK) {
          ans_j--;
          push(1, 'I');
          state = 2;
        } else if (ptr[idx] & DELETE_MASK) {
          ans_i--;
          push(1, 'D');
          state = 1;
        }
      } else if (state == 1) {
        if (ptr[idx] & DELETE_EXT_MASK) {
          ans_i--;
          push(1, 'D');
        } else {
          state = 0;
        }
      } else if (state == 2) {
        if (ptr[idx] & INSERT_EXT_MASK) {
          ans_j--;
          push(1, 'I');
        } else {
          state = 0;
        }
      }
    }

    if (ans_i > 0) push(ans_i, 'D');
    if (ans_j > 0) push(ans_j, 'I');

    std::reverse(cigar.begin(), cigar.end());
    return cigar;
  }

  template <
    bool REF_BEG_GAP_IS_PENALIZED = 1, // 初始值
    bool QUE_BEG_GAP_IS_PENALIZED = 1, // 初始值
    bool REF_END_GAP_IS_PENALIZED = 1, // 影響我要看最後一個 row
    bool QUE_END_GAP_IS_PENALIZED = 1  //     我要看最後一個 col
  >
  auto
  simd_global_align(
    sw::SequenceView ref,
    sw::SequenceView que,
    const uint8_t M,
    const uint8_t X,
    const uint8_t IGO,
    const uint8_t IGE,
    const uint8_t DGO,
    const uint8_t DGE,
    auto &d
  ) {
    size_t rn = ref.size(), qn = que.size();

    auto gap_penalty = [](int gap_size, auto GO, auto GE) {
      return gap_size ? -GO - gap_size * GE : 0;
    };
    
    if (rn == 0) {
      if constexpr (REF_BEG_GAP_IS_PENALIZED and REF_END_GAP_IS_PENALIZED) {
        return std::tuple{
          gap_penalty(qn, IGO, IGE),
          qn ? sw::Cigar{std::to_string(qn) + 'I'} : sw::Cigar{}
        };
      } else {
        return std::tuple{
          0,
          qn ? sw::Cigar{std::to_string(qn) + 'I'} : sw::Cigar{}
        };
      }
    }

    if (qn == 0) {
      if constexpr (QUE_BEG_GAP_IS_PENALIZED and QUE_END_GAP_IS_PENALIZED) {
        return std::tuple{
          gap_penalty(rn, DGO, DGE),
          rn ? sw::Cigar{std::to_string(rn) + 'D'} : sw::Cigar{}
        };
      } else {
        return std::tuple{
          0,
          rn ? sw::Cigar{std::to_string(rn) + 'D'} : sw::Cigar{}
        };
      }
    }

    const uint8v vMATCH_MASK       = make_int(MATCH_MASK);
    const uint8v vDELETE_MASK      = make_int(DELETE_MASK);
    const uint8v vINSERT_MASK      = make_int(INSERT_MASK);
    const uint8v vDELETE_EXT_MASK  = make_int(DELETE_EXT_MASK);
    const uint8v vINSERT_EXT_MASK  = make_int(INSERT_EXT_MASK);
    const uint8v vZero = make_int(0);
    const uint8v vIGE = make_int(IGE);
    const uint8v vIGO = make_int(IGO);
    // const uint8v vDGE = make_int(DGE);
    const uint8v vDGO = make_int(DGO);

    // # of segement for the striped parallel strategy
    auto segn = (qn + SIMD_WIDTH - 1) / SIMD_WIDTH;

    // allocate d memory and clear
    if (d.size() < segn * rn)
      d.resize(segn * rn);
    auto d_ptr = size_t{};

    // initialize the memory
    auto H = SIMDVec{};
    if constexpr (REF_BEG_GAP_IS_PENALIZED) {
      H.assign(segn, vIGO);
      H[0] = insert<0>(H[0], 0);
    } else {
      H.assign(segn, vIGE + vIGO);
    }
    auto E = H;
    auto nxtH = SIMDVec(segn);
    auto nxtE = SIMDVec(segn);

    int lastColS = REF_BEG_GAP_IS_PENALIZED ? gap_penalty(qn, IGO, IGE) : 0;
    auto nxtV = SIMDVec{};
    if constexpr (not QUE_END_GAP_IS_PENALIZED) {
      nxtV.assign(segn, vZero);
    }

    // maximum value
    int ans = std::numeric_limits<int>::min();
    size_t ans_i = 0, ans_j = 0;
    auto update_ans = [&ans, &ans_i, &ans_j](int v, int i, int j) {
      if (ans < v)
        ans = v, ans_i = i, ans_j = j;
    };

    auto profile = get_profile(que, M, X, IGO, IGE, DGO, DGE);
    for (size_t i = 0; i < rn; i++) {
      auto *p = profile.data() + ref[i] * segn;

      auto v_init = DGO + DGE;
      if constexpr (QUE_BEG_GAP_IS_PENALIZED) {
        v_init = i == 0 ? 0 : DGO;
      }

      uint8v a = max(p[segn - 1], E[segn - 1]);
      uint8v v = a - H[segn - 1];
      v = move_r(v, v_init);
      uint8v f = v;


      // que:    A   T   T   G   C
      // p: A    M  -X  -X  -X  -X
      // p: C   -X  -X  -X  -X   M
      // p: G   -X  -X  -X   M  -X
      // p: T   -X   M   M  -X  -X

      while (1) {
        auto pref = f;
        auto prev = v;
        for (size_t j = 0; j < segn; j++) { // p: que
          uint8v A = max(p[j], E[j]);
                 A = max(   A,    f);
          uint8v hij = A - v;
          uint8v vij = A - H[j];
          uint8v eij = max(A, E[j] + vDGO) - v;
          uint8v fij = max(A,    f + vIGO) - H[j];

          uint8v d_mask = vZero;
          d_mask = d_mask | ((vij + H[j] == p[j]) & vMATCH_MASK);
          d_mask = d_mask | ((vij + H[j] == E[j]) & vDELETE_MASK);
          d_mask = d_mask | ((vij + H[j] ==    f) & vINSERT_MASK);
          d_mask = d_mask | ((eij        !=  hij) & vDELETE_EXT_MASK);
          d_mask = d_mask | ((fij        !=  vij) & vINSERT_EXT_MASK);
          d[d_ptr++] = d_mask;

          nxtH[j] = hij;
          nxtE[j] = eij;
          v = vij;
          f = fij;
          if constexpr (not QUE_END_GAP_IS_PENALIZED) {
            nxtV[j] = vij;
          }
        }

        v = move_r(v, v_init);
        f = move_r(f, v_init);

        if (not reduce_or((f - pref) | (v - prev)))
          break;

        d_ptr -= segn;
      }

      std::swap(H, nxtH);
      std::swap(E, nxtE);

      if constexpr (not QUE_END_GAP_IS_PENALIZED) {
        update_ans(lastColS, i, qn);
        lastColS += ((int8_t*)&(nxtV[(qn - 1) % segn]))[(qn - 1) / segn] - DGO - DGE;
      }
    }

    int lastRowS = QUE_BEG_GAP_IS_PENALIZED ? gap_penalty(rn, DGO, DGE) : 0;
    auto *p = (uint8_t*)H.data();
    for (size_t i = 0, ptr = 0; i < SIMD_WIDTH and ptr < qn; i++) {
      for (size_t j = i; j < segn * SIMD_WIDTH and ptr < qn; j += SIMD_WIDTH) {
        if constexpr (not REF_END_GAP_IS_PENALIZED)
          update_ans(lastRowS, rn, ptr);
        lastRowS += p[j] - IGO - IGE;
        ptr++;
      }
    }

    update_ans(lastRowS, rn, qn);

    auto cigar = global_backtrace(rn, qn, d, ans_i, ans_j);
    return std::tuple{ ans, cigar };
  }

} // namespace SIMDPP_ARCH_NAMESPACE

namespace sw {
template <
  bool REF_BEG_GAP_IS_PENALIZED = 1,
  bool QUE_BEG_GAP_IS_PENALIZED = 1,
  bool REF_END_GAP_IS_PENALIZED = 1,
  bool QUE_END_GAP_IS_PENALIZED = 1
>
struct GlobalAlignmentStriped {
  const uint8_t MATCH_SCORE               = 1;
  const uint8_t MISMATCH_PENALTY          = 4;
  const uint8_t INSERT_GAP_OPEN_PENALTY   = 6;
  const uint8_t INSERT_GAP_EXTEND_PENALTY = 1;
  const uint8_t DELETE_GAP_OPEN_PENALTY   = 6;
  const uint8_t DELETE_GAP_EXTEND_PENALTY = 1;

  auto
  align(sw::SequenceView ref, sw::SequenceView que) {
    auto d = simdpp::SIMDPP_ARCH_NAMESPACE::SIMDVec{};
    auto [score, cigar] = simdpp::SIMDPP_ARCH_NAMESPACE::simd_global_align<
                    REF_BEG_GAP_IS_PENALIZED,
                    QUE_BEG_GAP_IS_PENALIZED,
                    REF_END_GAP_IS_PENALIZED,
                    QUE_END_GAP_IS_PENALIZED
                  >(
                    ref, que,
                    MATCH_SCORE,
                    MISMATCH_PENALTY,
                    INSERT_GAP_OPEN_PENALTY - INSERT_GAP_EXTEND_PENALTY,
                    INSERT_GAP_EXTEND_PENALTY,
                    DELETE_GAP_OPEN_PENALTY - DELETE_GAP_EXTEND_PENALTY,
                    DELETE_GAP_EXTEND_PENALTY,
                    d
                  );
    return std::tuple{score, cigar};
  }

  auto
  align(sw::SequenceView ref, sw::SequenceView que, auto &d) {
    auto [score, cigar] = simdpp::SIMDPP_ARCH_NAMESPACE::simd_global_align<
                    REF_BEG_GAP_IS_PENALIZED,
                    QUE_BEG_GAP_IS_PENALIZED,
                    REF_END_GAP_IS_PENALIZED,
                    QUE_END_GAP_IS_PENALIZED
                  >(
                    ref, que,
                    MATCH_SCORE,
                    MISMATCH_PENALTY,
                    INSERT_GAP_OPEN_PENALTY - INSERT_GAP_EXTEND_PENALTY,
                    INSERT_GAP_EXTEND_PENALTY,
                    DELETE_GAP_OPEN_PENALTY - DELETE_GAP_EXTEND_PENALTY,
                    DELETE_GAP_EXTEND_PENALTY,
                    d
                  );
    return std::tuple{score, cigar};
  }
};
}
