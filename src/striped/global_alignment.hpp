#pragma once

#include <sw/sw.hpp>

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
    bool REF_BEG_GAP_IS_PENALIZED = 1, // Top boundary
    bool QUE_BEG_GAP_IS_PENALIZED = 1, // Left boundary
    bool REF_END_GAP_IS_PENALIZED = 1, // Last-row endpoint
    bool QUE_END_GAP_IS_PENALIZED = 1, // Last-column endpoint
    bool TRACEBACK = 1                 // 0: score only, d is untouched
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

    // Core GO = public open penalty - GE; gap(k) = -GO - k*GE.
    auto gap_penalty = [](int gap_size, auto GO, auto GE) {
      return gap_size ? -GO - gap_size * GE : 0;
    };
    
    // Empty input: ref empty -> all I; query empty -> all D.
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

    // Broadcast constants to every lane.
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

    // Query layout (segn=2):
    //          lane 0  lane 1  lane 2  ...
    // vec 0:     q0      q2      q4    ...
    // vec 1:     q1      q3      q5    ...
    auto segn = (qn + SIMD_WIDTH - 1) / SIMD_WIDTH;

    // Reuse trace; each cell has one byte of flags.
    // d: [ref row 0: segn vectors][ref row 1: segn vectors]...
    // Grow if needed; active entries are overwritten below.
    if constexpr (TRACEBACK) {
      if (d.size() < segn * rn)
        d.resize(segn * rn);
    }
    auto d_ptr = size_t{};

    // Logical DP grid (S = absolute score; E/F = gap scores):
    //                         query j ->
    //              S(i-1,j-1)          S(i-1,j)
    //                        \            | E: deletion
    // ref i                   \ diagonal  v
    //   |          S(i,j-1) ----------> S(i,j)
    //   v                    F: insertion
    //
    // E(i,j) = max(S(i-1,j) + gap_open, E(i-1,j) + gap_extend)
    // F(i,j) = max(S(i,j-1) + gap_open, F(i,j-1) + gap_extend)
    // S(i,j) = max(diagonal + substitution, E(i,j), F(i,j))
    //
    // Stored bytes (OI=IGO+IGE, OD=DGO+DGE):
    // H/hij = S(i,j) - S(i,j-1) + OI  (horizontal difference)
    // v/vij = S(i,j) - S(i-1,j) + OD  (vertical difference)
    // E/eij and f/fij encode gap candidates relative to nearby S.
    // [H, E] --compute row--> [nxtH, nxtE] --swap--> [H, E]
    // Initial H: horizontal differences along the top row (ref empty).
    // Example: IGO=5, IGE=1, OI=6.
    auto H = SIMDVec{};
    if constexpr (REF_BEG_GAP_IS_PENALIZED) {
      //                       query ->
      //                  empty    A     C     G
      // ref empty, S:   [  0  ][ -6 ][ -7 ][ -8 ]
      //                     -- -6 ->-- -1 ->-- -1 ->
      // H = difference + 6:      0     5     5
      //
      // Striped H (segn=2):
      //          lane 0  lane 1  lane 2  ...
      // H[0]:      0       5       5    ...  (q0, q2, q4, ...)
      // H[1]:      5       5       5    ...  (q1, q3, q5, ...)
      H.assign(segn, vIGO);       // Extension: -IGE + OI = IGO.
      H[0] = insert<0>(H[0], 0); // First gap: -OI + OI = 0.
    } else {
      // Free start:
      // S:         [0] -- 0 --> [0] -- 0 --> [0] ...
      // H lanes:              [6,            6, ...] = 0 + OI
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

    // Best endpoint: (score, ref position, query position).
    int ans = std::numeric_limits<int>::min();
    size_t ans_i = 0, ans_j = 0;
    auto update_ans = [&ans, &ans_i, &ans_j](int v, int i, int j) {
      if (ans < v)
        ans = v, ans_i = i, ans_j = j;
    };

    // Build once; select one base block per reference row.
    auto profile = get_profile(que, M, X, IGO, IGE, DGO, DGE);
    for (size_t i = 0; i < rn; i++) {
      auto *p = profile.data() + ref[i] * segn;

      // Left-edge seed for lane 0.
      auto v_init = DGO + DGE;
      if constexpr (QUE_BEG_GAP_IS_PENALIZED) {
        v_init = i == 0 ? 0 : DGO;
      }

      // Initial carry estimate (segn=2):
      //                  lane 0       lane 1       lane 2
      // last vec:          q1           q3           q5
      //                       \            \            \ ...
      //                        \            \            \ ...
      //                         \            \            \ ...
      // move_r:       [ v_init ][ q1 carry ][ q3 carry ] ...
      //                    |            |            |
      // first vec:         q0           q2           q4
      //                              ^ lane 0 tail feeds lane 1 head
      uint8v a = max(p[segn - 1], E[segn - 1]);
      uint8v v = a - H[segn - 1];
      v = move_r(v, v_init);
      uint8v f = v;


      // Scan -> shift lane carries -> repeat until unchanged.

      while (1) {
        auto pref = f;
        auto prev = v;
        for (size_t j = 0; j < segn; j++) {
          // Normalize all candidates by -S(i-1,j-1) + OI + OD:
          // diagonal -> p[j]; deletion -> E[j]; insertion -> f.
          //                 p[j]   E[j]   f
          //                   \     |    /
          //                    max = A
          // A = S(i,j) - S(i-1,j-1) + OI + OD.
          uint8v A = max(p[j], E[j]);
                 A = max(   A,    f);
          // Update differences; gaps choose open vs. extend.
          uint8v hij = A - v;
          uint8v vij = A - H[j];
          uint8v eij = max(A, E[j] + vDGO) - v;
          uint8v fij = max(A,    f + vIGO) - H[j];

          // Trace byte: [unused | I-ext | D-ext | I | D | diagonal]
          // Ties may set multiple direction bits.
          if constexpr (TRACEBACK) {
            uint8v d_mask = vZero;
            d_mask = d_mask | ((vij + H[j] == p[j]) & vMATCH_MASK);
            d_mask = d_mask | ((vij + H[j] == E[j]) & vDELETE_MASK);
            d_mask = d_mask | ((vij + H[j] ==    f) & vINSERT_MASK);
            d_mask = d_mask | ((eij        !=  hij) & vDELETE_EXT_MASK);
            d_mask = d_mask | ((fij        !=  vij) & vINSERT_EXT_MASK);
            d[d_ptr++] = d_mask;
          }

          nxtH[j] = hij;
          nxtE[j] = eij;
          v = vij;
          f = fij;
          if constexpr (not QUE_END_GAP_IS_PENALIZED) {
            nxtV[j] = vij;
          }
        }

        // End of pass 1 (segn=2); both v and f shift:
        // before: [q1 carry][q3 carry][q5 carry]...
        // after:  [v_init  ][q1 carry][q3 carry]...
        // next pass:  q0        q2        q4
        //                      ^ q1 from lane 0 -> q2 in lane 1
        v = move_r(v, v_init);
        f = move_r(f, v_init);

        // No changed lane -> row converged.
        if (not reduce_or((f - pref) | (v - prev)))
          break;

        if constexpr (TRACEBACK)
          d_ptr -= segn; // Retry overwrites this row's trace.
      }

      // Current row becomes the next row's input.
      std::swap(H, nxtH);
      std::swap(E, nxtE);

      // Free query end: consider the right edge.
      if constexpr (not QUE_END_GAP_IS_PENALIZED) {
        update_ans(lastColS, i, qn);
        lastColS += ((int8_t*)&(nxtV[(qn - 1) % segn]))[(qn - 1) / segn] - DGO - DGE;
      }
    }

    // Restore bottom-row scores in query order:
    // striped H: [q0 q2 q4 ...][q1 q3 q5 ...]
    // read:       q0 -> q1 -> q2 -> q3 -> ...
    // score += stored difference - (IGO + IGE)
    int lastRowS = QUE_BEG_GAP_IS_PENALIZED ? gap_penalty(rn, DGO, DGE) : 0;
    auto *p = (uint8_t*)H.data();
    for (size_t i = 0, ptr = 0; i < SIMD_WIDTH and ptr < qn; i++) {
      for (size_t j = i; j < segn * SIMD_WIDTH and ptr < qn; j += SIMD_WIDTH) {
        // Free reference end: consider the bottom edge.
        if constexpr (not REF_END_GAP_IS_PENALIZED)
          update_ans(lastRowS, rn, ptr);
        lastRowS += p[j] - IGO - IGE;
        ptr++;
      }
    }

    update_ans(lastRowS, rn, qn); // Bottom-right endpoint.

    // Endpoint --trace flags--> reversed path --reverse--> CIGAR.
    if constexpr (not TRACEBACK)
      return std::tuple{ ans, sw::Cigar{} };
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

  // Same DP without trace writes or backtrace; only the score is computed.
  int
  score(sw::SequenceView ref, sw::SequenceView que) {
    auto unused = simdpp::SIMDPP_ARCH_NAMESPACE::SIMDVec{};
    auto [score, cigar] = simdpp::SIMDPP_ARCH_NAMESPACE::simd_global_align<
                    REF_BEG_GAP_IS_PENALIZED,
                    QUE_BEG_GAP_IS_PENALIZED,
                    REF_END_GAP_IS_PENALIZED,
                    QUE_END_GAP_IS_PENALIZED,
                    false
                  >(
                    ref, que,
                    MATCH_SCORE,
                    MISMATCH_PENALTY,
                    INSERT_GAP_OPEN_PENALTY - INSERT_GAP_EXTEND_PENALTY,
                    INSERT_GAP_EXTEND_PENALTY,
                    DELETE_GAP_OPEN_PENALTY - DELETE_GAP_EXTEND_PENALTY,
                    DELETE_GAP_EXTEND_PENALTY,
                    unused
                  );
    return score;
  }
};
}
