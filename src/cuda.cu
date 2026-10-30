#include <sw/io.hpp>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Ported from the earlier Biovoltron SmithWatermanCuda kernel, changed to global
// alignment and score-only: no trace matrix, no gap-length tracking, no sink search.
namespace sw::cuda {
namespace {
constexpr int warp_size = 32;                // One warp scores one pair.
constexpr int warps_per_block = 8;           // 256 threads per block.
constexpr unsigned full_mask = 0xffffffffu;  // Every lane takes part in each shuffle.

// A stripe is 32 * R query columns wide, so a short query leaves lanes idle when R is
// large. With regs == 0, tasks are grouped by query length and each group runs with its
// own R: the first bucket whose max_query >= m.
struct Bucket {
  std::size_t max_query;
  int regs;
};
constexpr Bucket buckets[] = {{64, 2}, {1024, 8}, {SIZE_MAX, 16}};
constexpr std::size_t bucket_count = sizeof(buckets) / sizeof(buckets[0]);

void check(cudaError_t e, const char* what) {
  if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// Half precision is exact for integers up to 2048. Every value of the difference
// recurrence stays within 3 * (max(match, mismatch, 0) - gap_open) whatever the sequence
// length, so the scoring decides whether half is safe. The substitution score and the
// gain match - mismatch (used by an fma) must fit as well.
bool fits_half(Scoring p) {
  constexpr std::int64_t limit = 2048;
  const std::int64_t top = std::max<std::int64_t>({p.match, p.mismatch, 0});
  return 3 * (top - p.gap_open) <= limit && std::min(p.match, p.mismatch) >= -limit &&
         std::int64_t(p.match) - p.mismatch <= limit && std::int64_t(p.mismatch) - p.match <= limit;
}

// Owning device array. resize() discards contents and only grows the allocation.
template <class T> class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer() { cudaFree(data_); }
  void resize(std::size_t n) {
    if (n > capacity_) {
      cudaFree(data_);
      data_ = nullptr, capacity_ = 0;
      check(cudaMalloc(&data_, n * sizeof(T)), "cudaMalloc");
      capacity_ = n;
    }
  }
  void upload(const T* host, std::size_t n) {
    resize(n);
    if (n) check(cudaMemcpy(data_, host, n * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
  }
  T* data() const noexcept { return data_; }

 private:
  T* data_ = nullptr;
  std::size_t capacity_ = 0;
};

// Device view of a SequenceSet: sequence i is data[offsets[i] .. offsets[i + 1]).
struct DeviceSet {
  const Base* data;
  const std::size_t* offsets;
};

// Score of a gap of length k: open + (k - 1) * extend, and 0 for k = 0. Gives the borders
// S(i,0) = gap(i) and S(0,j) = gap(j), and turns the summed differences back into S(n,m).
__device__ __forceinline__ int gap(int k, Scoring p) {
  return k ? p.gap_open + (k - 1) * p.gap_extend : 0;
}

// Both halves of a half2 hold the same value.
__device__ __forceinline__ __half2 both(int x) { return __half2half2(__int2half_rn(x)); }
// Small integers (|x| <= 2048) are exact in half.
__device__ __forceinline__ __half2 pack(int first, int second) {
  return __halves2half2(__int2half_rn(first), __int2half_rn(second));
}

// One pair of sequences and their lengths.
struct Task {
  const Base* ref;
  const Base* query;
  int n, m;
};

struct PairScores {
  int first, second;
};

// One warp scores TWO pairs at once. Every state variable is a half2 whose .x belongs to
// the first pair and .y to the second, so one instruction (and one shuffle) advances both.
// The pairs follow the same schedule: reference rows 1..n and query columns 1..m with
// n = max(a.n, b.n) and m = max(a.m, b.m). The shorter pair just keeps going on padding,
// which only changes cells below and to the right of its real end, so its score is
// unaffected. The query is cut into stripes of 32 * R columns; lane l owns R consecutive
// columns of a stripe. Lanes sweep anti-diagonals (lane l handles row d - l - 1 at step d).
// The stripe's last column is stored in temp_v/temp_f (one entry per reference row) and
// becomes the next stripe's left edge.
//
// Parameters:
//   R               columns per lane; wider stripes mean fewer shuffles per cell
//   a, b            the two pairs: encoded bases (0..4), n and m long, both >= 1
//   p               match / mismatch / gap_open (a 1-long gap) / gap_extend (each further
//                   gap base); requires gap_open <= gap_extend <= 0 and fits_half(p)
//   temp_v, temp_f  this warp's edge scratch, max(a.n, b.n) half2 each
//
// Half precision: every value below is a small integer, bounded by 3 * (max(match,
// mismatch, 0) - open) however long the sequences are (see fits_half), so half arithmetic
// is exact. Only the final score sums, kept in int, can be large.
//
// Difference recurrence (Suzuki-Kasahara), with the letters of the striped backend: S is
// the absolute score and
//   H(i,j) = S(i,j) - S(i,j-1)      horizontal difference
//   v(i,j) = S(i,j) - S(i-1,j)      vertical difference
//   e(i,j) = E(i,j) - S(i-1,j-1)    vertical gap candidate, measured from the diagonal cell
//   f(i,j) = F(i,j) - S(i-1,j-1)    horizontal gap candidate, measured from the diagonal cell
//   A(i,j) = S(i,j) - S(i-1,j-1)
// stay within a few times the largest |scoring value| however long the sequences are.
//   h[k]    = H(i-1,j)   one per column, from the previous row         (striped H[j])
//   e[k]    = e(i,j)     one per column                                (striped E[j])
//   v_left  = v(i,j-1)   vertical difference, carried along a row      (striped v)
//   f_left  = f(i,j)     horizontal gap candidate, carried along a row (striped f)
// The recurrence of sw::scalar becomes
//   A        = max(sub, e(i,j), f(i,j))
//   v(i,j)   = A - H(i-1,j)                H(i,j) = A - v(i,j-1)
//   e(i+1,j) = max(A + open, e(i,j) + extend) - v(i,j-1)
//   f(i,j+1) = max(A + open, f(i,j) + extend) - H(i-1,j)
// and its boundaries S(i,0) = gap(i), S(0,j) = gap(j), E(0,j) = F(i,0) = -inf become
//   H(0,j) = v(i,0) = open for the first cell and extend after it;
//   e(1,j) = H(0,j) + open;  f(i,1) = v(i,0) + open.
// The score is S(n,m) = gap(m) + sum of v(i,m) over all rows, summed by the lane that owns
// column m.
template <int R>
__device__ PairScores warp_score2(Task a, Task b, Scoring p, __half2* __restrict__ temp_v,
                                  __half2* __restrict__ temp_f) {
  constexpr int stripe = warp_size * R;  // Query columns covered by one pass of the warp.
  const int lane = threadIdx.x & (warp_size - 1);  // 0..31, position inside the warp.
  const int n = max(a.n, b.n), m = max(a.m, b.m);
  const __half2 zero = both(0), open2 = both(p.gap_open), extend2 = both(p.gap_extend);
  const __half2 mismatch2 = both(p.mismatch), gain2 = both(p.match - p.mismatch);
  // The first stripe's left edge is column 0: v(i,0) is open for the first row and extend
  // after it, and f(i,1) = v(i,0) + open.
  for (int row = lane; row < n; row += warp_size) {  // row is 0-based: reference row i = row + 1.
    const int v0 = row ? p.gap_extend : p.gap_open;
    temp_v[row] = both(v0);
    temp_f[row] = both(v0 + p.gap_open);
  }

  __syncwarp();

  // Sum of v(i,m) over the rows, one per pair; only the lane that owns column m of that
  // pair adds to it.
  int result0 = 0, result1 = 0;
  for (int block = 0; block < m; block += stripe) {  // block: columns done in earlier stripes.
    const int lanes = min(warp_size, (m - block + R - 1) / R); // Lanes owning a column <= m.
    const int col0 = block + lane * R + 1;  // My first query column (1-based).
    __half2 q[R], h[R], e[R];  // Per owned column: query base, H(i-1,col), e(i,col).
#pragma unroll
    for (int k = 0; k < R; ++k) {
      const int col = col0 + k;
      // Columns past a pair's end never match (-1 is no base).
      q[k] = pack(col <= a.m ? a.query[col - 1] : -1, col <= b.m ? b.query[col - 1] : -1);
      h[k] = both(col == 1 ? p.gap_open : p.gap_extend);  // H(0, col)
      e[k] = __hadd2(h[k], open2);                         // e(1, col) = H(0, col) + open
    }
    // From the lane on the left (lane 0 takes them from the cache): v(i, col0-1), f(i, col0)
    // and the reference bases r of row i.
    __half2 v_left = zero, f_left = zero, r = zero;
    // My last column's v and f, handed to lane + 1.
    __half2 v_out = zero, f_out = zero;
    __half2 cache_v = zero, cache_f = zero, cache_r = zero;  // Left-edge rows prefetched for 32 steps.

    // base is the first step of a 32-step block; step d puts lane l on row d - l - 1.
    for (int base = 2; base <= lanes + n; base += warp_size) {
      // Every 32 steps each lane prefetches one left-edge row; the values then shift
      // down one lane per step, so lane 0 always holds the row it needs next.
      const int row = base - 2 + lane;
      if (row < n) {
        cache_v = temp_v[row], cache_f = temp_f[row];
        cache_r = pack(row < a.n ? a.ref[row] : -2, row < b.n ? b.ref[row] : -2);  // Rows past the end: -2.
      } else {
        cache_v = zero, cache_f = zero, cache_r = zero;
      }

      for (int d = base; d < base + warp_size; ++d) {
        const int i = d - lane - 1;  // Reference row of this lane at step d.
        if (lane < lanes && i >= 1 && i <= n) {
          if (lane == 0) r = cache_r, v_left = cache_v, f_left = cache_f;
          // v(i, col-1) and f(i, col): start from the left neighbour's values and are
          // updated in place while walking over my R columns from left to right.
          __half2 v = v_left, f = f_left;
#pragma unroll
          for (int k = 0; k < R; ++k) {
            // A = S(i,j) - S(i-1,j-1): best of the diagonal (sub), the vertical gap e and
            // the horizontal gap f. All three are measured from the diagonal cell.
            // sub = r == q ? match : mismatch, for both pairs (__heq2 gives 1.0 or 0.0).
            const __half2 sub = __hfma2(__heq2(r, q[k]), gain2, mismatch2);
            const __half2 A = __hmax2(__hmax2(sub, e[k]), f);
            const __half2 vij = __hsub2(A, h[k]);  // v(i,j) = A - H(i-1,j)
            const __half2 hij = __hsub2(A, v);     // H(i,j) = A - v(i,j-1)
            // Column m of each pair: sum of v(i,m) gives S(n,m) - S(0,m). Rows below a
            // pair's end are padding and must not be counted.
            if (col0 + k == a.m && i <= a.n) result0 += __half2int_rn(__low2half(vij));
            if (col0 + k == b.m && i <= b.n) result1 += __half2int_rn(__high2half(vij));
            // Gap candidates for the next row (e) and the next column (f); each subtracts
            // the difference that shifts its reference cell to the new diagonal cell.
            const __half2 Ao = __hadd2(A, open2);
            const __half2 eij = __hsub2(__hmax2(Ao, __hadd2(e[k], extend2)), v);  // e(i+1,j)
            const __half2 fij = __hsub2(__hmax2(Ao, __hadd2(f, extend2)), h[k]);  // f(i,j+1)
            // Write back only now: all four values above use the old h[k], e[k], v and f.
            h[k] = hij, e[k] = eij, v = vij, f = fij;
          }
          // The last lane saves its last column as the next stripe's left edge.
          if (lane == warp_size - 1) temp_v[i - 1] = v, temp_f[i - 1] = f;
          v_out = v, f_out = f;
        }
        // Hand my last column (and the reference bases) to the lane on the right ...
        r = __shfl_up_sync(full_mask, r, 1);
        v_left = __shfl_up_sync(full_mask, v_out, 1);
        f_left = __shfl_up_sync(full_mask, f_out, 1);
        // ... and move the prefetched left-edge rows one lane down.
        cache_v = __shfl_down_sync(full_mask, cache_v, 1);
        cache_f = __shfl_down_sync(full_mask, cache_f, 1);
        cache_r = __shfl_down_sync(full_mask, cache_r, 1);
      }
    }
    __syncwarp();  // The next stripe reads the edge lane 31 just wrote.
  }
  // S(n,m) = gap(m) + sum of v(i,m); the lane that owned column m of a pair in the last
  // stripe has that pair's sum.
  return {gap(a.m, p) + __shfl_sync(full_mask, result0, ((a.m - 1) % stripe) / R),
          gap(b.m, p) + __shfl_sync(full_mask, result1, ((b.m - 1) % stripe) / R)};
}

// Where task t lives: reference r and query q as in Workload::task(), with their lengths.
__device__ __forceinline__ Task locate(const DeviceSet& refs, const DeviceSet& queries, std::size_t t,
                                       std::size_t reference_count, bool zip) {
  const std::size_t r = zip ? t : t % reference_count, q = zip ? t : t / reference_count;
  return {refs.data + refs.offsets[r], queries.data + queries.offsets[q],
          int(refs.offsets[r + 1] - refs.offsets[r]), int(queries.offsets[q + 1] - queries.offsets[q])};
}

// Grid-stride over the pairs of tasks order[0..count): one warp scores two neighbouring
// tasks order[2g] and order[2g + 1] at once (an odd last task is paired with itself). Each
// warp slot owns 2 * temp_rows half2 of edge scratch, reused for every pair it processes.
// order holds task numbers.
//   refs, queries     device copies of the two sequence sets
//   reference_count   number of references; with all-vs-all it maps a task number to its pair
//   zip               true: task t is (reference t, query t); false: all-vs-all, query-major
//   p                 scoring
//   temp, temp_rows   edge scratch for every warp slot, temp_rows >= the longest reference
//   scores            output, scores[t] is the score of task t
template <int R>
__global__ void score_kernel(DeviceSet refs, DeviceSet queries, const std::size_t* order,
                             std::size_t count, std::size_t reference_count, bool zip, Scoring p,
                             __half2* temp, std::size_t temp_rows, int* scores) {
  const std::size_t thread = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  const std::size_t warp = thread / warp_size;
  const std::size_t warps = gridDim.x * std::size_t(blockDim.x) / warp_size;
  __half2* temp_v = temp + warp * 2 * temp_rows;
  __half2* temp_f = temp_v + temp_rows;
  const std::size_t groups = (count + 1) / 2;
  for (std::size_t g = warp; g < groups; g += warps) {
    const std::size_t t0 = order[2 * g];
    const bool has1 = 2 * g + 1 < count;
    const std::size_t t1 = has1 ? order[2 * g + 1] : t0;
    const Task a = locate(refs, queries, t0, reference_count, zip);
    const Task b = locate(refs, queries, t1, reference_count, zip);
    // An empty sequence needs no alignment: the score is one gap over the other sequence.
    const bool empty0 = !a.n || !a.m, empty1 = !b.n || !b.m;
    int s0 = gap(a.n + a.m, p), s1 = gap(b.n + b.m, p);
    if (!empty0 || !empty1) {
      // Pack a real pair into the slot of an empty one; its result is dropped.
      const PairScores s = warp_score2<R>(empty0 ? b : a, empty1 ? a : b, p, temp_v, temp_f);
      if (!empty0) s0 = s.first;
      if (!empty1) s1 = s.second;
    }
    if ((threadIdx.x & (warp_size - 1)) == 0) {
      scores[t0] = s0;
      if (has1) scores[t1] = s1;
    }
  }
}

__global__ void probe_kernel(int* out) { *out = encode_base('G') + 10 * complement_base(encode_base('A')); }
} // namespace

const char* unavailable_reason() noexcept {
  // Probing creates a CUDA context, so do it once per process.
  static const char* const reason = []() -> const char* {
    int count = 0;
    if (const auto e = cudaGetDeviceCount(&count); e != cudaSuccess) return cudaGetErrorString(e);
    if (count == 0) return "no CUDA device available";
    int* device = nullptr;
    int host = -1;
    if (const auto e = cudaMalloc(&device, sizeof(int)); e != cudaSuccess) return cudaGetErrorString(e);
    probe_kernel<<<1, 1>>>(device);
    auto e = cudaGetLastError();  // e.g. no kernel image for this GPU's architecture
    if (e == cudaSuccess) e = cudaMemcpy(&host, device, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(device);
    if (e != cudaSuccess) return cudaGetErrorString(e);
    return host == 2 + 10 * 3 ? "" : "CUDA probe kernel returned a wrong value";
  }();
  return reason;
}

struct Batch::Impl {
  DeviceBuffer<Base> ref_data, query_data;
  DeviceBuffer<std::size_t> ref_offsets, query_offsets;
  DeviceBuffer<__half2> temp;               // Edge scratch of every warp slot.
  DeviceBuffer<int> scores;
  DeviceBuffer<std::size_t> order;          // Task numbers: grouped by bucket, longest first.
  std::size_t bucket_begin[bucket_count + 1] = {};  // Bucket b is order[bucket_begin[b], bucket_begin[b+1]).
  std::size_t tasks = 0, reference_count = 0, max_ref = 0, max_query = 0;
  bool zip = true;
};

Batch::Batch(const Workload& workload) : impl_(new Impl) {
  if (const auto reason = unavailable_reason(); *reason) throw std::runtime_error(reason);
  auto& d = *impl_;
  const auto& refs = workload.references();
  const auto& queries = workload.queries();
  for (const auto* set : {&refs, &queries})
    for (std::size_t i = 0; i < set->total_bases(); ++i)
      if (set->data()[i] < 0 || set->data()[i] > 4) throw std::invalid_argument("Encoded DNA must contain 0..4");
  d.tasks = workload.size();
  d.reference_count = refs.size();
  d.max_ref = refs.max_length();
  d.max_query = queries.max_length();
  d.zip = workload.pairing() == Pairing::zip;
  d.ref_data.upload(refs.data(), refs.total_bases());
  d.ref_offsets.upload(refs.offsets().data(), refs.offsets().size());
  d.query_data.upload(queries.data(), queries.total_bases());
  d.query_offsets.upload(queries.offsets().data(), queries.offsets().size());
  d.scores.resize(d.tasks);

  // Group tasks by query length. Inside a group sort by (m, n), longest first: the kernel
  // scores neighbours order[2g], order[2g + 1] together, so neighbours of similar shape waste
  // the least padding, and the grid-stride loop ends with small tasks so warps finish together.
  std::vector<std::size_t> order(d.tasks), n_len(d.tasks), m_len(d.tasks), bucket(d.tasks);
  for (std::size_t t = 0; t < d.tasks; ++t) {
    const auto [r, q] = d.zip ? std::pair{t, t} : std::pair{t % d.reference_count, t / d.reference_count};
    const std::size_t n = refs.offsets()[r + 1] - refs.offsets()[r];
    const std::size_t m = queries.offsets()[q + 1] - queries.offsets()[q];
    std::size_t b = 0;
    while (m > buckets[b].max_query) ++b;
    order[t] = t, n_len[t] = n, m_len[t] = m, bucket[t] = b;
    ++d.bucket_begin[b + 1];
  }
  for (std::size_t b = 0; b < bucket_count; ++b) d.bucket_begin[b + 1] += d.bucket_begin[b];
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    if (bucket[a] != bucket[b]) return bucket[a] < bucket[b];
    if (m_len[a] != m_len[b]) return m_len[a] > m_len[b];
    if (n_len[a] != n_len[b]) return n_len[a] > n_len[b];
    return a < b;
  });
  d.order.upload(order.data(), order.size());
}

Batch::~Batch() = default;

void Batch::score(Scoring p, std::vector<std::int32_t>& out, int regs) {
  auto& d = *impl_;
  if (p.gap_extend > 0 || p.gap_open > p.gap_extend)
    throw std::invalid_argument("Requires gap_open<=gap_extend<=0");
  // The kernel computes in half precision, which is exact only for small scoring values.
  if (!fits_half(p)) throw std::length_error("Scoring exceeds the half2 range of the cuda backend");
  // Same int32 headroom rule as sw::scalar, applied to the longest pair.
  const std::int64_t magnitude = std::max({std::int64_t(p.match), -std::int64_t(p.mismatch),
      -std::int64_t(p.gap_open), -std::int64_t(p.gap_extend), std::int64_t(1)});
  const auto limit = static_cast<std::size_t>(INT32_MAX / (4 * magnitude));
  if (d.max_ref > limit || d.max_query > limit - d.max_ref)
    throw std::length_error("Alignment exceeds cuda score range");
  out.resize(d.tasks);
  if (!d.tasks) return;

  if (regs != 0 && regs != 1 && regs != 2 && regs != 4 && regs != 8 && regs != 16)
    throw std::invalid_argument("regs must be 0, 1, 2, 4, 8 or 16");

  // A fixed regs runs everything in one launch; 0 runs one launch per length bucket.
  struct Launch {
    std::size_t first, count;
    int regs;
  };
  std::vector<Launch> launches;
  if (regs) launches.push_back({0, d.tasks, regs});
  else
    for (std::size_t b = 0; b < bucket_count; ++b)
      if (const auto count = d.bucket_begin[b + 1] - d.bucket_begin[b])
        launches.push_back({d.bucket_begin[b], count, buckets[b].regs});

  // As many concurrent warps as the edge-scratch budget allows; the launches share it.
  constexpr std::size_t budget = std::size_t(256) << 20;
  const std::size_t rows = std::max<std::size_t>(1, d.max_ref);
  const auto fit = std::max<std::size_t>(1, budget / (2 * rows * sizeof(__half2)));
  // A warp scores two tasks at once, so a launch needs one warp per pair of tasks.
  auto blocks_for = [&](std::size_t count) {
    return (std::min((count + 1) / 2, fit) + warps_per_block - 1) / warps_per_block;
  };
  std::size_t max_blocks = 0;
  for (const auto& l : launches) max_blocks = std::max(max_blocks, blocks_for(l.count));
  d.temp.resize(max_blocks * warps_per_block * 2 * rows);

  for (const auto& l : launches) {
    auto launch = [&](auto kernel) {
      kernel<<<unsigned(blocks_for(l.count)), warps_per_block * warp_size>>>(
          DeviceSet{d.ref_data.data(), d.ref_offsets.data()},
          DeviceSet{d.query_data.data(), d.query_offsets.data()},
          d.order.data() + l.first, l.count, d.reference_count, d.zip, p, d.temp.data(), rows,
          d.scores.data());
    };
    switch (l.regs) {
      case 1: launch(score_kernel<1>); break;
      case 2: launch(score_kernel<2>); break;
      case 4: launch(score_kernel<4>); break;
      case 8: launch(score_kernel<8>); break;
      default: launch(score_kernel<16>); break;
    }
  }
  check(cudaGetLastError(), "score_kernel launch");
  check(cudaMemcpy(out.data(), d.scores.data(), d.tasks * sizeof(int), cudaMemcpyDeviceToHost),
        "score_kernel");
}
} // namespace sw::cuda
