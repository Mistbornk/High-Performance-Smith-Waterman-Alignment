#include <sw/io.hpp>
#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Ported from the earlier Biovoltron SmithWatermanCuda kernel, changed to global
// alignment and score-only: no trace matrix, no gap-length tracking, no sink search.
namespace sw::cuda {
namespace {
constexpr int warp_size = 32;
constexpr int warps_per_block = 8;
constexpr unsigned full_mask = 0xffffffffu;
constexpr int neg_inf = INT_MIN / 2;

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

struct DeviceSet {
  const Base* data;
  const std::size_t* offsets;
};

__device__ __forceinline__ int gap(int k, Scoring p) {
  return k ? p.gap_open + (k - 1) * p.gap_extend : 0;
}

// One warp scores one pair: reference rows 1..n, query columns 1..m. The query is cut
// into stripes of 32 * R columns; lane l owns R consecutive columns of a stripe and keeps
// their H and vertical-gap E in registers. Lanes sweep anti-diagonals (lane l handles row
// d - l - 1 at step d). Only the border between two lanes travels through shuffles: H and
// the horizontal gap F of a lane's last column, plus the reference character. The
// stripe's last column is stored in temp_h/temp_f (one entry per reference row) and
// becomes the next stripe's left edge.
//
//   E(i,j) = max(H(i-1,j) + open, E(i-1,j) + extend)   vertical, from up = h[k]
//   F(i,j) = max(H(i,j-1) + open, F(i,j-1) + extend)   horizontal, from left_h / left_f
//   H(i,j) = max(H(i-1,j-1) + sub, E(i,j), F(i,j))
// with H(i,0) = gap(i), H(0,j) = gap(j), E(0,j) = F(i,0) = -inf: the same DP as
// sw::scalar, so scores match it exactly.
template <int R>
__device__ int warp_score(const Base* __restrict__ ref, const Base* __restrict__ query, int n, int m,
                          Scoring p, int* __restrict__ temp_h, int* __restrict__ temp_f) {
  constexpr int stripe = warp_size * R;
  const int lane = threadIdx.x & (warp_size - 1);
  // The first stripe's left edge is column 0.
  for (int row = lane; row < n; row += warp_size) temp_h[row] = gap(row + 1, p), temp_f[row] = neg_inf;
  __syncwarp();

  int result = 0;
  for (int block = 0; block < m; block += stripe) {
    const int lanes = min(warp_size, (m - block + R - 1) / R); // Lanes owning a column <= m.
    const int col0 = block + lane * R + 1;
    int q[R], h[R], e[R];
#pragma unroll
    for (int k = 0; k < R; ++k) {
      const int col = col0 + k;
      q[k] = col <= m ? query[col-1] : -1;
      h[k] = gap(col, p);
      e[k] = neg_inf;
    }
    // H(0, col0 - 1)
    int h_diag = gap(col0 - 1, p);
    // Lane 0 takes these from the cache.
    int h_left = 0, f_left = neg_inf, r = 0;
    // last column, handed to lane + 1.
    int h_out = 0, f_out = neg_inf;
    int cache_h = 0, cache_f = neg_inf, cache_r = 0;

    for (int base = 2; base <= lanes + n; base += warp_size) {
      // Every 32 steps each lane prefetches one left-edge row; the values then shift
      // down one lane per step, so lane 0 always holds the row it needs next.
      const int row = base - 2 + lane;
      if (row < n) cache_h = temp_h[row], cache_f = temp_f[row], cache_r = ref[row];
      else cache_h = 0, cache_f = neg_inf, cache_r = 0;

      for (int d = base; d < base + warp_size; ++d) {
        const int i = d - lane - 1;  // Reference row of this lane at step d.
        if (lane < lanes && i >= 1 && i <= n) {
          if (lane == 0) r = cache_r, h_left = cache_h, f_left = cache_f;
          int diag = h_diag, left_h = h_left, left_f = f_left;
#pragma unroll
          for (int k = 0; k < R; ++k) {
            const int up = h[k];
            e[k] = max(up + p.gap_open, e[k] + p.gap_extend);
            const int f = max(left_h + p.gap_open, left_f + p.gap_extend);
            const int v = max(max(diag + (r == q[k] ? p.match : p.mismatch), e[k]), f);
            if (col0 + k == m && i == n) result = v;
            diag = up, h[k] = v, left_h = v, left_f = f;
          }
          if (lane == warp_size - 1) temp_h[i - 1] = left_h, temp_f[i - 1] = left_f;
          h_diag = h_left;
          h_out = left_h, f_out = left_f;
        }
        r = __shfl_up_sync(full_mask, r, 1);
        h_left = __shfl_up_sync(full_mask, h_out, 1);
        f_left = __shfl_up_sync(full_mask, f_out, 1);
        cache_h = __shfl_down_sync(full_mask, cache_h, 1);
        cache_f = __shfl_down_sync(full_mask, cache_f, 1);
        cache_r = __shfl_down_sync(full_mask, cache_r, 1);
      }
    }
    __syncwarp();  // The next stripe reads the edge lane 31 just wrote.
  }
  return __shfl_sync(full_mask, result, ((m - 1) % stripe) / R);
}

// Grid-stride over order[0..count), one warp per task; each warp slot owns 2 * temp_rows
// ints of edge scratch, reused for every task it processes. order holds task numbers.
template <int R>
__global__ void score_kernel(DeviceSet refs, DeviceSet queries, const std::size_t* order,
                             std::size_t count, std::size_t reference_count, bool zip, Scoring p,
                             int* temp, std::size_t temp_rows, int* scores) {
  const std::size_t thread = blockIdx.x * std::size_t(blockDim.x) + threadIdx.x;
  const std::size_t warp = thread / warp_size;
  const std::size_t warps = gridDim.x * std::size_t(blockDim.x) / warp_size;
  int* temp_h = temp + warp * 2 * temp_rows;
  int* temp_f = temp_h + temp_rows;
  for (std::size_t i = warp; i < count; i += warps) {
    const std::size_t t = order[i];
    // Same mapping as Workload::task().
    const std::size_t r = zip ? t : t % reference_count, q = zip ? t : t / reference_count;
    const int n = int(refs.offsets[r + 1] - refs.offsets[r]);
    const int m = int(queries.offsets[q + 1] - queries.offsets[q]);
    const int s = !n || !m ? gap(n + m, p)
                           : warp_score<R>(refs.data + refs.offsets[r], queries.data + queries.offsets[q],
                                        n, m, p, temp_h, temp_f);
    if ((threadIdx.x & (warp_size - 1)) == 0) scores[t] = s;
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
  DeviceBuffer<int> temp, scores;
  DeviceBuffer<std::size_t> order;          // Task numbers: grouped by bucket, heaviest first.
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

  // Group tasks by query length, and inside a group put the largest ones first so the
  // grid-stride loop ends with small tasks and the warps finish together.
  std::vector<std::size_t> order(d.tasks), cells(d.tasks), bucket(d.tasks);
  for (std::size_t t = 0; t < d.tasks; ++t) {
    const auto [r, q] = d.zip ? std::pair{t, t} : std::pair{t % d.reference_count, t / d.reference_count};
    const std::size_t n = refs.offsets()[r + 1] - refs.offsets()[r];
    const std::size_t m = queries.offsets()[q + 1] - queries.offsets()[q];
    std::size_t b = 0;
    while (m > buckets[b].max_query) ++b;
    order[t] = t, cells[t] = n * m, bucket[t] = b;
    ++d.bucket_begin[b + 1];
  }
  for (std::size_t b = 0; b < bucket_count; ++b) d.bucket_begin[b + 1] += d.bucket_begin[b];
  std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    return bucket[a] != bucket[b] ? bucket[a] < bucket[b] : cells[a] != cells[b] ? cells[a] > cells[b] : a < b;
  });
  d.order.upload(order.data(), order.size());
}

Batch::~Batch() = default;

void Batch::score(Scoring p, std::vector<std::int32_t>& out, int regs) {
  auto& d = *impl_;
  if (p.gap_extend > 0 || p.gap_open > p.gap_extend)
    throw std::invalid_argument("Requires gap_open<=gap_extend<=0");
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
  const auto fit = std::max<std::size_t>(1, budget / (2 * rows * sizeof(int)));
  auto blocks_for = [&](std::size_t count) {
    return (std::min(count, fit) + warps_per_block - 1) / warps_per_block;
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
