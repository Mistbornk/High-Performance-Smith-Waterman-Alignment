#include "backend.hpp"
#include <sw/batch.hpp>
#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>

namespace sw {
namespace {
// Thread-parallel driver for CPU backends: one Aligner (and its scratch) per worker,
// tasks handed out dynamically in small chunks because pair sizes can vary widely.
class ThreadedBatchAligner final : public BatchAligner {
 public:
  ThreadedBatchAligner(Backend backend, Scoring scoring, BatchOptions options)
      : backend_(backend), options_(options) {
    workers_.reserve(options.threads);
    for (unsigned i = 0; i < options.threads; ++i) workers_.push_back(make_aligner(backend, scoring));
  }
  Backend backend() const noexcept override { return backend_; }
  BatchOptions options() const noexcept override { return options_; }
  void align(const Workload& workload, std::vector<AlignmentResult>& results) override {
    results.resize(workload.size());
    std::atomic<std::size_t> next{0};
    std::atomic<bool> stop{false};
    std::exception_ptr error;
    std::mutex error_mutex;
    auto run = [&](Aligner& aligner) {
      try {
        while (!stop.load(std::memory_order_relaxed)) {
          const auto begin = next.fetch_add(chunk, std::memory_order_relaxed);
          if (begin >= workload.size()) return;
          const auto end = std::min(workload.size(), begin + chunk);
          for (auto i = begin; i < end; ++i) {
            const auto ref = workload.reference(i), query = workload.query(i);
            if (options_.traceback) {
              results[i] = aligner.align(ref, query);
            } else {
              auto& result = results[i];
              result.score = aligner.score(ref, query);
              result.cigar.clear();
              result.reference_begin = 0, result.reference_end = ref.size();
              result.query_begin = 0, result.query_end = query.size();
            }
          }
        }
      } catch (...) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!error) error = std::current_exception();
        stop = true;
      }
    };
    // Never start more threads than chunks; the caller's thread is worker 0.
    const auto chunks = (workload.size() + chunk - 1) / chunk;
    const auto count = std::max<std::size_t>(1, std::min<std::size_t>(workers_.size(), chunks));
    std::vector<std::thread> threads;
    threads.reserve(count - 1);
    for (std::size_t i = 1; i < count; ++i) threads.emplace_back(run, std::ref(*workers_[i]));
    run(*workers_[0]);
    for (auto& thread : threads) thread.join();
    if (error) std::rethrow_exception(error);
  }

 private:
  static constexpr std::size_t chunk = 16;
  Backend backend_;
  BatchOptions options_;
  std::vector<std::unique_ptr<Aligner>> workers_;
};
} // namespace

std::unique_ptr<BatchAligner> make_batch_aligner(Backend backend, Scoring scoring, BatchOptions options) {
  if (const auto reason = unavailable_reason(backend); *reason) throw std::runtime_error(reason);
  if (options.threads == 0 || options.threads > 1024)
    throw std::invalid_argument("BatchOptions::threads must be 1..1024");
  switch (backend) {
    case Backend::cuda:
      return detail::make_cuda_batch(scoring, options);
    case Backend::striped:
    case Backend::scalar:
      return std::make_unique<ThreadedBatchAligner>(backend, scoring, options);
  }
  throw std::invalid_argument("Unknown alignment backend");
}
} // namespace sw
