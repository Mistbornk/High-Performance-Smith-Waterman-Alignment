// sw-bench: time one backend on one FASTA workload and print a single CSV row.
// Loading and encoding are measured separately from the timed runs; see
// docs/benchmark.md for the full measurement contract.
#include <sw/io.hpp>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int skip_code = 77;  // CTest/driver convention: backend unavailable.

enum class Backend { striped, scalar, cuda };
const char* backend_name(Backend backend) {
  switch (backend) {
    case Backend::striped: return "striped";
    case Backend::scalar: return "scalar";
    case Backend::cuda: return "cuda";
  }
  return "unknown";
}

struct Args {
  Backend backend = Backend::striped;
  std::string reference, query, results, label;
  sw::Pairing pairing = sw::Pairing::zip;
  sw::Scoring scoring;
  unsigned threads = 1;
  bool traceback = true;
  sw::io::ReadOptions read;
  unsigned warmup = 1, repetitions = 5;
  bool header = true;
};

const char* usage =
    "Usage: sw-bench --reference R.fa --query Q.fa [options]\n"
    "  --backend striped|scalar|cuda   (default striped)\n"
    "  --threads N                     CPU worker threads (default 1)\n"
    "  --pairing zip|all_vs_all        (default zip)\n"
    "  --scoring M,X,O,E               match,mismatch,gap_open,gap_extend (default 1,-4,-6,-1)\n"
    "  --score-only                    request score-only results (no traceback)\n"
    "  --unknown reject|as_n           non-ACGTN letters (default reject)\n"
    "  --warmup N --repetitions N      (default 1 and 5)\n"
    "  --results PATH                  write per-task TSV from the last repetition\n"
    "  --label TEXT  --no-header\n"
    "Prints one CSV row to stdout. Exit 77 when the backend is unavailable.\n";

unsigned parse_unsigned(const std::string& text, const std::string& key) {
  if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos || text.size() > 9)
    throw std::invalid_argument(key + " expects an unsigned integer");
  return static_cast<unsigned>(std::stoul(text));
}
sw::Scoring parse_scoring(const std::string& text) {
  std::vector<int> values;
  std::size_t start = 0;
  while (start <= text.size()) {
    const auto end = std::min(text.find(',', start), text.size());
    std::size_t used = 0;
    const auto item = text.substr(start, end - start);
    values.push_back(std::stoi(item, &used));
    if (used != item.size()) throw std::invalid_argument("bad --scoring value " + item);
    start = end + 1;
  }
  if (values.size() != 4) throw std::invalid_argument("--scoring expects M,X,O,E");
  return {values[0], values[1], values[2], values[3]};
}

Args parse(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help") { std::cout << usage; std::exit(0); }
    if (key == "--score-only") { args.traceback = false; continue; }
    if (key == "--no-header") { args.header = false; continue; }
    if (i + 1 == argc) throw std::invalid_argument("missing value for " + key);
    const std::string value = argv[++i];
    if (key == "--backend") {
      if (value == "striped") args.backend = Backend::striped;
      else if (value == "scalar") args.backend = Backend::scalar;
      else if (value == "cuda") args.backend = Backend::cuda;
      else throw std::invalid_argument("unknown backend " + value);
    } else if (key == "--pairing") {
      if (value == "zip") args.pairing = sw::Pairing::zip;
      else if (value == "all_vs_all") args.pairing = sw::Pairing::all_vs_all;
      else throw std::invalid_argument("unknown pairing " + value);
    } else if (key == "--unknown") {
      if (value == "reject") args.read.unknown = sw::io::UnknownBase::reject;
      else if (value == "as_n") args.read.unknown = sw::io::UnknownBase::as_n;
      else throw std::invalid_argument("unknown --unknown mode " + value);
    } else if (key == "--reference") args.reference = value;
    else if (key == "--query") args.query = value;
    else if (key == "--results") args.results = value;
    else if (key == "--label") args.label = value;
    else if (key == "--scoring") args.scoring = parse_scoring(value);
    else if (key == "--threads") args.threads = parse_unsigned(value, key);
    else if (key == "--warmup") args.warmup = parse_unsigned(value, key);
    else if (key == "--repetitions") args.repetitions = parse_unsigned(value, key);
    else throw std::invalid_argument("unknown option " + key);
  }
  if (args.reference.empty() || args.query.empty())
    throw std::invalid_argument("--reference and --query are required");
  if (!args.repetitions) throw std::invalid_argument("--repetitions must be positive");
  if (args.threads == 0 || args.threads > 1024) throw std::invalid_argument("--threads must be 1..1024");
  if (args.label.find_first_of(",\n\"") != std::string::npos)
    throw std::invalid_argument("--label must not contain commas, quotes or newlines");
  return args;
}

double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// FNV-1a over scores in task order. Backends may pick different CIGARs on ties,
// so only scores are compared across backends.
std::uint64_t score_checksum(const std::vector<sw::Alignment>& results) {
  std::uint64_t hash = 1469598103934665603ull;
  for (const auto& result : results) {
    auto value = static_cast<std::uint32_t>(result.score);
    for (int b = 0; b < 4; ++b, value >>= 8) hash = (hash ^ (value & 0xff)) * 1099511628211ull;
  }
  return hash;
}

void write_results(const std::string& path, const sw::Workload& workload,
                   const std::vector<sw::Alignment>& results) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path);
  out << "task\treference\tquery\tscore\tcigar\n";
  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto task = workload.task(i);
    out << i << '\t' << workload.references().name(task.reference) << '\t'
        << workload.queries().name(task.query) << '\t' << results[i].score << '\t'
        << (results[i].cigar.empty() ? "*" : results[i].cigar.to_string()) << '\n';
  }
  if (!out) throw std::runtime_error("failed writing " + path);
}
// The backend's align/score pair. Unbuilt backends return false (exit 77).
struct Functions {
  std::int32_t (*score)(sw::SequenceView, sw::SequenceView, sw::Scoring);
  sw::Alignment (*align)(sw::SequenceView, sw::SequenceView, sw::Scoring);
};
bool lookup(Backend backend, Functions& out) {
  switch (backend) {
    case Backend::scalar: out = {sw::scalar::score, sw::scalar::align}; return true;
#if SW_HAVE_STRIPED
    case Backend::striped: out = {sw::striped::score, sw::striped::align}; return true;
#endif
    default: return false;
  }
}

} // namespace

int main(int argc, char** argv) {
  try {
    const auto args = parse(argc, argv);
    Functions run;
    if (!lookup(args.backend, run)) {
      std::cerr << "sw-bench: " << backend_name(args.backend) << " unavailable: not built\n";
      return skip_code;
    }

    auto start = std::chrono::steady_clock::now();
    sw::Workload workload(sw::io::read_sequences(args.reference, args.read),
                          sw::io::read_sequences(args.query, args.read), args.pairing);
    const auto load_s = seconds_since(start);
    // CPU backends have nothing to set up; device backends will time uploads here.
    const double setup_s = 0;

    std::vector<sw::Alignment> results(workload.size());
    auto align_all = [&] {
      sw::parallel_for(workload.size(), args.threads, [&](std::size_t i) {
        const auto ref = workload.reference(i), query = workload.query(i);
        if (args.traceback) results[i] = run.align(ref, query, args.scoring);
        else results[i].score = run.score(ref, query, args.scoring);
      });
    };
    for (unsigned i = 0; i < args.warmup; ++i) align_all();
    std::vector<double> samples;
    for (unsigned i = 0; i < args.repetitions; ++i) {
      start = std::chrono::steady_clock::now();
      align_all();
      samples.push_back(seconds_since(start));
    }
    if (!args.results.empty()) write_results(args.results, workload, results);

    std::sort(samples.begin(), samples.end());
    const auto mid = samples.size() / 2;
    const double median = samples.size() % 2 ? samples[mid] : (samples[mid - 1] + samples[mid]) / 2;
    const auto gcups = [&](double s) { return s > 0 ? workload.cells() / s / 1e9 : 0.0; };
    if (args.header)
      std::cout << "label,backend,threads,traceback,pairing,references,queries,tasks,cells,"
                   "load_s,setup_s,min_s,median_s,max_s,gcups_median,gcups_best,score_checksum\n";
    std::cout << args.label << ',' << backend_name(args.backend) << ',' << args.threads
              << ',' << (args.traceback ? 1 : 0) << ',' << sw::pairing_name(workload.pairing())
              << ',' << workload.references().size() << ',' << workload.queries().size() << ','
              << workload.size() << ',' << workload.cells() << ',' << load_s << ',' << setup_s << ','
              << samples.front() << ',' << median << ',' << samples.back() << ',' << gcups(median)
              << ',' << gcups(samples.front()) << ',' << std::hex << score_checksum(results)
              << std::dec << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "sw-bench: " << e.what() << '\n';
    return 1;
  }
}
