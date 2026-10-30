#include <sw/io.hpp>
#include <algorithm>
#include <fstream>
#include <istream>
#include <limits>
#include <ostream>
#include <stdexcept>

namespace sw::io {
void SequenceSet::add(std::string name, SequenceView sequence) {
  // Workload tasks address sequences with 32-bit indices.
  if (names_.size() >= std::numeric_limits<std::uint32_t>::max())
    throw std::length_error("SequenceSet exceeds 2^32-1 sequences");
  data_.append(sequence);
  offsets_.push_back(data_.size());
  names_.push_back(std::move(name));
  max_length_ = std::max(max_length_, sequence.size());
}
void SequenceSet::reserve(std::size_t sequences, std::size_t bases) {
  data_.reserve(bases);
  offsets_.reserve(sequences + 1);
  names_.reserve(sequences);
}

namespace {
[[noreturn]] void fail(std::size_t line, const std::string& what) {
  throw std::runtime_error("line " + std::to_string(line) + ": " + what);
}
std::string record_name(const std::string& header) {
  const auto end = header.find_first_of(" \t", 1);
  return header.substr(1, end == std::string::npos ? std::string::npos : end - 1);
}
void append_bases(Sequence& out, const std::string& text, ReadOptions options, std::size_t line) {
  for (char c : text) {
    switch (c) {
      case 'A': case 'a': case 'C': case 'c': case 'G': case 'g':
      case 'T': case 't': case 'N': case 'n':
        out.push_back(encode_base(c));
        break;
      case ' ': case '\t':
        break;
      default:
        if (options.unknown == UnknownBase::as_n &&
            ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '*' || c == '-')) {
          out.push_back(encode_base('N'));
          break;
        }
        fail(line, std::string("invalid DNA symbol '") + c + "'");
    }
  }
}
bool next_line(std::istream& input, std::string& line, std::size_t& number) {
  if (!std::getline(input, line)) return false;
  ++number;
  if (!line.empty() && line.back() == '\r') line.pop_back();
  return true;
}
} // namespace

SequenceSet read_sequences(std::istream& input, ReadOptions options) {
  SequenceSet result;
  std::string line, name;
  std::size_t number = 0;
  // Skip leading blank lines, then detect the format from the first header.
  while (next_line(input, line, number) && line.empty()) {}
  if (line.empty()) return result;
  Sequence sequence;
  if (line[0] == '>') {
    name = record_name(line);
    while (next_line(input, line, number)) {
      if (line.empty()) continue;
      if (line[0] == '>') {
        result.add(std::move(name), sequence);
        sequence.clear();
        name = record_name(line);
      } else {
        append_bases(sequence, line, options, number);
      }
    }
    result.add(std::move(name), sequence);
  } else if (line[0] == '@') {
    std::string bases, plus, quality;
    do {
      if (line.empty()) continue;
      if (line[0] != '@') fail(number, "expected FASTQ header '@'");
      name = record_name(line);
      if (!next_line(input, bases, number) || !next_line(input, plus, number) ||
          !next_line(input, quality, number))
        fail(number, "truncated FASTQ record");
      if (plus.empty() || plus[0] != '+') fail(number - 1, "expected FASTQ separator '+'");
      if (quality.size() != bases.size()) fail(number, "FASTQ quality length differs from sequence");
      sequence.clear();
      append_bases(sequence, bases, options, number - 2);
      result.add(std::move(name), sequence);
    } while (next_line(input, line, number));
  } else {
    fail(number, "expected FASTA '>' or FASTQ '@' header");
  }
  if (input.bad()) throw std::runtime_error("I/O error while reading sequences");
  return result;
}

SequenceSet read_sequences(const std::string& path, ReadOptions options) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open " + path);
  try {
    return read_sequences(input, options);
  } catch (const std::runtime_error& e) {
    throw std::runtime_error(path + ": " + e.what());
  }
}

void write_fasta(std::ostream& output, const SequenceSet& sequences, std::size_t width) {
  if (!width) throw std::invalid_argument("FASTA line width must be positive");
  std::string text;
  for (std::size_t i = 0; i < sequences.size(); ++i) {
    output << '>' << sequences.name(i) << '\n';
    const auto sequence = sequences[i];
    for (std::size_t start = 0; start < sequence.size(); start += width) {
      const auto end = std::min(sequence.size(), start + width);
      text.clear();
      for (std::size_t j = start; j < end; ++j) text.push_back(decode_base(sequence[j]));
      output << text << '\n';
    }
  }
}
} // namespace sw::io

namespace sw {
const char* pairing_name(Pairing pairing) noexcept {
  switch (pairing) {
    case Pairing::zip: return "zip";
    case Pairing::all_vs_all: return "all_vs_all";
  }
  return "unknown";
}

Workload::Workload(io::SequenceSet references, io::SequenceSet queries, Pairing pairing)
    : references_(std::move(references)), queries_(std::move(queries)), pairing_(pairing) {
  constexpr auto max = std::numeric_limits<std::uint64_t>::max();
  if (pairing_ == Pairing::zip) {
    if (references_.size() != queries_.size())
      throw std::invalid_argument("zip pairing needs equal reference and query counts (" +
          std::to_string(references_.size()) + " vs " + std::to_string(queries_.size()) + ")");
    size_ = references_.size();
    for (std::size_t i = 0; i < size_; ++i) {
      const std::uint64_t n = references_[i].size(), m = queries_[i].size();
      if (m && n > (max - cells_) / m) throw std::length_error("Workload cell count overflow");
      cells_ += n * m;
    }
  } else if (pairing_ == Pairing::all_vs_all) {
    const std::uint64_t r = references_.size(), q = queries_.size();
    if (q && r > std::numeric_limits<std::size_t>::max() / q)
      throw std::length_error("all_vs_all task count overflow");
    size_ = static_cast<std::size_t>(r * q);
    const std::uint64_t n = references_.total_bases(), m = queries_.total_bases();
    if (m && n > max / m) throw std::length_error("Workload cell count overflow");
    cells_ = n * m;
  } else {
    throw std::invalid_argument("Unknown pairing");
  }
}
} // namespace sw
