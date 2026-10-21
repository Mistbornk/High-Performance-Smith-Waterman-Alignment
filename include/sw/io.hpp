#pragma once
#include <sw/sequence.hpp>
#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace sw::io {
// Encoded sequences packed into one contiguous buffer (CUDASW4-style chars/offsets),
// so a device backend can upload a whole set with one copy. Sequence i occupies
// [offsets()[i], offsets()[i+1]) of data(); offsets() always has size()+1 entries.
class SequenceSet {
 public:
  SequenceSet() : offsets_{0} {}
  void add(std::string name, SequenceView sequence);
  void reserve(std::size_t sequences, std::size_t bases);
  std::size_t size() const noexcept { return names_.size(); }
  bool empty() const noexcept { return names_.empty(); }
  SequenceView operator[](std::size_t i) const noexcept {
    return {data_.data() + offsets_[i], offsets_[i + 1] - offsets_[i]};
  }
  const std::string& name(std::size_t i) const noexcept { return names_[i]; }
  const Base* data() const noexcept { return data_.data(); }
  const std::vector<std::size_t>& offsets() const noexcept { return offsets_; }
  std::size_t total_bases() const noexcept { return data_.size(); }
  std::size_t max_length() const noexcept { return max_length_; }

 private:
  Sequence data_;
  std::vector<std::size_t> offsets_;
  std::vector<std::string> names_;
  std::size_t max_length_ = 0;
};

enum class UnknownBase { reject, as_n };
struct ReadOptions {
  // reject: only A/C/G/T/N (any case); as_n: any other letter (IUPAC codes) becomes N.
  UnknownBase unknown = UnknownBase::reject;
};
// FASTA (multi-line) or FASTQ (4-line records), detected from the first record.
// Names are the header up to the first whitespace. Throws runtime_error with the
// line number on malformed input. Empty sequences are kept.
SequenceSet read_sequences(std::istream& input, ReadOptions options = {});
SequenceSet read_sequences(const std::string& path, ReadOptions options = {});
void write_fasta(std::ostream& output, const SequenceSet& sequences, std::size_t width = 80);
} // namespace sw::io
