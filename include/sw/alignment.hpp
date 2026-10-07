#pragma once
#include <sw/cigar.hpp>
#include <sw/sequence.hpp>
#include <memory>

namespace sw {
enum class Backend { striped, scalar, cuda };
struct Scoring {
  int match = 1;
  int mismatch = -4;
  int gap_open = -6;
  int gap_extend = -1;
};
struct AlignmentResult {
  std::int32_t score = 0;
  Cigar cigar;
  std::size_t reference_begin = 0, reference_end = 0;
  std::size_t query_begin = 0, query_end = 0;
};
// Empty reason means available. This checks build/hardware, not scoring/input limits.
const char* unavailable_reason(Backend backend) noexcept;
const char* backend_name(Backend backend) noexcept;

// Synchronous GLOBAL affine-gap alignment: gap(k)=open+(k-1)*extend.
// All input is host memory, borrowed only for this call. Results own their data.
// N is a fifth symbol (N/N matches); intervals are half-open and cover both inputs.
// One instance per worker; concurrent calls on the same instance are unsupported.
class Aligner {
 public:
  virtual ~Aligner() = default;
  virtual Backend backend() const noexcept = 0;
  virtual AlignmentResult align(SequenceView reference, SequenceView query) = 0;
  AlignmentResult align(std::string_view reference, std::string_view query) {
    const auto ref = encode_sequence(reference), que = encode_sequence(query);
    return align(SequenceView(ref), SequenceView(que));
  }
};
// Throws runtime_error for unavailable backends. Never silently falls back.
std::unique_ptr<Aligner> make_aligner(Backend backend = Backend::striped, Scoring scoring = {});
} // namespace sw
