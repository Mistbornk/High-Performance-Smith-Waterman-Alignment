#include "check.hpp"
#include <sw/workload.hpp>
#include <iostream>
#include <sstream>

namespace {
sw::io::SequenceSet parse(const std::string& text, sw::io::ReadOptions options = {}) {
  std::istringstream input(text);
  return sw::io::read_sequences(input, options);
}
std::string decoded(sw::SequenceView sequence) {
  std::string text;
  for (auto base : sequence) text.push_back(sw::decode_base(base));
  return text;
}
} // namespace

int main() {
  try {
    // FASTA: multi-line, CRLF, blank lines, lowercase, empty record, header comments.
    const auto fasta = parse("\n>r1 description here\r\nACgt\r\nNN\n\n>empty\n>r3\nT\n");
    CHECK(fasta.size() == 3);
    CHECK(fasta.name(0) == "r1" && decoded(fasta[0]) == "ACGTNN");
    CHECK(fasta.name(1) == "empty" && fasta[1].empty());
    CHECK(decoded(fasta[2]) == "T");
    CHECK(fasta.total_bases() == 7 && fasta.max_length() == 6);
    CHECK(fasta.offsets().size() == 4 && fasta.offsets().back() == 7);
    CHECK(fasta[2].data() == fasta.data() + 6);  // Packed contiguously.

    const auto fastq = parse("@q1 x\nACGT\n+\nIIII\n\n@q2\nN\n+q2\n#\n");
    CHECK(fastq.size() == 2 && fastq.name(0) == "q1" && decoded(fastq[1]) == "N");

    CHECK(parse("").empty() && parse("\n\n").empty());
    expect_throw<std::runtime_error>([] { parse("ACGT\n"); });
    expect_throw<std::runtime_error>([] { parse(">a\nACRT\n"); });
    CHECK(decoded(parse(">a\nACRT\n", {sw::io::UnknownBase::as_n})[0]) == "ACNT");
    expect_throw<std::runtime_error>([] { parse(">a\nAC1T\n", {sw::io::UnknownBase::as_n}); });
    expect_throw<std::runtime_error>([] { parse("@q\nACGT\n+\nIII\n"); });
    expect_throw<std::runtime_error>([] { parse("@q\nACGT\nIIII\n"); });
    expect_throw<std::runtime_error>([] { parse("@q\nACGT\n+\n"); });
    expect_throw<std::runtime_error>([] { sw::io::read_sequences("/nonexistent/x.fa"); });

    std::ostringstream out;
    sw::io::write_fasta(out, fasta, 4);
    CHECK(out.str() == ">r1\nACGT\nNN\n>empty\n>r3\nT\n");
    const auto again = parse(out.str());
    CHECK(again.size() == 3 && decoded(again[0]) == "ACGTNN");

    // Workloads.
    const auto refs = parse(">a\nACG\n>b\nAC\n");
    const auto queries = parse(">x\nA\n>y\nACGT\n");
    sw::Workload zip(refs, queries, sw::Pairing::zip);
    CHECK(zip.size() == 2 && zip.cells() == 3 * 1 + 2 * 4);
    CHECK(zip.task(1).reference == 1 && zip.task(1).query == 1);
    sw::Workload all(refs, parse(">x\nA\n>y\nACGT\n>z\nGG\n"), sw::Pairing::all_vs_all);
    CHECK(all.size() == 6 && all.cells() == 5 * 7);
    CHECK(all.task(0).reference == 0 && all.task(0).query == 0);
    CHECK(all.task(1).reference == 1 && all.task(1).query == 0);
    CHECK(all.task(5).reference == 1 && all.task(5).query == 2);
    CHECK(decoded(all.query(5)) == "GG" && decoded(all.reference(5)) == "AC");
    expect_throw<std::invalid_argument>([&] { sw::Workload(refs, parse(">x\nA\n"), sw::Pairing::zip); });
    CHECK(sw::Workload({}, queries, sw::Pairing::all_vs_all).size() == 0);
    std::cout << "FASTA/FASTQ IO and workload pairing passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
