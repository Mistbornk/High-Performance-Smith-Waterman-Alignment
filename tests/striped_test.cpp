#include "check.hpp"
#include "support/check_alignment.hpp"
#include <future>
#include <iostream>
#include <random>
#include <string>

int main() {
  if (const auto reason = sw::unavailable_reason(sw::Backend::striped); *reason) {
    std::cout << reason << '\n'; return 77;
  }
  try {
    auto aligner = sw::make_aligner();
    CHECK(aligner->backend() == sw::Backend::striped);
    struct Example { const char* ref; const char* query; int score; const char* cigar; };
    for (const auto x : {Example{"ACGT","ACGT",4,"4M"}, {"ACGT","AGGT",-1,"4M"},
                         {"ACGT","ACTGT",-2,"2M1I2M"}, {"ACTGT","ACGT",-2,"2M1D2M"},
                         {"","ACG",-8,"3I"}, {"ACG","",-8,"3D"}, {"","",0,""}}) {
      const auto result = aligner->align(x.ref,x.query);
      CHECK(result.score == x.score); CHECK(result.cigar.to_string() == x.cigar);
      check_alignment(*aligner,x.ref,x.query,{});
    }
    CHECK(aligner->align("aCgTn","ACGTN").score == 5);
    CHECK(aligner->align("N","A").score == -4);
    expect_throw<std::invalid_argument>([&] { aligner->align("AX","AA"); });
    expect_throw<std::invalid_argument>([&] { aligner->align(sw::Sequence{sw::Base{-1}},sw::Sequence{}); });
    expect_throw<std::invalid_argument>([&] { aligner->align(sw::Sequence{},sw::Sequence{sw::Base{5}}); });
    expect_throw<std::invalid_argument>([] { sw::make_aligner(sw::Backend::striped,{1,-4,-1,-6}); });
    expect_throw<std::length_error>([] { sw::make_aligner(sw::Backend::striped,{200,-150,-260,-11}); });
    expect_throw<std::length_error>([] { sw::make_aligner(sw::Backend::striped,{1,-4,INT32_MIN,-1}); });
    std::mt19937 rng(787);
    const std::string bases="ACGTN";
    for (const auto p : {sw::Scoring{}, sw::Scoring{3,-1,-4,-3}, sw::Scoring{10,-15,-30,-5}, sw::Scoring{63,-64,-32,-1}}) {
      auto worker = sw::make_aligner(sw::Backend::striped,p);
      for (const auto length : {0u,1u,2u,15u,16u,17u,31u,32u,33u,63u,65u,129u}) {
        for (unsigned n=0; n<8; ++n) {
          std::string ref,query;
          for (unsigned i=0;i<length;++i) ref += bases[rng()%5];
          for (unsigned i=0;i<length+n;++i) query += bases[rng()%5];
          check_alignment(*worker,ref,query,p);
        }
      }
      check_alignment(*worker,std::string(20,'A')+std::string(160,'C')+std::string(20,'G'),
                      std::string(20,'A')+std::string(20,'G'),p);
      check_alignment(*worker,std::string(513,'A'),std::string(513,'A'),p);
      check_alignment(*worker,"ACGT","ACTGT",p); // workspace shrinks after long input
    }
    std::vector<std::future<void>> workers;
    for (int i=0;i<4;++i) workers.push_back(std::async(std::launch::async, [] {
      auto worker=sw::make_aligner();
      for (int j=0;j<20;++j) check_alignment(*worker,"ACGTNACGT","ACGTACGT",{});
    }));
    for (auto& worker : workers) worker.get();
    std::cout << "Striped smoke, oracle, CIGAR replay, validation and independent workers passed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n'; return 1;
  }
}
