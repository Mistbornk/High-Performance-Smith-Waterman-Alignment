#include <sw/alignment.hpp>
#include <iostream>
#include <stdexcept>
#include <string_view>
int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: sw-align REFERENCE QUERY [striped|scalar|cuda]\n";
    return 2;
  }
  try {
    const std::string_view name = argc == 4 ? argv[3] : "striped";
    sw::Backend backend;
    if (name == "striped") backend = sw::Backend::striped;
    else if (name == "scalar") backend = sw::Backend::scalar;
    else if (name == "cuda") backend = sw::Backend::cuda;
    else throw std::invalid_argument("Unknown backend");
    auto aligner = sw::make_aligner(backend);
    const auto result = aligner->align(argv[1], argv[2]);
    std::cout << "backend=" << sw::backend_name(backend) << " model=global score=" << result.score
              << " cigar=" << result.cigar.to_string() << '\n';
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
