#include <sw/sw.hpp>
#include <iostream>
#include <stdexcept>
#include <string_view>
int main(int argc, char** argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: sw-align REFERENCE QUERY [striped|scalar]\n";
    return 2;
  }
  try {
    const std::string_view name = argc == 4 ? argv[3] : (SW_HAVE_STRIPED ? "striped" : "scalar");
    const auto ref = sw::encode_sequence(argv[1]), query = sw::encode_sequence(argv[2]);
    sw::Alignment result;
    if (name == "scalar") result = sw::scalar::align(ref, query);
#if SW_HAVE_STRIPED
    else if (name == "striped") result = sw::striped::align(ref, query);
#endif
    else throw std::invalid_argument("Unknown or unbuilt backend");
    std::cout << "backend=" << name << " model=global score=" << result.score
              << " cigar=" << result.cigar.to_string() << '\n';
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
