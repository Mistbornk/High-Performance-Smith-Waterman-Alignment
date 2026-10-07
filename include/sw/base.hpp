#pragma once
#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define SW_HOST_DEVICE __host__ __device__
#else
#define SW_HOST_DEVICE
#endif
namespace sw {
// Stable byte representation shared by host SIMD and device buffers.
using Base = std::int8_t;
static_assert(sizeof(Base) == 1, "DNA encoding requires one-byte elements");
SW_HOST_DEVICE constexpr Base encode_base(char c) noexcept {
  switch (c) {
    case 'A': case 'a': return 0;
    case 'C': case 'c': return 1;
    case 'G': case 'g': return 2;
    case 'T': case 't': return 3;
    default: return 4;
  }
}
SW_HOST_DEVICE constexpr char decode_base(int c) noexcept {
  switch (c) {
    case 0: return 'A'; case 1: return 'C'; case 2: return 'G'; case 3: return 'T';
    default: return 'N';
  }
}
SW_HOST_DEVICE constexpr Base complement_base(Base c) noexcept {
  return c >= 0 && c < 4 ? static_cast<Base>(3 - c) : Base{4};
}
// Non-owning: pointer must refer to memory accessible by the executing processor.
// A host view does NOT transfer or allocate device memory.
struct EncodedSequenceView {
  const Base* data = nullptr;
  std::size_t size = 0;
  SW_HOST_DEVICE constexpr Base operator[](std::size_t i) const noexcept { return data[i]; }
};
}  // namespace sw
#undef SW_HOST_DEVICE
