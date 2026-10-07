#pragma once
#include <stdexcept>
#include <string>
inline void check(bool condition, const char* expression) {
  if (!condition) throw std::runtime_error(std::string("Check failed: ") + expression);
}
#define CHECK(expression) check(static_cast<bool>(expression), #expression)
template<class Exception, class Function> void expect_throw(Function function) {
  try { function(); } catch (const Exception&) { return; }
  throw std::runtime_error("Expected exception was not thrown");
}
