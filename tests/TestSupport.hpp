#pragma once

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

/**
 * A dependency-free check helper. Each test is its own executable and reports
 * failure through its exit code, so CTest needs nothing else.
 */
namespace test_support {

inline int failures = 0;

inline void check(const bool condition, const std::string& description) {
    if (!condition) {
        std::cerr << "  FAILED: " << description << "\n";
        ++failures;
    }
}

inline void check_close(const double actual, const double expected, const double tolerance,
                        const std::string& description) {
    if (!(std::abs(actual - expected) <= tolerance)) {
        std::cerr << "  FAILED: " << description << " (expected " << expected << ", got " << actual
                  << ", tolerance " << tolerance << ")\n";
        ++failures;
    }
}

inline int report(const std::string& name) {
    if (failures == 0) {
        std::cout << name << ": all checks passed.\n";
        return 0;
    }
    std::cerr << name << ": " << failures << " check(s) failed.\n";
    return 1;
}

}  // namespace test_support
