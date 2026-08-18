#pragma once

#include "KvCache.hpp"

#include <span>
#include <vector>

namespace litemind {

/** @brief Portable scaled dot-product causal attention over an existing KV cache. */
class Attention final {
public:
    [[nodiscard]] static std::vector<float> attend(std::span<const float> query, const KvCache& cache);
};

}  // namespace litemind
