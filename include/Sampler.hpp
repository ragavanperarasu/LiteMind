#pragma once

#include <cstddef>
#include <span>

namespace litemind {

/** Token-selection methods supported by the initial decoder runtime. */
enum class SamplingMethod { Greedy };

/** @brief Converts decoder logits into the next token ID. */
class Sampler final {
public:
    [[nodiscard]] static std::size_t select_next(std::span<const float> logits,
                                                  SamplingMethod method = SamplingMethod::Greedy);
};

}  // namespace litemind
