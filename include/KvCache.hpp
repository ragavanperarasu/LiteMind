#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace litemind {

/**
 * @brief Append-only key/value state for one decoder attention head.
 *
 * Each generated token appends one key and value. Subsequent tokens attend to
 * the complete cache, preserving causal autoregressive execution.
 */
class KvCache final {
public:
    explicit KvCache(std::size_t head_dimension);

    void append(std::span<const float> key, std::span<const float> value);
    void clear() noexcept;

    [[nodiscard]] std::size_t head_dimension() const noexcept;
    [[nodiscard]] std::size_t token_count() const noexcept;
    [[nodiscard]] std::span<const float> key(std::size_t token_index) const;
    [[nodiscard]] std::span<const float> value(std::size_t token_index) const;

private:
    std::size_t head_dimension_;
    std::vector<float> keys_;
    std::vector<float> values_;
};

}  // namespace litemind
