#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace litemind {

/**
 * @brief Per-layer key/value state for multi-head latent attention.
 *
 * DeepSeek compresses keys and values into one low-rank latent vector before
 * caching, then expands it through kv_b_proj. Expanding once at append time and
 * caching the result costs more memory per token than caching the latent, but
 * it removes a [heads*(qk_nope+v), kv_lora_rank] matrix product from every past
 * position on every step, which otherwise makes decoding quadratic in an
 * expensive way rather than a cheap one.
 *
 * The rope half of the key is shared by all heads, as DeepSeek's decoupled
 * rotary key is a multi-query tensor.
 */
class KvCache final {
public:
    KvCache() = default;

    /** Sizes the cache and reserves storage for max_tokens positions. */
    void configure(std::size_t num_heads, std::size_t qk_nope_head_dim, std::size_t qk_rope_head_dim,
                   std::size_t v_head_dim, std::size_t max_tokens);

    /** Drops every cached position without releasing the reserved storage. */
    void clear() noexcept { token_count_ = 0U; }

    /**
     * Reserves one more position and returns its index, or capacity() when the
     * cache is full and the caller must stop generating.
     */
    [[nodiscard]] std::size_t append() noexcept;

    [[nodiscard]] std::size_t token_count() const noexcept { return token_count_; }
    [[nodiscard]] std::size_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] bool full() const noexcept { return token_count_ >= capacity_; }

    /** Total bytes this cache holds at its configured capacity. */
    [[nodiscard]] std::uint64_t reserved_bytes() const noexcept;

    // Writable accessors, used while appending the current position.
    [[nodiscard]] float* key_nope(std::size_t token) noexcept {
        return key_nope_.data() + token * num_heads_ * qk_nope_head_dim_;
    }
    [[nodiscard]] float* key_rope(std::size_t token) noexcept {
        return key_rope_.data() + token * qk_rope_head_dim_;
    }
    [[nodiscard]] float* value(std::size_t token) noexcept {
        return value_.data() + token * num_heads_ * v_head_dim_;
    }

    // Read-only accessors for one head of one cached position.
    [[nodiscard]] const float* key_nope(std::size_t token, std::size_t head) const noexcept {
        return key_nope_.data() + (token * num_heads_ + head) * qk_nope_head_dim_;
    }
    [[nodiscard]] const float* key_rope(std::size_t token) const noexcept {
        return key_rope_.data() + token * qk_rope_head_dim_;
    }
    [[nodiscard]] const float* value(std::size_t token, std::size_t head) const noexcept {
        return value_.data() + (token * num_heads_ + head) * v_head_dim_;
    }

private:
    std::size_t num_heads_{};
    std::size_t qk_nope_head_dim_{};
    std::size_t qk_rope_head_dim_{};
    std::size_t v_head_dim_{};
    std::size_t capacity_{};
    std::size_t token_count_{};

    std::vector<float> key_nope_;  ///< [capacity, heads, qk_nope_head_dim]
    std::vector<float> key_rope_;  ///< [capacity, qk_rope_head_dim] shared across heads
    std::vector<float> value_;     ///< [capacity, heads, v_head_dim]
};

}  // namespace litemind
