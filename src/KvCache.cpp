#include "KvCache.hpp"

namespace litemind {

void KvCache::configure(const std::size_t num_heads, const std::size_t qk_nope_head_dim,
                        const std::size_t qk_rope_head_dim, const std::size_t v_head_dim,
                        const std::size_t max_tokens) {
    num_heads_ = num_heads;
    qk_nope_head_dim_ = qk_nope_head_dim;
    qk_rope_head_dim_ = qk_rope_head_dim;
    v_head_dim_ = v_head_dim;
    capacity_ = max_tokens;
    token_count_ = 0U;

    key_nope_.assign(capacity_ * num_heads_ * qk_nope_head_dim_, 0.0F);
    key_rope_.assign(capacity_ * qk_rope_head_dim_, 0.0F);
    value_.assign(capacity_ * num_heads_ * v_head_dim_, 0.0F);
}

std::size_t KvCache::append() noexcept {
    if (token_count_ >= capacity_) {
        return capacity_;
    }
    return token_count_++;
}

std::uint64_t KvCache::reserved_bytes() const noexcept {
    const auto elements = static_cast<std::uint64_t>(key_nope_.size() + key_rope_.size() + value_.size());
    return elements * sizeof(float);
}

}  // namespace litemind
