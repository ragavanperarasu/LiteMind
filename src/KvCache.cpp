#include "KvCache.hpp"

#include <stdexcept>

namespace litemind {

KvCache::KvCache(const std::size_t head_dimension) : head_dimension_(head_dimension) {
    if (head_dimension_ == 0U) {
        throw std::invalid_argument("KV-cache head dimension must be positive.");
    }
}

void KvCache::append(const std::span<const float> key, const std::span<const float> value) {
    if (key.size() != head_dimension_ || value.size() != head_dimension_) {
        throw std::invalid_argument("KV-cache entries must match the configured head dimension.");
    }
    keys_.insert(keys_.end(), key.begin(), key.end());
    values_.insert(values_.end(), value.begin(), value.end());
}

void KvCache::clear() noexcept {
    keys_.clear();
    values_.clear();
}

std::size_t KvCache::head_dimension() const noexcept { return head_dimension_; }
std::size_t KvCache::token_count() const noexcept { return keys_.size() / head_dimension_; }

std::span<const float> KvCache::key(const std::size_t token_index) const {
    if (token_index >= token_count()) {
        throw std::out_of_range("KV-cache key index is outside the current sequence.");
    }
    return {keys_.data() + token_index * head_dimension_, head_dimension_};
}

std::span<const float> KvCache::value(const std::size_t token_index) const {
    if (token_index >= token_count()) {
        throw std::out_of_range("KV-cache value index is outside the current sequence.");
    }
    return {values_.data() + token_index * head_dimension_, head_dimension_};
}

}  // namespace litemind
