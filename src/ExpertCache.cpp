#include "ExpertCache.hpp"

namespace litemind {

void ExpertCache::configure(const std::uint64_t budget_bytes) noexcept {
    budget_bytes_ = budget_bytes;
}

void ExpertCache::touch(const std::uint64_t key, const Block& block) {
    if (!enabled()) {
        return;
    }

    const auto existing = entries_.find(key);
    if (existing != entries_.end()) {
        // Already resident: just refresh its position in the recency order.
        recency_.splice(recency_.begin(), recency_, existing->second.position);
        existing->second.position = recency_.begin();
        ++hits_;
        return;
    }

    const std::uint64_t bytes = block.byte_size();

    // Make room first, so a large expert never transiently doubles the budget.
    while (resident_bytes_ + bytes > budget_bytes_ && !recency_.empty()) {
        evict_oldest();
    }

    // Ask the operating system to stream this expert in from the SSD. The
    // compute that follows would fault the pages in anyway; issuing the hint up
    // front lets the read run as one sequential request per matrix.
    block.gate.prefetch();
    block.up.prefetch();
    block.down.prefetch();

    recency_.push_front(key);
    Entry entry;
    entry.block = block;
    entry.bytes = bytes;
    entry.position = recency_.begin();
    entries_.emplace(key, entry);

    resident_bytes_ += bytes;
    bytes_streamed_ += bytes;
    ++loads_;
}

void ExpertCache::evict_oldest() {
    if (recency_.empty()) {
        return;
    }
    const std::uint64_t key = recency_.back();
    const auto entry = entries_.find(key);
    recency_.pop_back();
    if (entry == entries_.end()) {
        return;
    }

    // The bytes stay readable; only their claim on physical memory is dropped.
    entry->second.block.gate.release();
    entry->second.block.up.release();
    entry->second.block.down.release();

    resident_bytes_ -= entry->second.bytes;
    entries_.erase(entry);
    ++evictions_;
}

void ExpertCache::clear() {
    while (!recency_.empty()) {
        evict_oldest();
    }
    entries_.clear();
    resident_bytes_ = 0U;
}

}  // namespace litemind
