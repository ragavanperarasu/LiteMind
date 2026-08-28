#include "ExpertCache.hpp"

#include <cstring>

namespace litemind {
namespace {

/// Matrices start on a cache-line boundary so the kernels see the alignment
/// they would have had inside the mapping.
constexpr std::uint64_t kAlignment = 64U;

[[nodiscard]] std::uint64_t align_up(const std::uint64_t value) noexcept {
    return (value + kAlignment - 1U) & ~(kAlignment - 1U);
}

}  // namespace

void ExpertCache::configure(const std::uint64_t budget_bytes) noexcept {
    budget_bytes_ = budget_bytes;
}

bool ExpertCache::reserve(const Block& representative) {
    if (!enabled() || arena_ != nullptr) {
        return copying();
    }

    const std::uint64_t gate = representative.gate.byte_size();
    const std::uint64_t up = representative.up.byte_size();
    const std::uint64_t down = representative.down.byte_size();
    if (gate == 0U || up == 0U || down == 0U) {
        return false;
    }

    layout_.gate_offset = 0U;
    layout_.gate_bytes = gate;
    layout_.up_offset = align_up(gate);
    layout_.up_bytes = up;
    layout_.down_offset = align_up(layout_.up_offset + up);
    layout_.down_bytes = down;
    layout_.stride = align_up(layout_.down_offset + down);

    const std::uint64_t capacity = budget_bytes_ / layout_.stride;
    if (capacity == 0U) {
        // Not even one expert fits. Leave the arena unallocated; touch() then
        // falls back to advisory hints rather than refusing to run.
        layout_ = SlotLayout{};
        return false;
    }

    capacity_ = static_cast<std::size_t>(capacity);
    arena_bytes_ = capacity * layout_.stride;
    // Default-initialised, so the pages are committed as experts are copied in
    // rather than by a multi-gigabyte zeroing pass at startup.
    arena_.reset(new std::byte[static_cast<std::size_t>(arena_bytes_)]);

    free_slots_.clear();
    free_slots_.reserve(capacity_);
    for (std::size_t slot = capacity_; slot-- > 0U;) {
        free_slots_.push_back(slot);
    }
    return true;
}

bool ExpertCache::fits(const Block& block) const noexcept {
    return block.gate.byte_size() == layout_.gate_bytes &&
           block.up.byte_size() == layout_.up_bytes &&
           block.down.byte_size() == layout_.down_bytes;
}

WeightView ExpertCache::copy_into(const WeightView& source, const std::size_t slot,
                                  const std::uint64_t offset) {
    std::byte* destination = arena_.get() + (slot * layout_.stride) + offset;

    // Ask for the read as one sequential request, take our own copy, then let
    // the file pages go. Keeping both would count this expert against RAM
    // twice: once in our arena and once in the page cache.
    source.prefetch();
    std::memcpy(destination, source.bytes, static_cast<std::size_t>(source.byte_size()));
    source.release();

    WeightView view;
    view.shard = nullptr;  // The arena is ours, so prefetch() and release() must do nothing.
    view.meta = source.meta;
    view.bytes = destination;
    return view;
}

ExpertCache::Block ExpertCache::touch(const std::uint64_t key, const Block& block) {
    if (!enabled()) {
        return block;
    }

    const auto existing = entries_.find(key);
    if (existing != entries_.end()) {
        // Already resident: refresh its position in the recency order and hand
        // back the copy we already hold.
        recency_.splice(recency_.begin(), recency_, existing->second.position);
        existing->second.position = recency_.begin();
        ++hits_;
        return existing->second.resident;
    }

    const std::uint64_t bytes = block.byte_size();

    if (!copying() || !fits(block)) {
        // No arena, or an expert this arena was not laid out for. Fall back to
        // asking the operating system to stream it in, which is still correct.
        block.gate.prefetch();
        block.up.prefetch();
        block.down.prefetch();
        bytes_streamed_ += bytes;
        ++loads_;
        return block;
    }

    // Make room first, so a load never transiently exceeds the budget.
    while (free_slots_.empty() && !recency_.empty()) {
        evict_oldest();
    }
    if (free_slots_.empty()) {
        return block;  // Unreachable while capacity_ > 0, but never compute on nothing.
    }

    const std::size_t slot = free_slots_.back();
    free_slots_.pop_back();

    Block resident;
    resident.gate = copy_into(block.gate, slot, layout_.gate_offset);
    resident.up = copy_into(block.up, slot, layout_.up_offset);
    resident.down = copy_into(block.down, slot, layout_.down_offset);

    recency_.push_front(key);
    Entry entry;
    entry.resident = resident;
    entry.slot = slot;
    entry.bytes = bytes;
    entry.position = recency_.begin();
    entries_.emplace(key, entry);

    resident_bytes_ += bytes;
    bytes_streamed_ += bytes;
    ++loads_;
    return resident;
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

    // The arena memory is ours, so eviction is just bookkeeping: the slot goes
    // back on the free list and the next load overwrites it. Nothing is asked
    // of the operating system, which is what makes the budget a hard ceiling.
    free_slots_.push_back(entry->second.slot);
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
