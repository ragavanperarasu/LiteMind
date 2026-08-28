#pragma once

#include "WeightStore.hpp"

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <unordered_map>
#include <vector>

namespace litemind {

/**
 * @brief Holds a bounded working set of mixture-of-experts weights in RAM.
 *
 * DeepSeek-V2-Lite holds 64 routed experts per layer but sends each token to
 * six of them, so roughly a tenth of the expert weights are needed to decode
 * any one token. This class turns that property into a memory budget.
 *
 * With a budget set, the six experts the router asked for are *copied* out of
 * the memory mapping into an arena this class owns, and the compute runs
 * against that copy. The file pages are released immediately afterwards, so an
 * expert is never counted against RAM twice. The arena is allocated once and
 * never grows, which is what makes the budget a ceiling rather than a request:
 * eviction is ours to decide, not the operating system's.
 *
 * With no budget, nothing is copied. Views address the mapping directly and
 * residency is left to the operating system's page cache, which is the faster
 * choice when the machine has enough RAM to hold the working set anyway.
 */
class ExpertCache final {
public:
    /** The three matrices of one expert's SwiGLU feed-forward block. */
    struct Block final {
        WeightView gate;
        WeightView up;
        WeightView down;

        [[nodiscard]] std::uint64_t byte_size() const noexcept {
            return gate.byte_size() + up.byte_size() + down.byte_size();
        }
    };

    ExpertCache() = default;

    /** Sets the residency budget in bytes. Zero leaves paging to the system. */
    void configure(std::uint64_t budget_bytes) noexcept;

    /**
     * Allocates the arena from a representative expert block, so the capacity is
     * known before the first prompt rather than part-way through it.
     *
     * Every routed expert in a DeepSeek-V2 model has the same shape, so one slot
     * size serves all of them. Returns false when the budget cannot hold even a
     * single expert, in which case the cache degrades to advisory paging hints.
     */
    bool reserve(const Block& representative);

    /**
     * Marks an expert as needed now and returns the block to compute with.
     *
     * When the arena is active the expert is copied in, evicting the
     * least-recently-used experts to make room, and the returned block addresses
     * that copy. Otherwise the block is returned unchanged and reads go through
     * the memory mapping as before.
     */
    [[nodiscard]] Block touch(std::uint64_t key, const Block& block);

    /** Returns every expert to the free list. The arena itself is kept. */
    void clear();

    [[nodiscard]] bool enabled() const noexcept { return budget_bytes_ != 0U; }
    /** True once an arena is allocated, meaning experts are copied into RAM. */
    [[nodiscard]] bool copying() const noexcept { return capacity_ != 0U; }
    [[nodiscard]] std::uint64_t budget_bytes() const noexcept { return budget_bytes_; }
    [[nodiscard]] std::uint64_t arena_bytes() const noexcept { return arena_bytes_; }
    [[nodiscard]] std::size_t capacity_experts() const noexcept { return capacity_; }
    [[nodiscard]] std::uint64_t resident_bytes() const noexcept { return resident_bytes_; }
    [[nodiscard]] std::size_t resident_experts() const noexcept { return entries_.size(); }

    [[nodiscard]] std::uint64_t loads() const noexcept { return loads_; }
    [[nodiscard]] std::uint64_t hits() const noexcept { return hits_; }
    [[nodiscard]] std::uint64_t evictions() const noexcept { return evictions_; }
    [[nodiscard]] std::uint64_t bytes_streamed() const noexcept { return bytes_streamed_; }

    /** Builds the cache key for one expert. */
    [[nodiscard]] static std::uint64_t make_key(std::size_t layer, std::size_t expert) noexcept {
        return (static_cast<std::uint64_t>(layer) << 32U) | static_cast<std::uint32_t>(expert);
    }

private:
    /** Byte offsets of the three matrices inside one arena slot. */
    struct SlotLayout final {
        std::uint64_t gate_offset{};
        std::uint64_t gate_bytes{};
        std::uint64_t up_offset{};
        std::uint64_t up_bytes{};
        std::uint64_t down_offset{};
        std::uint64_t down_bytes{};
        std::uint64_t stride{};
    };

    struct Entry final {
        Block resident;
        std::size_t slot{};
        std::uint64_t bytes{};
        std::list<std::uint64_t>::iterator position;
    };

    /** True when a block has exactly the shape the arena was laid out for. */
    [[nodiscard]] bool fits(const Block& block) const noexcept;

    /** Copies one matrix into a slot and returns a view addressing the copy. */
    [[nodiscard]] WeightView copy_into(const WeightView& source, std::size_t slot,
                                       std::uint64_t offset);

    void evict_oldest();

    std::uint64_t budget_bytes_{};
    std::uint64_t resident_bytes_{};

    std::unique_ptr<std::byte[]> arena_;
    std::uint64_t arena_bytes_{};
    std::size_t capacity_{};
    SlotLayout layout_{};
    std::vector<std::size_t> free_slots_;

    std::list<std::uint64_t> recency_;  ///< Most recently used at the front.
    std::unordered_map<std::uint64_t, Entry> entries_;

    std::uint64_t loads_{};
    std::uint64_t hits_{};
    std::uint64_t evictions_{};
    std::uint64_t bytes_streamed_{};
};

}  // namespace litemind
