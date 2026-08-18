#pragma once

#include "WeightStore.hpp"

#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

namespace litemind {

/**
 * @brief Keeps a bounded working set of mixture-of-experts weights in RAM.
 *
 * DeepSeek-V2-Lite holds 64 routed experts per layer but sends each token to
 * six of them, so roughly a tenth of the expert weights are needed to decode
 * any one token. This class turns that property into a memory budget.
 *
 * Nothing is copied. An expert is "resident" when its pages have been streamed
 * in from the SSD through the shard's memory mapping, and it is evicted by
 * telling the operating system those pages may go back. Reading an evicted
 * expert is still correct; it simply faults the pages in again. The budget
 * therefore trades SSD traffic against RAM without ever changing the result.
 *
 * A zero budget disables the bookkeeping entirely and leaves residency to the
 * operating system's own page cache, which is the right choice when the machine
 * has enough RAM to hold the whole checkpoint.
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
     * Marks an expert as needed now, streaming it in and evicting the
     * least-recently-used experts if that pushes the working set over budget.
     */
    void touch(std::uint64_t key, const Block& block);

    /** Evicts everything and returns the working set to zero. */
    void clear();

    [[nodiscard]] bool enabled() const noexcept { return budget_bytes_ != 0U; }
    [[nodiscard]] std::uint64_t budget_bytes() const noexcept { return budget_bytes_; }
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
    struct Entry final {
        Block block;
        std::uint64_t bytes{};
        std::list<std::uint64_t>::iterator position;
    };

    void evict_oldest();

    std::uint64_t budget_bytes_{};
    std::uint64_t resident_bytes_{};
    std::list<std::uint64_t> recency_;  ///< Most recently used at the front.
    std::unordered_map<std::uint64_t, Entry> entries_;

    std::uint64_t loads_{};
    std::uint64_t hits_{};
    std::uint64_t evictions_{};
    std::uint64_t bytes_streamed_{};
};

}  // namespace litemind
