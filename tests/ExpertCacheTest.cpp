#include "ExpertCache.hpp"
#include "TestSupport.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

using litemind::DataType;
using litemind::ExpertCache;
using litemind::Tensor;
using litemind::WeightView;
using namespace test_support;

namespace {

/**
 * One expert's worth of storage standing in for a memory mapping. A view with a
 * null shard behaves exactly like a mapped one apart from prefetch() and
 * release(), which the cache is allowed to call and which do nothing here.
 */
struct FakeExpert final {
    static constexpr std::size_t kRows = 8;
    static constexpr std::size_t kColumns = 4;
    static constexpr std::size_t kBytes = kRows * kColumns * 2U;  // BF16

    Tensor gate_meta{"gate", {kRows, kColumns}, DataType::BFloat16, 0U};
    Tensor up_meta{"up", {kRows, kColumns}, DataType::BFloat16, 0U};
    Tensor down_meta{"down", {kColumns, kRows}, DataType::BFloat16, 0U};
    std::vector<std::byte> gate_bytes{kBytes, std::byte{0}};
    std::vector<std::byte> up_bytes{kBytes, std::byte{0}};
    std::vector<std::byte> down_bytes{kBytes, std::byte{0}};

    explicit FakeExpert(const std::uint8_t fill) {
        gate_bytes.assign(kBytes, std::byte{fill});
        up_bytes.assign(kBytes, std::byte{static_cast<std::uint8_t>(fill + 1U)});
        down_bytes.assign(kBytes, std::byte{static_cast<std::uint8_t>(fill + 2U)});
    }

    [[nodiscard]] ExpertCache::Block block() const {
        ExpertCache::Block block;
        block.gate = WeightView{nullptr, &gate_meta, gate_bytes.data()};
        block.up = WeightView{nullptr, &up_meta, up_bytes.data()};
        block.down = WeightView{nullptr, &down_meta, down_bytes.data()};
        return block;
    }

    /** True when every byte of a view equals value. */
    [[nodiscard]] static bool all_equal(const WeightView& view, const std::uint8_t value) {
        for (std::uint64_t index = 0; index < view.byte_size(); ++index) {
            if (view.bytes[index] != std::byte{value}) {
                return false;
            }
        }
        return true;
    }
};

}  // namespace

int main() {
    const std::uint64_t expert_bytes = 3U * FakeExpert::kBytes;

    // ── With no budget the cache must stay out of the way ────────────────────
    {
        ExpertCache cache;
        FakeExpert expert(0x11U);
        const ExpertCache::Block source = expert.block();
        const ExpertCache::Block resident = cache.touch(ExpertCache::make_key(0, 0), source);

        check(!cache.enabled(), "a zero budget leaves the cache disabled");
        check(!cache.copying(), "a disabled cache allocates no arena");
        check(resident.gate.bytes == source.gate.bytes, "views still address the mapping");
        check(resident.down.bytes == source.down.bytes, "every matrix is passed through");
    }

    // ── A budget too small for one expert degrades rather than failing ───────
    {
        ExpertCache cache;
        cache.configure(expert_bytes / 2U);
        FakeExpert expert(0x21U);
        check(!cache.reserve(expert.block()), "reserve reports that nothing fits");
        check(cache.enabled(), "the budget is still recorded");
        check(!cache.copying(), "no arena is allocated when one expert will not fit");

        const ExpertCache::Block resident = cache.touch(ExpertCache::make_key(0, 0), expert.block());
        check(resident.gate.bytes == expert.gate_bytes.data(), "the mapped view is handed back");
        check(cache.loads() == 1U, "the load is still counted");
    }

    // ── The copy must be independent of the mapping it came from ────────────
    {
        ExpertCache cache;
        cache.configure(expert_bytes * 4U);  // Slot padding means this is not exactly 4 experts.
        FakeExpert expert(0x31U);
        check(cache.reserve(expert.block()), "an arena is allocated");
        check(cache.copying(), "the cache reports that it copies");
        check(cache.capacity_experts() >= 1U, "the arena holds at least one expert");
        check(cache.arena_bytes() <= cache.budget_bytes(), "the arena never exceeds the budget");

        const ExpertCache::Block source = expert.block();
        const ExpertCache::Block resident = cache.touch(ExpertCache::make_key(0, 0), source);
        check(resident.gate.bytes != source.gate.bytes, "gate addresses the arena, not the mapping");
        check(resident.up.bytes != source.up.bytes, "up addresses the arena");
        check(resident.down.bytes != source.down.bytes, "down addresses the arena");
        check(resident.gate.byte_size() == FakeExpert::kBytes, "the copy keeps its metadata");
        check(FakeExpert::all_equal(resident.gate, 0x31U), "gate bytes copied faithfully");
        check(FakeExpert::all_equal(resident.up, 0x32U), "up bytes copied faithfully");
        check(FakeExpert::all_equal(resident.down, 0x33U), "down bytes copied faithfully");

        // Overwrite the "mapping" behind the cache's back. A real copy is
        // unaffected; a view that still points at the source would change.
        expert.gate_bytes.assign(FakeExpert::kBytes, std::byte{0xFFU});
        check(FakeExpert::all_equal(resident.gate, 0x31U), "the copy survives the source changing");

        // Asking again is a hit and returns the same copy.
        const ExpertCache::Block again = cache.touch(ExpertCache::make_key(0, 0), expert.block());
        check(cache.hits() == 1U, "the second request is a hit");
        check(cache.loads() == 1U, "a hit does not count as a load");
        check(again.gate.bytes == resident.gate.bytes, "a hit returns the existing copy");
    }

    // ── The budget is a ceiling: eviction is least-recently-used ────────────
    {
        ExpertCache cache;
        FakeExpert probe(0x40U);
        // Two slots exactly, once slot padding is taken into account.
        FakeExpert first(0x41U);
        cache.configure(expert_bytes * 8U);
        check(cache.reserve(probe.block()), "an arena is allocated");
        const std::size_t capacity = cache.capacity_experts();
        check(capacity >= 2U, "the arena holds at least two experts for this test");

        std::vector<std::unique_ptr<FakeExpert>> experts;
        for (std::size_t index = 0; index <= capacity; ++index) {
            experts.push_back(std::make_unique<FakeExpert>(static_cast<std::uint8_t>(0x50U + index)));
        }

        // Fill the arena exactly.
        for (std::size_t index = 0; index < capacity; ++index) {
            static_cast<void>(cache.touch(ExpertCache::make_key(0, index), experts[index]->block()));
        }
        check(cache.resident_experts() == capacity, "the arena fills to capacity");
        check(cache.evictions() == 0U, "nothing is evicted while there is room");
        check(cache.resident_bytes() <= cache.budget_bytes(), "residency stays within budget");

        // Touch the oldest so it is no longer the eviction candidate, then
        // overflow by one: the second-oldest must go instead.
        static_cast<void>(cache.touch(ExpertCache::make_key(0, 0), experts[0]->block()));
        static_cast<void>(cache.touch(ExpertCache::make_key(0, capacity), experts[capacity]->block()));

        check(cache.resident_experts() == capacity, "capacity is never exceeded");
        check(cache.evictions() == 1U, "exactly one expert was evicted");
        check(cache.resident_bytes() <= cache.budget_bytes(), "residency still within budget");

        // Expert 0 was refreshed, so it must still be resident: a request for it
        // is a hit, while the expert evicted in its place reloads.
        const std::uint64_t hits_before = cache.hits();
        static_cast<void>(cache.touch(ExpertCache::make_key(0, 0), experts[0]->block()));
        check(cache.hits() == hits_before + 1U, "the refreshed expert survived eviction");

        const std::uint64_t loads_before = cache.loads();
        static_cast<void>(cache.touch(ExpertCache::make_key(0, 1), experts[1]->block()));
        check(cache.loads() == loads_before + 1U, "the least-recently-used expert was the one dropped");

        cache.clear();
        check(cache.resident_experts() == 0U, "clear empties the arena");
        check(cache.resident_bytes() == 0U, "clear returns residency to zero");

        // The arena is kept, so the budget still applies after a clear.
        check(cache.copying(), "the arena survives a clear");
        static_cast<void>(cache.touch(ExpertCache::make_key(1, 0), first.block()));
        check(cache.resident_experts() == 1U, "the cache is usable again after a clear");
    }

    return report("ExpertCacheTest");
}
