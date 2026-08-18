#include "Attention.hpp"
#include "Config.hpp"
#include "KvCache.hpp"
#include "TestSupport.hpp"

#include <cmath>
#include <numeric>
#include <vector>

using namespace test_support;

int main() {
    // ── The KV cache stores keys and values where attention expects them ────
    litemind::KvCache cache;
    cache.configure(/*heads=*/2U, /*qk_nope=*/4U, /*qk_rope=*/2U, /*v=*/3U, /*max_tokens=*/5U);
    check(cache.token_count() == 0U, "a fresh cache is empty");
    check(cache.capacity() == 5U, "the cache honours its configured capacity");

    const std::size_t slot = cache.append();
    check(slot == 0U, "the first append lands at position zero");
    check(cache.token_count() == 1U, "appending advances the length");

    // Head 0 gets 1..4, head 1 gets 5..8; the rope key is shared.
    float* keys = cache.key_nope(slot);
    for (std::size_t index = 0; index < 8U; ++index) {
        keys[index] = static_cast<float>(index + 1U);
    }
    float* values = cache.value(slot);
    for (std::size_t index = 0; index < 6U; ++index) {
        values[index] = static_cast<float>(index + 10U);
    }
    cache.key_rope(slot)[0] = 0.5F;
    cache.key_rope(slot)[1] = 1.5F;

    check(cache.key_nope(0U, 0U)[0] == 1.0F, "head 0's key starts at the first element");
    check(cache.key_nope(0U, 1U)[0] == 5.0F, "head 1's key starts after head 0's");
    check(cache.value(0U, 1U)[0] == 13.0F, "head 1's value starts after head 0's");
    check(cache.key_rope(0U)[1] == 1.5F, "the rope key is shared across heads");

    // ── Capacity is a hard stop, not an overflow ────────────────────────────
    for (std::size_t index = 1; index < 5U; ++index) {
        check(cache.append() == index, "append returns consecutive positions");
    }
    check(cache.full(), "the cache reports itself full at capacity");
    check(cache.append() == cache.capacity(), "a full cache refuses to append");

    cache.clear();
    check(cache.token_count() == 0U, "clearing empties the cache");
    check(cache.append() == 0U, "a cleared cache restarts at position zero");

    // ── Reserved memory scales with the configured context ──────────────────
    litemind::KvCache large;
    large.configure(16U, 128U, 64U, 128U, 1024U);
    // 1024 positions * (16*128 + 64 + 16*128) floats.
    const std::uint64_t expected =
        static_cast<std::uint64_t>(1024U) * (16U * 128U + 64U + 16U * 128U) * sizeof(float);
    check(large.reserved_bytes() == expected, "reserved_bytes matches the configured shape");

    return report("AttentionTest");
}
