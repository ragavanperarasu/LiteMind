#pragma once

#include "SafeTensor.hpp"
#include "Tensor.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace litemind {

/**
 * @brief A borrowed, zero-copy handle onto one weight inside a mapped shard.
 *
 * The pointer addresses the shard's memory mapping directly. Reading through it
 * is what pulls pages from the SSD, so a view is cheap to create and only costs
 * physical memory once the runtime actually multiplies with it.
 */
struct WeightView final {
    const SafeTensor* shard{nullptr};
    const Tensor* meta{nullptr};
    const std::byte* bytes{nullptr};

    [[nodiscard]] bool valid() const noexcept { return bytes != nullptr; }
    [[nodiscard]] std::uint64_t element_count() const noexcept {
        return meta != nullptr ? meta->element_count() : 0U;
    }
    [[nodiscard]] std::uint64_t byte_size() const noexcept {
        return meta != nullptr ? meta->byte_size() : 0U;
    }
    [[nodiscard]] DataType data_type() const noexcept {
        return meta != nullptr ? meta->data_type() : DataType::Unknown;
    }

    /** Reinterprets the payload as raw BF16 bit patterns. Only valid for BF16 tensors. */
    [[nodiscard]] const std::uint16_t* as_bf16() const noexcept {
        return reinterpret_cast<const std::uint16_t*>(bytes);
    }

    /** Asks the operating system to stream this weight in from the SSD now. */
    void prefetch() const noexcept;

    /** Asks the operating system to drop this weight back to the SSD. */
    void release() const noexcept;
};

/** Counters describing how much weight traffic a run has generated. */
struct StoreStatistics final {
    std::uint64_t shards{};
    std::uint64_t tensors{};
    std::uint64_t mapped_bytes{};
    std::uint64_t resident_bytes{};   ///< Bytes currently pinned by the expert cache.
    std::uint64_t expert_loads{};     ///< Expert blocks paged in from the SSD.
    std::uint64_t expert_hits{};      ///< Expert blocks already resident when requested.
    std::uint64_t expert_evictions{}; ///< Expert blocks returned to the SSD.
};

/**
 * @brief Indexes every SafeTensors shard in a model directory.
 *
 * The store never copies weights. It resolves a tensor name to a pointer inside
 * a memory mapping and enforces the shape and dtype the runtime expects, so a
 * checkpoint that does not match config.json fails immediately with a specific
 * message instead of producing plausible-looking noise.
 */
class WeightStore final {
public:
    WeightStore() = default;

    /** Opens and indexes every .safetensors shard directly inside model_directory. */
    [[nodiscard]] bool open(const std::filesystem::path& model_directory, std::string& error);

    /** Returns a view of the named tensor, or an invalid view when it is absent. */
    [[nodiscard]] WeightView find(const std::string& name) const;

    /**
     * Returns a view of the named tensor, checking its shape and element type.
     * On any mismatch this fills error with the expected and actual layout.
     */
    [[nodiscard]] WeightView require(const std::string& name,
                                     const std::vector<std::size_t>& expected_shape,
                                     DataType expected_type, std::string& error) const;

    /**
     * Reads a tensor and widens it to float32. Used for the small per-layer
     * norm vectors, which are cheap to keep resident and are touched every step.
     */
    [[nodiscard]] bool read_float32(const std::string& name,
                                    const std::vector<std::size_t>& expected_shape,
                                    std::vector<float>& values, std::string& error) const;

    [[nodiscard]] bool contains(const std::string& name) const;
    [[nodiscard]] const StoreStatistics& statistics() const noexcept { return statistics_; }
    [[nodiscard]] StoreStatistics& statistics() noexcept { return statistics_; }

    /** Every tensor name in the store, sorted, for diagnostics. */
    [[nodiscard]] std::vector<std::string> names() const;

private:
    std::vector<std::unique_ptr<SafeTensor>> shards_;
    std::unordered_map<std::string, std::pair<std::size_t, std::size_t>> index_;
    StoreStatistics statistics_{};
};

/** Widens a BF16 bit pattern to float32. The exponent layout is identical. */
[[nodiscard]] float bfloat16_to_float(std::uint16_t bits) noexcept;

/** Widens an IEEE half-precision bit pattern to float32. */
[[nodiscard]] float float16_to_float(std::uint16_t bits) noexcept;

}  // namespace litemind
