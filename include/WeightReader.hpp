#pragma once

#include "SafeTensor.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace litemind {

/**
 * @brief Reads tensor payloads from a parsed SafeTensors shard.
 *
 * The reader owns no persistent cache. It provides the safe, synchronous data
 * access boundary that a future memory cache and CPU runtime will build upon.
 */
class WeightReader final {
public:
    /** Reads the raw on-disk payload bytes for one tensor. */
    [[nodiscard]] bool read_bytes(const SafeTensor& shard, const Tensor& tensor,
                                  std::vector<std::byte>& bytes, std::string& error) const;

    /** Reads a BF16 tensor and widens each element to a CPU float. */
    [[nodiscard]] bool read_bfloat16(const SafeTensor& shard, const Tensor& tensor,
                                     std::vector<float>& values, std::string& error) const;
};

}  // namespace litemind
