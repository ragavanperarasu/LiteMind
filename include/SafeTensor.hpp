#pragma once

#include "Tensor.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace litemind {

/**
 * @brief Reads SafeTensors metadata without loading tensor payload bytes.
 *
 * SafeTensors keeps a JSON header behind an eight-byte little-endian length.
 * The reader validates that header and translates each tensor entry into
 * LiteMind metadata. It never allocates space for model weights.
 */
class SafeTensor final {
public:
    SafeTensor() = default;

    /** Opens and validates one SafeTensors shard's metadata header. */
    [[nodiscard]] bool open(const std::filesystem::path& path, std::string& error);

    [[nodiscard]] const std::filesystem::path& path() const noexcept;
    [[nodiscard]] const std::vector<Tensor>& tensors() const noexcept;

private:
    std::filesystem::path path_;
    std::vector<Tensor> tensors_;
};

}  // namespace litemind
