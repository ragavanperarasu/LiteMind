#pragma once

#include "MappedFile.hpp"
#include "Tensor.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace litemind {

/**
 * @brief One memory-mapped SafeTensors shard.
 *
 * A shard is an eight-byte little-endian header length, a JSON header naming
 * every tensor and its byte range, and then the payload. Opening a shard maps
 * the file and parses the header only; payload pages arrive from the SSD when
 * a weight is first touched, so opening a 9 GB shard costs milliseconds and
 * almost no physical memory.
 */
class SafeTensor final {
public:
    SafeTensor() = default;

    SafeTensor(const SafeTensor&) = delete;
    SafeTensor& operator=(const SafeTensor&) = delete;
    SafeTensor(SafeTensor&&) noexcept = default;
    SafeTensor& operator=(SafeTensor&&) noexcept = default;

    /** Maps the shard and parses its metadata header. */
    [[nodiscard]] bool open(const std::filesystem::path& path, std::string& error);

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return file_.path(); }
    [[nodiscard]] const std::vector<Tensor>& tensors() const noexcept { return tensors_; }
    [[nodiscard]] const MappedFile& file() const noexcept { return file_; }

    /** Returns the named tensor's metadata, or nullptr when this shard lacks it. */
    [[nodiscard]] const Tensor* find(const std::string& name) const;

    /**
     * Returns a pointer to a tensor's payload inside the mapping, or nullptr if
     * the declared byte range falls outside the file.
     */
    [[nodiscard]] const std::byte* payload(const Tensor& tensor) const noexcept;

    /** Hints that a tensor's pages should be streamed in from the SSD now. */
    void prefetch(const Tensor& tensor) const noexcept;

    /** Hints that a tensor's pages may be dropped back to the SSD. */
    void release(const Tensor& tensor) const noexcept;

private:
    MappedFile file_;
    std::vector<Tensor> tensors_;
    std::unordered_map<std::string, std::size_t> index_;
};

}  // namespace litemind
