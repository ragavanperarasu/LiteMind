#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace litemind {

/**
 * @brief A read-only memory mapping of a whole file.
 *
 * Mapping rather than reading is what lets LiteMind run a 31 GB checkpoint on a
 * laptop. The virtual address space is reserved up front, but a page only costs
 * physical memory once it is touched, and the operating system reclaims cold
 * pages back to the SSD on its own when memory runs short. A token that routes
 * to six of sixty-four experts therefore pays for six experts, not for the
 * whole model.
 *
 * prefetch() and release() steer that behaviour explicitly: prefetch() asks the
 * kernel to stream a range in before it is needed, and release() drops a range
 * from the resident set once a layer is finished with it.
 */
class MappedFile final {
public:
    MappedFile() = default;
    ~MappedFile();

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept;
    MappedFile& operator=(MappedFile&& other) noexcept;

    /** Maps path read-only for the lifetime of this object. */
    [[nodiscard]] bool open(const std::filesystem::path& path, std::string& error);

    /** Unmaps the file and closes its handles. Safe to call more than once. */
    void close() noexcept;

    [[nodiscard]] bool is_open() const noexcept { return data_ != nullptr; }
    [[nodiscard]] const std::byte* data() const noexcept { return data_; }
    [[nodiscard]] std::uint64_t size() const noexcept { return size_; }
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

    /**
     * Returns a pointer to length bytes at offset, or nullptr when that range
     * does not lie completely inside the mapping.
     */
    [[nodiscard]] const std::byte* view(std::uint64_t offset, std::uint64_t length) const noexcept;

    /**
     * Asks the operating system to page a range in from the SSD. This is a
     * hint: it never fails the caller, and the data is valid with or without it.
     */
    void prefetch(std::uint64_t offset, std::uint64_t length) const noexcept;

    /**
     * Asks the operating system to drop a range from the resident set. The
     * bytes stay readable; touching them again pages them back in from the SSD.
     */
    void release(std::uint64_t offset, std::uint64_t length) const noexcept;

    /** The system page size, used to align prefetch and release ranges. */
    [[nodiscard]] static std::size_t page_size() noexcept;

private:
    std::filesystem::path path_;
    const std::byte* data_{nullptr};
    std::uint64_t size_{};
    void* file_handle_{nullptr};     ///< HANDLE on Windows, encoded file descriptor elsewhere.
    void* mapping_handle_{nullptr};  ///< HANDLE on Windows, unused elsewhere.
};

}  // namespace litemind
