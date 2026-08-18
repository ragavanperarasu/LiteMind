#include "MappedFile.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <memoryapi.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace litemind {
namespace {

#if defined(_WIN32)
/** Formats the last Win32 error as readable text rather than a bare number. */
[[nodiscard]] std::string last_error_text() {
    const DWORD code = ::GetLastError();
    LPSTR buffer = nullptr;
    const DWORD length = ::FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), reinterpret_cast<LPSTR>(&buffer), 0,
        nullptr);

    std::string message = length != 0U && buffer != nullptr ? std::string(buffer, length) : std::string();
    if (buffer != nullptr) {
        ::LocalFree(buffer);
    }
    while (!message.empty() && (message.back() == '\n' || message.back() == '\r' || message.back() == ' ')) {
        message.pop_back();
    }
    if (message.empty()) {
        message = "Windows error " + std::to_string(code);
    }
    return message;
}
#endif

#if !defined(_WIN32)
/** POSIX file descriptors are stored in the void* handle slot. */
[[nodiscard]] void* encode_descriptor(const int descriptor) noexcept {
    return reinterpret_cast<void*>(static_cast<std::intptr_t>(descriptor));
}

[[nodiscard]] int decode_descriptor(void* const slot) noexcept {
    return static_cast<int>(reinterpret_cast<std::intptr_t>(slot));
}
#endif

}  // namespace

MappedFile::~MappedFile() { close(); }

MappedFile::MappedFile(MappedFile&& other) noexcept
    : path_(std::move(other.path_)),
      data_(other.data_),
      size_(other.size_),
      file_handle_(other.file_handle_),
      mapping_handle_(other.mapping_handle_) {
    other.data_ = nullptr;
    other.size_ = 0U;
    other.file_handle_ = nullptr;
    other.mapping_handle_ = nullptr;
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
    if (this != &other) {
        close();
        path_ = std::move(other.path_);
        data_ = other.data_;
        size_ = other.size_;
        file_handle_ = other.file_handle_;
        mapping_handle_ = other.mapping_handle_;
        other.data_ = nullptr;
        other.size_ = 0U;
        other.file_handle_ = nullptr;
        other.mapping_handle_ = nullptr;
    }
    return *this;
}

std::size_t MappedFile::page_size() noexcept {
#if defined(_WIN32)
    static const std::size_t cached = [] {
        SYSTEM_INFO information{};
        ::GetSystemInfo(&information);
        return static_cast<std::size_t>(information.dwPageSize);
    }();
#else
    static const std::size_t cached = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
#endif
    return cached == 0U ? 4096U : cached;
}

#if defined(_WIN32)

bool MappedFile::open(const std::filesystem::path& path, std::string& error) {
    close();

    const HANDLE file = ::CreateFileW(path.wstring().c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                      OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (file == INVALID_HANDLE_VALUE) {
        error = "Unable to open " + path.string() + ": " + last_error_text();
        return false;
    }

    LARGE_INTEGER file_size{};
    if (::GetFileSizeEx(file, &file_size) == 0 || file_size.QuadPart <= 0) {
        error = "Unable to determine the size of " + path.string() + ": " + last_error_text();
        ::CloseHandle(file);
        return false;
    }

    const HANDLE mapping = ::CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping == nullptr) {
        error = "Unable to create a file mapping for " + path.string() + ": " + last_error_text()
              + " (a 32-bit build cannot map a file this large; build for x64)";
        ::CloseHandle(file);
        return false;
    }

    const LPVOID address = ::MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, 0);
    if (address == nullptr) {
        error = "Unable to map " + path.string() + " into memory: " + last_error_text();
        ::CloseHandle(mapping);
        ::CloseHandle(file);
        return false;
    }

    path_ = path;
    data_ = static_cast<const std::byte*>(address);
    size_ = static_cast<std::uint64_t>(file_size.QuadPart);
    file_handle_ = file;
    mapping_handle_ = mapping;
    return true;
}

void MappedFile::close() noexcept {
    if (data_ != nullptr) {
        ::UnmapViewOfFile(data_);
        data_ = nullptr;
    }
    if (mapping_handle_ != nullptr) {
        ::CloseHandle(mapping_handle_);
        mapping_handle_ = nullptr;
    }
    if (file_handle_ != nullptr) {
        ::CloseHandle(file_handle_);
        file_handle_ = nullptr;
    }
    size_ = 0U;
    path_.clear();
}

void MappedFile::prefetch(const std::uint64_t offset, const std::uint64_t length) const noexcept {
    if (data_ == nullptr || length == 0U || offset >= size_) {
        return;
    }
    const std::uint64_t clamped = std::min(length, size_ - offset);

    WIN32_MEMORY_RANGE_ENTRY range{};
    range.VirtualAddress = const_cast<std::byte*>(data_ + offset);
    range.NumberOfBytes = static_cast<SIZE_T>(clamped);
    // A hint only: on Windows 7 the entry point is absent and this simply fails.
    ::PrefetchVirtualMemory(::GetCurrentProcess(), 1U, &range, 0U);
}

void MappedFile::release(const std::uint64_t offset, const std::uint64_t length) const noexcept {
    if (data_ == nullptr || length == 0U || offset >= size_) {
        return;
    }
    const std::size_t granularity = page_size();
    const std::uint64_t clamped = std::min(length, size_ - offset);
    const std::uint64_t first = (offset + granularity - 1U) / granularity * granularity;
    const std::uint64_t last = (offset + clamped) / granularity * granularity;
    if (last <= first) {
        return;
    }

    // VirtualUnlock on pages that were never locked reports ERROR_NOT_LOCKED but
    // still evicts them from the working set, which is exactly what is wanted:
    // the bytes stay readable and are re-read from the SSD on the next touch.
    ::VirtualUnlock(const_cast<std::byte*>(data_ + first), static_cast<SIZE_T>(last - first));
}

#else

bool MappedFile::open(const std::filesystem::path& path, std::string& error) {
    close();

    const int descriptor = ::open(path.c_str(), O_RDONLY);
    if (descriptor < 0) {
        error = "Unable to open " + path.string() + ".";
        return false;
    }

    struct stat status {};
    if (::fstat(descriptor, &status) != 0 || status.st_size <= 0) {
        error = "Unable to determine the size of " + path.string() + ".";
        ::close(descriptor);
        return false;
    }

    void* const address = ::mmap(nullptr, static_cast<std::size_t>(status.st_size), PROT_READ,
                                 MAP_PRIVATE, descriptor, 0);
    if (address == MAP_FAILED) {
        error = "Unable to map " + path.string() + " into memory.";
        ::close(descriptor);
        return false;
    }

    path_ = path;
    data_ = static_cast<const std::byte*>(address);
    size_ = static_cast<std::uint64_t>(status.st_size);
    file_handle_ = encode_descriptor(descriptor);
    mapping_handle_ = nullptr;
    return true;
}

void MappedFile::close() noexcept {
    if (data_ != nullptr) {
        ::munmap(const_cast<std::byte*>(data_), static_cast<std::size_t>(size_));
        data_ = nullptr;
    }
    if (file_handle_ != nullptr) {
        ::close(decode_descriptor(file_handle_));
        file_handle_ = nullptr;
    }
    size_ = 0U;
    path_.clear();
}

void MappedFile::prefetch(const std::uint64_t offset, const std::uint64_t length) const noexcept {
    if (data_ == nullptr || length == 0U || offset >= size_) {
        return;
    }
    const std::size_t granularity = page_size();
    const std::uint64_t clamped = std::min(length, size_ - offset);
    const std::uint64_t first = offset / granularity * granularity;
    const std::uint64_t last = std::min(size_, offset + clamped);
    ::madvise(const_cast<std::byte*>(data_ + first), static_cast<std::size_t>(last - first), MADV_WILLNEED);
}

void MappedFile::release(const std::uint64_t offset, const std::uint64_t length) const noexcept {
    if (data_ == nullptr || length == 0U || offset >= size_) {
        return;
    }
    const std::size_t granularity = page_size();
    const std::uint64_t clamped = std::min(length, size_ - offset);
    const std::uint64_t first = (offset + granularity - 1U) / granularity * granularity;
    const std::uint64_t last = (offset + clamped) / granularity * granularity;
    if (last <= first) {
        return;
    }
    ::madvise(const_cast<std::byte*>(data_ + first), static_cast<std::size_t>(last - first), MADV_DONTNEED);
}

#endif

const std::byte* MappedFile::view(const std::uint64_t offset, const std::uint64_t length) const noexcept {
    if (data_ == nullptr || offset > size_ || length > size_ - offset) {
        return nullptr;
    }
    return data_ + offset;
}

}  // namespace litemind
