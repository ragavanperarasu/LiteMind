#include "WeightReader.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <fstream>
#include <limits>

namespace litemind {
namespace {

constexpr std::size_t read_chunk_size = 64U * 1024U * 1024U;

[[nodiscard]] bool belongs_to_shard(const SafeTensor& shard, const Tensor& tensor) {
    return std::any_of(shard.tensors().begin(), shard.tensors().end(), [&tensor](const Tensor& candidate) {
        return candidate.name() == tensor.name() && candidate.offset() == tensor.offset()
            && candidate.byte_size() == tensor.byte_size() && candidate.data_type() == tensor.data_type();
    });
}

[[nodiscard]] bool read_exact(std::ifstream& file, std::vector<std::byte>& bytes) {
    std::size_t offset{};
    while (offset < bytes.size()) {
        const std::size_t remaining = bytes.size() - offset;
        const std::size_t chunk_size = std::min(remaining, read_chunk_size);
        file.read(reinterpret_cast<char*>(bytes.data() + offset), static_cast<std::streamsize>(chunk_size));
        if (file.gcount() != static_cast<std::streamsize>(chunk_size)) {
            return false;
        }
        offset += chunk_size;
    }
    return true;
}

}  // namespace

bool WeightReader::read_bytes(const SafeTensor& shard, const Tensor& tensor,
                              std::vector<std::byte>& bytes, std::string& error) const {
    bytes.clear();
    if (shard.path().empty() || !belongs_to_shard(shard, tensor)) {
        error = "Tensor metadata does not belong to the supplied SafeTensors shard.";
        return false;
    }
    if (tensor.byte_size() > std::numeric_limits<std::size_t>::max()) {
        error = "Tensor is too large to address on this platform.";
        return false;
    }
    if (tensor.offset() > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
        error = "Tensor offset cannot be represented by this platform's file stream.";
        return false;
    }

    std::ifstream file(shard.path(), std::ios::binary);
    if (!file) {
        error = "Unable to open SafeTensors shard: " + shard.path().string();
        return false;
    }
    file.seekg(static_cast<std::streamoff>(tensor.offset()), std::ios::beg);
    if (!file) {
        error = "Unable to seek to tensor payload: " + tensor.name();
        return false;
    }

    bytes.resize(static_cast<std::size_t>(tensor.byte_size()));
    if (!read_exact(file, bytes)) {
        bytes.clear();
        error = "SafeTensors shard ended before tensor payload was complete: " + tensor.name();
        return false;
    }
    return true;
}

bool WeightReader::read_bfloat16(const SafeTensor& shard, const Tensor& tensor,
                                 std::vector<float>& values, std::string& error) const {
    values.clear();
    if (tensor.data_type() != DataType::BFloat16) {
        error = "Tensor is not stored as BF16: " + tensor.name();
        return false;
    }

    std::vector<std::byte> bytes;
    if (!read_bytes(shard, tensor, bytes, error)) {
        return false;
    }
    if (bytes.size() % 2U != 0U) {
        error = "BF16 tensor payload has an invalid byte count: " + tensor.name();
        return false;
    }

    values.resize(bytes.size() / 2U);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const std::uint16_t bits = static_cast<std::uint16_t>(std::to_integer<unsigned char>(bytes[index * 2U]))
            | (static_cast<std::uint16_t>(std::to_integer<unsigned char>(bytes[index * 2U + 1U])) << 8U);
        values[index] = std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
    }
    return true;
}

}  // namespace litemind
