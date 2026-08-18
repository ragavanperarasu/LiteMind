#include "SafeTensor.hpp"
#include "WeightReader.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

void write_u64_little_endian(std::ofstream& file, const std::uint64_t value) {
    for (std::size_t byte = 0; byte < 8U; ++byte) {
        file.put(static_cast<char>((value >> (byte * 8U)) & 0xffU));
    }
}

}  // namespace

int main() {
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "litemind_weight_reader_test.safetensors";
    const std::string header = R"({"test.weight":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]}})";

    {
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        if (!file) {
            std::cerr << "Unable to create toy SafeTensors shard.\n";
            return 1;
        }
        write_u64_little_endian(file, header.size());
        file.write(header.data(), static_cast<std::streamsize>(header.size()));
        constexpr std::array<unsigned char, 6U> payload{0x80U, 0x3fU, 0x20U, 0xc0U, 0x00U, 0x3fU};
        file.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    }

    litemind::SafeTensor shard;
    std::string error;
    if (!shard.open(path, error) || shard.tensors().size() != 1U) {
        std::cerr << error << '\n';
        return 1;
    }

    litemind::WeightReader reader;
    std::vector<float> values;
    if (!reader.read_bfloat16(shard, shard.tensors().front(), values, error)) {
        std::cerr << error << '\n';
        return 1;
    }
    std::error_code remove_error;
    std::filesystem::remove(path, remove_error);

    if (values.size() != 3U || values[0] != 1.0F || values[1] != -2.5F || values[2] != 0.5F) {
        std::cerr << "BF16 conversion produced unexpected values.\n";
        return 1;
    }
    return 0;
}
