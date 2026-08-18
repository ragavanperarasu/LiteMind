#include "SafeTensor.hpp"
#include "TestSupport.hpp"
#include "WeightStore.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace test_support;

namespace {

/** Writes a minimal SafeTensors shard so the reader can be tested offline. */
bool write_shard(const std::filesystem::path& path, const std::string& header,
                 const std::vector<unsigned char>& payload) {
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file) {
        return false;
    }
    const auto header_size = static_cast<std::uint64_t>(header.size());
    for (std::size_t byte = 0; byte < 8U; ++byte) {
        file.put(static_cast<char>((header_size >> (byte * 8U)) & 0xffU));
    }
    file.write(header.data(), static_cast<std::streamsize>(header.size()));
    file.write(reinterpret_cast<const char*>(payload.data()),
               static_cast<std::streamsize>(payload.size()));
    return file.good();
}

}  // namespace

int main() {
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() / "litemind_safetensor_test";
    std::error_code filesystem_error;
    std::filesystem::remove_all(directory, filesystem_error);
    std::filesystem::create_directories(directory, filesystem_error);

    const std::filesystem::path path = directory / "model.safetensors";
    // 1.0, -2.5 and 0.5 as little-endian BF16, then a 2x2 float32 matrix.
    const std::string header =
        R"({"__metadata__":{"format":"pt"},)"
        R"("test.weight":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]},)"
        R"("norm.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[8,24]}})";
    const std::vector<unsigned char> payload{
        0x80, 0x3f, 0x20, 0xc0, 0x00, 0x3f,  // BF16: 1.0, -2.5, 0.5
        0x00, 0x00,                          // padding to an 8-byte boundary
        0x00, 0x00, 0x80, 0x3f,              // F32: 1.0
        0x00, 0x00, 0x00, 0x40,              // F32: 2.0
        0x00, 0x00, 0x40, 0x40,              // F32: 3.0
        0x00, 0x00, 0x80, 0x40,              // F32: 4.0
    };
    check(write_shard(path, header, payload), "the toy shard is written");

    litemind::SafeTensor shard;
    std::string error;
    check(shard.open(path, error), "the shard opens: " + error);
    check(shard.tensors().size() == 2U, "__metadata__ is skipped and both tensors are indexed");

    const litemind::Tensor* tensor = shard.find("test.weight");
    check(tensor != nullptr, "a tensor resolves by name");
    if (tensor != nullptr) {
        check(tensor->data_type() == litemind::DataType::BFloat16, "the dtype reads back");
        check(tensor->byte_size() == 6U, "the byte size follows from the shape");
        check(tensor->element_count() == 3U, "the element count follows from the shape");

        // The payload must be readable straight out of the mapping.
        const std::uint16_t* raw = reinterpret_cast<const std::uint16_t*>(shard.payload(*tensor));
        check(raw != nullptr, "the payload maps inside the file");
        if (raw != nullptr) {
            check_close(litemind::bfloat16_to_float(raw[0]), 1.0, 1e-6, "BF16 1.0 widens");
            check_close(litemind::bfloat16_to_float(raw[1]), -2.5, 1e-6, "BF16 -2.5 widens");
            check_close(litemind::bfloat16_to_float(raw[2]), 0.5, 1e-6, "BF16 0.5 widens");
        }
    }

    // The store indexes the directory and enforces the expected layout.
    litemind::WeightStore store;
    check(store.open(directory, error), "the store opens the directory: " + error);

    std::vector<float> values;
    check(store.read_float32("norm.weight", {2U, 2U}, values, error),
          "a float32 tensor reads back: " + error);
    check(values.size() == 4U && values[3] == 4.0F, "float32 values survive the round trip");

    // A shape that disagrees with the checkpoint must fail loudly.
    check(!store.require("test.weight", {4U}, litemind::DataType::BFloat16, error).valid(),
          "a wrong shape is rejected");
    check(error.find("shape") != std::string::npos, "the shape error names the problem: " + error);
    check(!store.require("test.weight", {3U}, litemind::DataType::Float32, error).valid(),
          "a wrong element type is rejected");
    check(!store.require("absent.weight", {1U}, litemind::DataType::BFloat16, error).valid(),
          "a missing tensor is rejected");

    // A truncated header must be refused rather than read past the end.
    const std::filesystem::path broken = directory / "broken.safetensors";
    check(write_shard(broken, R"({"a":{"dtype":"BF16","shape":[9999],"data_offsets":[0,19998]}})", {}),
          "the truncated shard is written");
    litemind::SafeTensor damaged;
    check(!damaged.open(broken, error), "a tensor extending past the end is rejected");

    std::filesystem::remove_all(directory, filesystem_error);
    return report("SafeTensorTest");
}
