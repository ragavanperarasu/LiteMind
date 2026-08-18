#include "WeightStore.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <system_error>

namespace litemind {

float bfloat16_to_float(const std::uint16_t bits) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(bits) << 16U);
}

float float16_to_float(const std::uint16_t bits) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(bits & 0x8000U) << 16U;
    const std::uint32_t exponent = (bits >> 10U) & 0x1fU;
    const std::uint32_t mantissa = bits & 0x3ffU;

    if (exponent == 0U) {
        if (mantissa == 0U) {
            return std::bit_cast<float>(sign);
        }
        // Subnormal half: renormalise into a float32 normal.
        std::uint32_t shifted_exponent = 127U - 15U + 1U;
        std::uint32_t shifted_mantissa = mantissa;
        while ((shifted_mantissa & 0x400U) == 0U) {
            shifted_mantissa <<= 1U;
            --shifted_exponent;
        }
        shifted_mantissa &= 0x3ffU;
        return std::bit_cast<float>(sign | (shifted_exponent << 23U) | (shifted_mantissa << 13U));
    }
    if (exponent == 0x1fU) {
        return std::bit_cast<float>(sign | 0x7f800000U | (mantissa << 13U));
    }
    return std::bit_cast<float>(sign | ((exponent + 127U - 15U) << 23U) | (mantissa << 13U));
}

void WeightView::prefetch() const noexcept {
    if (shard != nullptr && meta != nullptr) {
        shard->prefetch(*meta);
    }
}

void WeightView::release() const noexcept {
    if (shard != nullptr && meta != nullptr) {
        shard->release(*meta);
    }
}

bool WeightStore::open(const std::filesystem::path& model_directory, std::string& error) {
    shards_.clear();
    index_.clear();
    statistics_ = StoreStatistics{};

    std::error_code filesystem_error;
    if (!std::filesystem::is_directory(model_directory, filesystem_error)) {
        error = "Not a model directory: " + model_directory.string();
        return false;
    }

    // Sorting keeps shard order, and therefore diagnostics, reproducible.
    std::vector<std::filesystem::path> shard_paths;
    for (const auto& entry : std::filesystem::directory_iterator(model_directory, filesystem_error)) {
        if (filesystem_error) {
            break;
        }
        if (entry.is_regular_file(filesystem_error) && entry.path().extension() == ".safetensors") {
            shard_paths.push_back(entry.path());
        }
    }
    std::sort(shard_paths.begin(), shard_paths.end());

    if (shard_paths.empty()) {
        error = "No .safetensors files were found in " + model_directory.string()
              + ". Download the model weights into this directory first "
                "(scripts/download_model.ps1 does this on Windows).";
        return false;
    }

    for (const std::filesystem::path& path : shard_paths) {
        auto shard = std::make_unique<SafeTensor>();
        if (!shard->open(path, error)) {
            shards_.clear();
            index_.clear();
            return false;
        }

        const std::size_t shard_index = shards_.size();
        for (std::size_t tensor_index = 0; tensor_index < shard->tensors().size(); ++tensor_index) {
            const Tensor& tensor = shard->tensors()[tensor_index];
            // A duplicate name across shards means a broken download; keep the first.
            index_.emplace(tensor.name(), std::pair{shard_index, tensor_index});
        }
        statistics_.tensors += shard->tensors().size();
        statistics_.mapped_bytes += shard->file().size();
        shards_.push_back(std::move(shard));
    }

    statistics_.shards = shards_.size();
    return true;
}

WeightView WeightStore::find(const std::string& name) const {
    const auto entry = index_.find(name);
    if (entry == index_.end()) {
        return WeightView{};
    }
    const SafeTensor& shard = *shards_[entry->second.first];
    const Tensor& tensor = shard.tensors()[entry->second.second];
    return WeightView{&shard, &tensor, shard.payload(tensor)};
}

bool WeightStore::contains(const std::string& name) const {
    return index_.find(name) != index_.end();
}

WeightView WeightStore::require(const std::string& name,
                                const std::vector<std::size_t>& expected_shape,
                                const DataType expected_type, std::string& error) const {
    const WeightView view = find(name);
    if (!view.valid()) {
        error = "The checkpoint has no tensor named '" + name + "'.";
        return WeightView{};
    }

    Tensor expected(name, expected_shape, expected_type, 0U);
    if (!view.meta->has_shape(expected_shape)) {
        error = "Tensor '" + name + "' has shape " + view.meta->shape_text() + " but config.json implies "
              + expected.shape_text() + ".";
        return WeightView{};
    }
    if (view.meta->data_type() != expected_type) {
        error = "Tensor '" + name + "' is stored as " + std::string(type_name(view.meta->data_type()))
              + " but this build reads " + std::string(type_name(expected_type)) + " here.";
        return WeightView{};
    }
    return view;
}

bool WeightStore::read_float32(const std::string& name,
                               const std::vector<std::size_t>& expected_shape,
                               std::vector<float>& values, std::string& error) const {
    const WeightView view = find(name);
    if (!view.valid()) {
        error = "The checkpoint has no tensor named '" + name + "'.";
        return false;
    }
    if (!view.meta->has_shape(expected_shape)) {
        Tensor expected(name, expected_shape, view.meta->data_type(), 0U);
        error = "Tensor '" + name + "' has shape " + view.meta->shape_text() + " but config.json implies "
              + expected.shape_text() + ".";
        return false;
    }

    const auto count = static_cast<std::size_t>(view.element_count());
    values.resize(count);

    switch (view.meta->data_type()) {
        case DataType::Float32:
            std::memcpy(values.data(), view.bytes, count * sizeof(float));
            break;
        case DataType::BFloat16: {
            const std::uint16_t* source = view.as_bf16();
            for (std::size_t index = 0; index < count; ++index) {
                values[index] = bfloat16_to_float(source[index]);
            }
            break;
        }
        case DataType::Float16: {
            const std::uint16_t* source = view.as_bf16();
            for (std::size_t index = 0; index < count; ++index) {
                values[index] = float16_to_float(source[index]);
            }
            break;
        }
        default:
            error = "Tensor '" + name + "' is stored as " + std::string(type_name(view.meta->data_type()))
                  + ", which cannot be widened to float32.";
            values.clear();
            return false;
    }
    return true;
}

std::vector<std::string> WeightStore::names() const {
    std::vector<std::string> result;
    result.reserve(index_.size());
    for (const auto& [name, location] : index_) {
        result.push_back(name);
    }
    std::sort(result.begin(), result.end());
    return result;
}

}  // namespace litemind
