#include "SafeTensor.hpp"

#include "Json.hpp"

#include <cstring>
#include <limits>
#include <string_view>

namespace litemind {
namespace {

/** A header larger than this is treated as a corrupt or non-SafeTensors file. */
constexpr std::uint64_t maximum_header_size = 512ULL * 1024ULL * 1024ULL;

[[nodiscard]] DataType parse_data_type(const std::string_view text) noexcept {
    if (text == "BF16") return DataType::BFloat16;
    if (text == "F16") return DataType::Float16;
    if (text == "F32") return DataType::Float32;
    if (text == "F64") return DataType::Float64;
    if (text == "I8") return DataType::Int8;
    if (text == "I16") return DataType::Int16;
    if (text == "I32") return DataType::Int32;
    if (text == "I64") return DataType::Int64;
    if (text == "U8") return DataType::UInt8;
    if (text == "BOOL") return DataType::Bool;
    return DataType::Unknown;
}

/** Translates one header entry into tensor metadata with a file-absolute offset. */
[[nodiscard]] bool read_entry(const std::string& name, const Json& entry,
                              const std::uint64_t payload_start, const std::uint64_t file_size,
                              Tensor& tensor, std::string& error) {
    if (!entry.is_object()) {
        error = "tensor entry '" + name + "' is not a JSON object.";
        return false;
    }

    const DataType data_type = parse_data_type(entry.string_or("dtype", ""));
    if (data_type == DataType::Unknown) {
        error = "tensor '" + name + "' uses an unsupported dtype '" + entry.string_or("dtype", "?") + "'.";
        return false;
    }

    const Json* shape_node = entry.find("shape");
    if (shape_node == nullptr || !shape_node->is_array()) {
        error = "tensor '" + name + "' has no shape array.";
        return false;
    }
    std::vector<std::size_t> shape;
    shape.reserve(shape_node->elements().size());
    for (const Json& dimension : shape_node->elements()) {
        if (!dimension.is_number() || dimension.number_value() < 0.0) {
            error = "tensor '" + name + "' has a non-numeric dimension.";
            return false;
        }
        shape.push_back(static_cast<std::size_t>(dimension.number_value()));
    }

    const Json* offsets = entry.find("data_offsets");
    if (offsets == nullptr || !offsets->is_array() || offsets->elements().size() != 2U
        || !offsets->elements()[0].is_number() || !offsets->elements()[1].is_number()) {
        error = "tensor '" + name + "' has a malformed data_offsets pair.";
        return false;
    }
    const auto begin = static_cast<std::uint64_t>(offsets->elements()[0].number_value());
    const auto end = static_cast<std::uint64_t>(offsets->elements()[1].number_value());
    if (end < begin) {
        error = "tensor '" + name + "' has data_offsets in descending order.";
        return false;
    }

    tensor = Tensor(name, std::move(shape), data_type, payload_start + begin);
    if (tensor.byte_size() != end - begin) {
        error = "tensor '" + name + "' declares " + std::to_string(end - begin)
              + " payload bytes but its shape and dtype need " + std::to_string(tensor.byte_size()) + ".";
        return false;
    }
    if (tensor.offset() > file_size || tensor.byte_size() > file_size - tensor.offset()) {
        error = "tensor '" + name + "' extends past the end of the shard.";
        return false;
    }
    return true;
}

}  // namespace

bool SafeTensor::open(const std::filesystem::path& path, std::string& error) {
    tensors_.clear();
    index_.clear();

    if (!file_.open(path, error)) {
        return false;
    }
    if (file_.size() < 8U) {
        error = path.string() + " is too small to be a SafeTensors shard.";
        file_.close();
        return false;
    }

    std::uint64_t header_size{};
    const std::byte* const length_bytes = file_.data();
    for (std::size_t index = 0; index < 8U; ++index) {
        header_size |= static_cast<std::uint64_t>(std::to_integer<unsigned char>(length_bytes[index]))
                    << (index * 8U);
    }
    if (header_size == 0U || header_size > maximum_header_size || header_size > file_.size() - 8U) {
        error = path.string() + " declares an implausible SafeTensors header length of "
              + std::to_string(header_size) + " bytes.";
        file_.close();
        return false;
    }

    const std::uint64_t payload_start = 8U + header_size;
    const std::string_view header(reinterpret_cast<const char*>(file_.data() + 8U),
                                  static_cast<std::size_t>(header_size));

    Json document;
    if (!Json::parse(header, document, error)) {
        error = path.string() + ": " + error;
        file_.close();
        return false;
    }
    if (!document.is_object()) {
        error = path.string() + ": the SafeTensors header must be a JSON object.";
        file_.close();
        return false;
    }

    tensors_.reserve(document.members().size());
    for (const auto& [name, entry] : document.members()) {
        if (name == "__metadata__") {
            continue;
        }
        Tensor tensor;
        if (!read_entry(name, entry, payload_start, file_.size(), tensor, error)) {
            error = path.string() + ": " + error;
            tensors_.clear();
            index_.clear();
            file_.close();
            return false;
        }
        index_.emplace(name, tensors_.size());
        tensors_.push_back(std::move(tensor));
    }

    if (tensors_.empty()) {
        error = path.string() + " contains no tensors.";
        file_.close();
        return false;
    }
    return true;
}

const Tensor* SafeTensor::find(const std::string& name) const {
    const auto entry = index_.find(name);
    return entry == index_.end() ? nullptr : &tensors_[entry->second];
}

const std::byte* SafeTensor::payload(const Tensor& tensor) const noexcept {
    return file_.view(tensor.offset(), tensor.byte_size());
}

void SafeTensor::prefetch(const Tensor& tensor) const noexcept {
    file_.prefetch(tensor.offset(), tensor.byte_size());
}

void SafeTensor::release(const Tensor& tensor) const noexcept {
    file_.release(tensor.offset(), tensor.byte_size());
}

}  // namespace litemind
