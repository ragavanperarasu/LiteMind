#include "Tensor.hpp"

#include <sstream>

namespace litemind {

std::size_t byte_width(const DataType data_type) noexcept {
    switch (data_type) {
        case DataType::BFloat16:
        case DataType::Float16:
        case DataType::Int16: return 2U;
        case DataType::Float32:
        case DataType::Int32: return 4U;
        case DataType::Float64:
        case DataType::Int64: return 8U;
        case DataType::Int8:
        case DataType::UInt8:
        case DataType::Bool: return 1U;
        case DataType::Unknown: break;
    }
    return 0U;
}

std::string_view type_name(const DataType data_type) noexcept {
    switch (data_type) {
        case DataType::BFloat16: return "BF16";
        case DataType::Float16: return "F16";
        case DataType::Float32: return "F32";
        case DataType::Float64: return "F64";
        case DataType::Int8: return "I8";
        case DataType::Int16: return "I16";
        case DataType::Int32: return "I32";
        case DataType::Int64: return "I64";
        case DataType::UInt8: return "U8";
        case DataType::Bool: return "BOOL";
        case DataType::Unknown: break;
    }
    return "UNKNOWN";
}

Tensor::Tensor(std::string name, std::vector<std::size_t> shape, const DataType data_type,
               const std::uint64_t offset) noexcept
    : name_(std::move(name)), shape_(std::move(shape)), data_type_(data_type), offset_(offset) {
    element_count_ = 1U;
    for (const std::size_t dimension : shape_) {
        element_count_ *= static_cast<std::uint64_t>(dimension);
    }
    byte_size_ = element_count_ * static_cast<std::uint64_t>(byte_width(data_type_));
}

std::string Tensor::shape_text() const {
    std::ostringstream text;
    text << '[';
    for (std::size_t index = 0; index < shape_.size(); ++index) {
        text << (index == 0U ? "" : ", ") << shape_[index];
    }
    text << ']';
    return text.str();
}

}  // namespace litemind
