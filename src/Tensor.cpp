#include "Tensor.hpp"

namespace litemind {

Tensor::Tensor(std::string name, std::vector<std::size_t> shape, const DataType data_type,
               const std::uint64_t offset) noexcept
    : name_(std::move(name)),
      shape_(std::move(shape)),
      data_type_(data_type),
      offset_(offset) {
    std::uint64_t element_count = 1U;
    for (const std::size_t dimension : shape_) {
        element_count *= dimension;
    }

    switch (data_type_) {
        case DataType::BFloat16:
        case DataType::Float16: byte_size_ = element_count * 2U; break;
        case DataType::Float32: byte_size_ = element_count * 4U; break;
        case DataType::Float64: byte_size_ = element_count * 8U; break;
        case DataType::Int8:
        case DataType::UInt8:
        case DataType::Bool: byte_size_ = element_count; break;
        case DataType::Int16: byte_size_ = element_count * 2U; break;
        case DataType::Int32: byte_size_ = element_count * 4U; break;
        case DataType::Int64: byte_size_ = element_count * 8U; break;
        case DataType::Unknown: byte_size_ = 0U; break;
    }
}

const std::string& Tensor::name() const noexcept { return name_; }
const std::vector<std::size_t>& Tensor::shape() const noexcept { return shape_; }
DataType Tensor::data_type() const noexcept { return data_type_; }
std::uint64_t Tensor::offset() const noexcept { return offset_; }
std::uint64_t Tensor::byte_size() const noexcept { return byte_size_; }

}  // namespace litemind
