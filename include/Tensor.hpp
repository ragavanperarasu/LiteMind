#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace litemind {

/** Data types understood by tensor metadata. No tensor memory is owned here. */
enum class DataType {
    Unknown,
    BFloat16,
    Float16,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    Bool
};

/**
 * @brief Describes a tensor's location and layout without loading its data.
 */
class Tensor final {
public:
    Tensor(std::string name, std::vector<std::size_t> shape, DataType data_type,
           std::uint64_t offset) noexcept;

    [[nodiscard]] const std::string& name() const noexcept;
    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept;
    [[nodiscard]] DataType data_type() const noexcept;
    [[nodiscard]] std::uint64_t offset() const noexcept;
    [[nodiscard]] std::uint64_t byte_size() const noexcept;

private:
    std::string name_;
    std::vector<std::size_t> shape_;
    DataType data_type_;
    std::uint64_t offset_;
    std::uint64_t byte_size_{};
};

}  // namespace litemind
