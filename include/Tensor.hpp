#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace litemind {

/** Element types a SafeTensors header can declare. No tensor memory is owned here. */
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

/** Returns the width of one element in bytes, or zero for DataType::Unknown. */
[[nodiscard]] std::size_t byte_width(DataType data_type) noexcept;

/** Returns the SafeTensors spelling of a data type, for diagnostics. */
[[nodiscard]] std::string_view type_name(DataType data_type) noexcept;

/**
 * @brief Describes where one tensor lives inside a SafeTensors shard.
 *
 * A Tensor is pure metadata: a name, a shape, an element type, and a byte range
 * measured from the start of the file. Weights are read through the shard's
 * memory mapping, so nothing here allocates or copies payload bytes.
 */
class Tensor final {
public:
    Tensor() = default;
    Tensor(std::string name, std::vector<std::size_t> shape, DataType data_type,
           std::uint64_t offset) noexcept;

    [[nodiscard]] const std::string& name() const noexcept { return name_; }
    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] DataType data_type() const noexcept { return data_type_; }
    [[nodiscard]] std::uint64_t offset() const noexcept { return offset_; }
    [[nodiscard]] std::uint64_t byte_size() const noexcept { return byte_size_; }
    [[nodiscard]] std::uint64_t element_count() const noexcept { return element_count_; }

    /** Renders the shape as "[rows, columns]" for error messages. */
    [[nodiscard]] std::string shape_text() const;

    /** True when the shape matches expected exactly. */
    [[nodiscard]] bool has_shape(const std::vector<std::size_t>& expected) const noexcept {
        return shape_ == expected;
    }

private:
    std::string name_;
    std::vector<std::size_t> shape_;
    DataType data_type_{DataType::Unknown};
    std::uint64_t offset_{};
    std::uint64_t byte_size_{};
    std::uint64_t element_count_{};
};

}  // namespace litemind
