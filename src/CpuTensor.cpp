#include "CpuTensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace litemind {

CpuTensor::CpuTensor(std::vector<std::size_t> shape)
    : shape_(std::move(shape)), values_(checked_element_count(shape_), 0.0F) {}

CpuTensor::CpuTensor(std::vector<std::size_t> shape, std::vector<float> values)
    : shape_(std::move(shape)), values_(std::move(values)) {
    if (values_.size() != checked_element_count(shape_)) {
        throw std::invalid_argument("CPU tensor data length does not match its shape.");
    }
}

const std::vector<std::size_t>& CpuTensor::shape() const noexcept { return shape_; }
std::size_t CpuTensor::element_count() const noexcept { return values_.size(); }
std::span<const float> CpuTensor::values() const noexcept { return values_; }
std::span<float> CpuTensor::values() noexcept { return values_; }

void CpuTensor::add_inplace(const CpuTensor& other) {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Residual addition requires tensors with identical shapes.");
    }
    for (std::size_t index = 0; index < values_.size(); ++index) {
        values_[index] += other.values_[index];
    }
}

void CpuTensor::silu_inplace() noexcept {
    for (float& value : values_) {
        value /= 1.0F + std::exp(-value);
    }
}

void CpuTensor::rms_norm_inplace(const std::span<const float> weight, const float epsilon) {
    if (shape_.empty() || weight.size() != shape_.back() || epsilon <= 0.0F) {
        throw std::invalid_argument("RMSNorm requires a positive epsilon and a final-dimension weight.");
    }
    const std::size_t width = shape_.back();
    for (std::size_t offset = 0; offset < values_.size(); offset += width) {
        float squared_sum{};
        for (std::size_t column = 0; column < width; ++column) {
            squared_sum += values_[offset + column] * values_[offset + column];
        }
        const float inverse_rms = 1.0F / std::sqrt(squared_sum / static_cast<float>(width) + epsilon);
        for (std::size_t column = 0; column < width; ++column) {
            values_[offset + column] *= inverse_rms * weight[column];
        }
    }
}

void CpuTensor::softmax_last_dimension_inplace() {
    if (shape_.empty()) {
        throw std::invalid_argument("Softmax requires at least one tensor dimension.");
    }
    const std::size_t width = shape_.back();
    if (width == 0U) {
        throw std::invalid_argument("Softmax final dimension cannot be zero.");
    }
    for (std::size_t offset = 0; offset < values_.size(); offset += width) {
        const auto first = values_.begin() + static_cast<std::ptrdiff_t>(offset);
        const auto last = first + static_cast<std::ptrdiff_t>(width);
        const float maximum = *std::max_element(first, last);
        float sum{};
        for (auto iterator = first; iterator != last; ++iterator) {
            *iterator = std::exp(*iterator - maximum);
            sum += *iterator;
        }
        for (auto iterator = first; iterator != last; ++iterator) {
            *iterator /= sum;
        }
    }
}

CpuTensor CpuTensor::matmul(const CpuTensor& left, const CpuTensor& right) {
    if (left.shape_.size() != 2U || right.shape_.size() != 2U || left.shape_[1] != right.shape_[0]) {
        throw std::invalid_argument("Matrix multiplication requires [M, K] and [K, N] tensors.");
    }
    const std::size_t rows = left.shape_[0];
    const std::size_t shared = left.shape_[1];
    const std::size_t columns = right.shape_[1];
    CpuTensor result({rows, columns});

    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t inner = 0; inner < shared; ++inner) {
            const float left_value = left.values_[row * shared + inner];
            const float* right_row = right.values_.data() + inner * columns;
            float* output_row = result.values_.data() + row * columns;
            for (std::size_t column = 0; column < columns; ++column) {
                output_row[column] += left_value * right_row[column];
            }
        }
    }
    return result;
}

std::size_t CpuTensor::argmax(const std::span<const float> values) {
    if (values.empty()) {
        throw std::invalid_argument("Argmax requires at least one value.");
    }
    return static_cast<std::size_t>(std::distance(values.begin(), std::max_element(values.begin(), values.end())));
}

std::size_t CpuTensor::checked_element_count(const std::span<const std::size_t> shape) {
    if (shape.empty()) {
        throw std::invalid_argument("Scalar CPU tensors are not supported by this runtime.");
    }
    std::size_t count = 1U;
    for (const std::size_t dimension : shape) {
        if (dimension == 0U || count > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::invalid_argument("CPU tensor shape is invalid or exceeds addressable memory.");
        }
        count *= dimension;
    }
    return count;
}

}  // namespace litemind
