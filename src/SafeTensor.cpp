#include "SafeTensor.hpp"

#include <array>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <limits>
#include <optional>
#include <string_view>

namespace litemind {
namespace {

constexpr std::uint64_t maximum_header_size = 128U * 1024U * 1024U;

class HeaderReader final {
public:
    HeaderReader(const std::string_view source, const std::uint64_t data_start)
        : source_(source), data_start_(data_start) {}

    [[nodiscard]] bool parse(std::vector<Tensor>& tensors, std::string& error) {
        skip_whitespace();
        if (!consume('{')) {
            error = "SafeTensors header must begin with a JSON object.";
            return false;
        }

        skip_whitespace();
        while (!consume('}')) {
            const auto name = read_string();
            if (!name || !consume(':')) {
                error = "SafeTensors header has an invalid object entry.";
                return false;
            }

            if (*name == "__metadata__") {
                if (!skip_value()) {
                    error = "SafeTensors metadata value is malformed.";
                    return false;
                }
            } else {
                Tensor tensor("", {}, DataType::Unknown, 0U);
                if (!read_tensor(*name, tensor)) {
                    error = "SafeTensors tensor entry is malformed: " + *name;
                    return false;
                }
                tensors.push_back(std::move(tensor));
            }

            skip_whitespace();
            if (consume('}')) {
                break;
            }
            if (!consume(',')) {
                error = "SafeTensors header entries must be comma separated.";
                return false;
            }
            skip_whitespace();
        }

        skip_whitespace();
        if (position_ != source_.size()) {
            error = "SafeTensors header contains trailing data.";
            return false;
        }
        return true;
    }

private:
    [[nodiscard]] bool read_tensor(const std::string& name, Tensor& tensor) {
        if (!consume('{')) {
            return false;
        }

        std::optional<DataType> data_type;
        std::optional<std::vector<std::size_t>> shape;
        std::optional<std::array<std::uint64_t, 2U>> offsets;
        skip_whitespace();

        while (!consume('}')) {
            const auto key = read_string();
            if (!key || !consume(':')) {
                return false;
            }

            if (*key == "dtype") {
                const auto text = read_string();
                if (!text) {
                    return false;
                }
                data_type = parse_data_type(*text);
                if (*data_type == DataType::Unknown) {
                    return false;
                }
            } else if (*key == "shape") {
                shape = read_shape();
                if (!shape) {
                    return false;
                }
            } else if (*key == "data_offsets") {
                offsets = read_offsets();
                if (!offsets) {
                    return false;
                }
            } else if (!skip_value()) {
                return false;
            }

            skip_whitespace();
            if (consume('}')) {
                break;
            }
            if (!consume(',')) {
                return false;
            }
            skip_whitespace();
        }

        if (!data_type || !shape || !offsets || (*offsets)[1] < (*offsets)[0]) {
            return false;
        }
        tensor = Tensor(name, std::move(*shape), *data_type, data_start_ + (*offsets)[0]);
        return tensor.byte_size() == (*offsets)[1] - (*offsets)[0];
    }

    [[nodiscard]] std::optional<std::vector<std::size_t>> read_shape() {
        if (!consume('[')) {
            return std::nullopt;
        }

        std::vector<std::size_t> shape;
        skip_whitespace();
        if (consume(']')) {
            return shape;
        }
        while (true) {
            const auto value = read_unsigned();
            if (!value || *value > std::numeric_limits<std::size_t>::max()) {
                return std::nullopt;
            }
            shape.push_back(static_cast<std::size_t>(*value));
            skip_whitespace();
            if (consume(']')) {
                return shape;
            }
            if (!consume(',')) {
                return std::nullopt;
            }
        }
    }

    [[nodiscard]] std::optional<std::array<std::uint64_t, 2U>> read_offsets() {
        if (!consume('[')) {
            return std::nullopt;
        }
        const auto begin = read_unsigned();
        if (!begin || !consume(',')) {
            return std::nullopt;
        }
        const auto end = read_unsigned();
        if (!end || !consume(']')) {
            return std::nullopt;
        }
        return std::array<std::uint64_t, 2U>{*begin, *end};
    }

    [[nodiscard]] std::optional<std::uint64_t> read_unsigned() {
        skip_whitespace();
        const std::size_t begin = position_;
        while (position_ < source_.size() && std::isdigit(static_cast<unsigned char>(source_[position_])) != 0) {
            ++position_;
        }
        if (begin == position_) {
            return std::nullopt;
        }
        try {
            return std::stoull(std::string(source_.substr(begin, position_ - begin)));
        } catch (const std::exception&) {
            return std::nullopt;
        }
    }

    [[nodiscard]] std::optional<std::string> read_string() {
        skip_whitespace();
        if (position_ == source_.size() || source_[position_] != '"') {
            return std::nullopt;
        }
        ++position_;

        std::string value;
        while (position_ < source_.size()) {
            const char character = source_[position_++];
            if (character == '"') {
                return value;
            }
            if (character != '\\') {
                value += character;
                continue;
            }
            if (position_ == source_.size()) {
                return std::nullopt;
            }
            const char escaped = source_[position_++];
            switch (escaped) {
                case '"': value += '"'; break;
                case '\\': value += '\\'; break;
                case '/': value += '/'; break;
                case 'b': value += '\b'; break;
                case 'f': value += '\f'; break;
                case 'n': value += '\n'; break;
                case 'r': value += '\r'; break;
                case 't': value += '\t'; break;
                default: return std::nullopt;
            }
        }
        return std::nullopt;
    }

    [[nodiscard]] bool skip_value() {
        skip_whitespace();
        if (position_ == source_.size()) {
            return false;
        }
        if (source_[position_] == '"') {
            return read_string().has_value();
        }
        if (source_[position_] == '{') {
            ++position_;
            skip_whitespace();
            if (consume('}')) {
                return true;
            }
            while (true) {
                if (!read_string() || !consume(':') || !skip_value()) {
                    return false;
                }
                skip_whitespace();
                if (consume('}')) {
                    return true;
                }
                if (!consume(',')) {
                    return false;
                }
            }
        }
        if (source_[position_] == '[') {
            ++position_;
            skip_whitespace();
            if (consume(']')) {
                return true;
            }
            while (true) {
                if (!skip_value()) {
                    return false;
                }
                skip_whitespace();
                if (consume(']')) {
                    return true;
                }
                if (!consume(',')) {
                    return false;
                }
            }
        }
        while (position_ < source_.size()) {
            const char character = source_[position_];
            if (character == ',' || character == '}' || character == ']'
                || std::isspace(static_cast<unsigned char>(character)) != 0) {
                return true;
            }
            ++position_;
        }
        return position_ > 0U;
    }

    void skip_whitespace() {
        while (position_ < source_.size()
               && std::isspace(static_cast<unsigned char>(source_[position_])) != 0) {
            ++position_;
        }
    }

    [[nodiscard]] bool consume(const char expected) {
        skip_whitespace();
        if (position_ < source_.size() && source_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    [[nodiscard]] static DataType parse_data_type(const std::string_view text) noexcept {
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

    std::string_view source_;
    std::uint64_t data_start_{};
    std::size_t position_{};
};

}  // namespace

bool SafeTensor::open(const std::filesystem::path& path, std::string& error) {
    path_.clear();
    tensors_.clear();

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        error = "Unable to open SafeTensors shard: " + path.string();
        return false;
    }

    std::array<unsigned char, 8U> length_bytes{};
    file.read(reinterpret_cast<char*>(length_bytes.data()), static_cast<std::streamsize>(length_bytes.size()));
    if (file.gcount() != static_cast<std::streamsize>(length_bytes.size())) {
        error = "SafeTensors shard does not contain a complete header length.";
        return false;
    }

    std::uint64_t header_size{};
    for (std::size_t index = 0; index < length_bytes.size(); ++index) {
        header_size |= static_cast<std::uint64_t>(length_bytes[index]) << (index * 8U);
    }
    if (header_size == 0U || header_size > maximum_header_size) {
        error = "SafeTensors header size is invalid.";
        return false;
    }

    std::string header(header_size, '\0');
    file.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (file.gcount() != static_cast<std::streamsize>(header.size())) {
        error = "SafeTensors shard ended before its header was complete.";
        return false;
    }

    HeaderReader reader(header, header_size + length_bytes.size());
    if (!reader.parse(tensors_, error)) {
        tensors_.clear();
        return false;
    }
    path_ = path;
    return true;
}

const std::filesystem::path& SafeTensor::path() const noexcept { return path_; }
const std::vector<Tensor>& SafeTensor::tensors() const noexcept { return tensors_; }

}  // namespace litemind
