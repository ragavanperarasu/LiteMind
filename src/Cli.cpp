#include "Cli.hpp"

#include "DeepSeekRunner.hpp"
#include "Logger.hpp"
#include "SafeTensor.hpp"
#include "Tokenizer.hpp"
#include "WeightReader.hpp"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>

namespace litemind {
namespace {

struct ModelStatus {
    std::string architecture{"unknown"};
    std::string data_type{"unknown"};
    std::uint64_t logical_size_bytes{};
    std::uint64_t on_disk_size_bytes{};
    std::uint64_t estimated_parameter_count{};
    std::uint64_t tensor_count{};
    std::uint64_t metadata_shard_count{};
    bool metadata_complete{true};
    std::uint64_t routed_expert_count{};
    std::uint64_t shared_expert_count{};
    std::uint64_t experts_per_token{};
    std::uint64_t layer_count{};
    std::uint64_t dense_layer_count{};
};

[[nodiscard]] std::optional<std::string> read_text_file(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return std::nullopt;
    }

    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

// This deliberately extracts only known manifest fields; it is not a JSON parser.
[[nodiscard]] std::optional<std::string> string_field(const std::string_view text,
                                                        const std::string_view key) {
    const std::string marker = '"' + std::string(key) + '"';
    const std::size_t key_position = text.find(marker);
    if (key_position == std::string_view::npos) {
        return std::nullopt;
    }

    const std::size_t colon = text.find(':', key_position + marker.size());
    const std::size_t first_quote = text.find('"', colon + 1U);
    if (colon == std::string_view::npos || first_quote == std::string_view::npos) {
        return std::nullopt;
    }

    const std::size_t second_quote = text.find('"', first_quote + 1U);
    if (second_quote == std::string_view::npos) {
        return std::nullopt;
    }
    return std::string(text.substr(first_quote + 1U, second_quote - first_quote - 1U));
}

[[nodiscard]] std::optional<std::uint64_t> unsigned_field(const std::string_view text,
                                                            const std::string_view key) {
    const std::string marker = '"' + std::string(key) + '"';
    const std::size_t key_position = text.find(marker);
    if (key_position == std::string_view::npos) {
        return std::nullopt;
    }

    const std::size_t colon = text.find(':', key_position + marker.size());
    if (colon == std::string_view::npos) {
        return std::nullopt;
    }

    std::size_t start = colon + 1U;
    while (start < text.size() && std::isspace(static_cast<unsigned char>(text[start])) != 0) {
        ++start;
    }
    std::size_t end = start;
    while (end < text.size() && std::isdigit(static_cast<unsigned char>(text[end])) != 0) {
        ++end;
    }
    if (start == end) {
        return std::nullopt;
    }

    try {
        return std::stoull(std::string(text.substr(start, end - start)));
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

[[nodiscard]] std::uint64_t bytes_per_parameter(const std::string_view data_type) noexcept {
    if (data_type == "bfloat16" || data_type == "float16") {
        return 2U;
    }
    if (data_type == "float32") {
        return 4U;
    }
    if (data_type == "float64") {
        return 8U;
    }
    return 0U;
}

[[nodiscard]] std::uint64_t bytes_per_element(const DataType data_type) noexcept {
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
        case DataType::Unknown: return 0U;
    }
    return 0U;
}

[[nodiscard]] double gibibytes(const std::uint64_t bytes) noexcept {
    constexpr double bytes_per_gibibyte = 1024.0 * 1024.0 * 1024.0;
    return static_cast<double>(bytes) / bytes_per_gibibyte;
}

[[nodiscard]] ModelStatus inspect_model(const std::filesystem::path& model_directory) {
    ModelStatus status;
    std::error_code error;

    for (const auto& entry : std::filesystem::directory_iterator(model_directory, error)) {
        if (error || !entry.is_regular_file(error) || entry.path().extension() != ".safetensors") {
            continue;
        }
        status.on_disk_size_bytes += entry.file_size(error);

        SafeTensor shard;
        std::string parse_error;
        if (!shard.open(entry.path(), parse_error)) {
            status.metadata_complete = false;
            continue;
        }
        ++status.metadata_shard_count;
        status.tensor_count += shard.tensors().size();
        for (const Tensor& tensor : shard.tensors()) {
            const std::uint64_t element_size = bytes_per_element(tensor.data_type());
            if (element_size != 0U) {
                status.estimated_parameter_count += tensor.byte_size() / element_size;
            }
        }
    }

    if (const auto config = read_text_file(model_directory / "config.json")) {
        status.architecture = string_field(*config, "model_type").value_or(status.architecture);
        status.data_type = string_field(*config, "torch_dtype").value_or(status.data_type);
        status.routed_expert_count = unsigned_field(*config, "n_routed_experts").value_or(0U);
        status.shared_expert_count = unsigned_field(*config, "n_shared_experts").value_or(0U);
        status.experts_per_token = unsigned_field(*config, "num_experts_per_tok").value_or(0U);
        status.layer_count = unsigned_field(*config, "num_hidden_layers").value_or(0U);
        status.dense_layer_count = unsigned_field(*config, "first_k_dense_replace").value_or(0U);
    }

    if (const auto index = read_text_file(model_directory / "model.safetensors.index.json")) {
        status.logical_size_bytes = unsigned_field(*index, "total_size").value_or(0U);
    }

    const std::uint64_t parameter_bytes = bytes_per_parameter(status.data_type);
    if (status.estimated_parameter_count == 0U && parameter_bytes != 0U) {
        status.estimated_parameter_count = status.logical_size_bytes / parameter_bytes;
    }
    return status;
}

[[nodiscard]] std::optional<std::string> verify_weight_reader(const std::filesystem::path& model_directory) {
    constexpr std::uint64_t maximum_probe_size = 64U * 1024U;
    std::error_code error;
    for (const auto& entry : std::filesystem::directory_iterator(model_directory, error)) {
        if (error || !entry.is_regular_file(error) || entry.path().extension() != ".safetensors") {
            continue;
        }

        SafeTensor shard;
        std::string parse_error;
        if (!shard.open(entry.path(), parse_error)) {
            continue;
        }
        const auto tensor = std::find_if(shard.tensors().begin(), shard.tensors().end(), [](const Tensor& candidate) {
            return candidate.data_type() == DataType::BFloat16 && candidate.byte_size() > 0U
                && candidate.byte_size() <= maximum_probe_size;
        });
        if (tensor == shard.tensors().end()) {
            continue;
        }

        WeightReader reader;
        std::vector<float> values;
        std::string read_error;
        if (reader.read_bfloat16(shard, *tensor, values, read_error)) {
            return tensor->name() + " (" + std::to_string(values.size()) + " BF16 values)";
        }
    }
    return std::nullopt;
}

void print_status(const ModelStatus& status, const std::filesystem::path& model_directory) {
    const std::uint64_t moe_layers = status.layer_count > status.dense_layer_count
        ? status.layer_count - status.dense_layer_count
        : 0U;
    std::cout << "\nModel status\n"
              << "  Directory: " << model_directory.string() << '\n'
              << "  Architecture: " << status.architecture << '\n'
              << "  Weight files on disk: " << std::fixed << std::setprecision(2)
              << gibibytes(status.on_disk_size_bytes) << " GiB\n"
              << "  Logical tensor data: " << gibibytes(status.logical_size_bytes) << " GiB\n";

    if (status.estimated_parameter_count != 0U) {
        std::cout << "  Estimated parameters: " << std::setprecision(2)
                  << static_cast<double>(status.estimated_parameter_count) / 1'000'000'000.0
                  << " billion\n";
    }
    std::cout << "  Tensor records: " << status.tensor_count << " from "
              << status.metadata_shard_count << " parsed SafeTensors shards"
              << (status.metadata_complete ? "\n" : " (one or more headers could not be read)\n");

    std::cout << "  Layers: " << status.layer_count << " (" << status.dense_layer_count
              << " dense, " << moe_layers << " MoE)\n"
              << "  Routed experts: " << status.routed_expert_count << '\n'
              << "  Selected routed experts per MoE layer/token: " << status.experts_per_token << '\n'
              << "  Shared experts per MoE layer/token: " << status.shared_expert_count << "\n";
}

}  // namespace

int Cli::run(const std::filesystem::path& model_directory) const {
    Logger logger;
    if (!std::filesystem::is_directory(model_directory)) {
        logger.error("The model directory does not exist.");
        return 1;
    }

    const ModelStatus status = inspect_model(model_directory);
    print_status(status, model_directory);

    if (const auto probe = verify_weight_reader(model_directory)) {
        std::cout << "  Weight reader: verified " << *probe << "\n";
    } else {
        logger.warning("No small BF16 tensor could be read to verify payload access.");
    }

    Tokenizer tokenizer;
    std::string tokenizer_error;
    if (!tokenizer.load(model_directory / "tokenizer.json", tokenizer_error)) {
        logger.error(tokenizer_error);
        return 1;
    }
    std::cout << "  Tokenizer vocabulary: " << tokenizer.vocabulary_size() << " tokens\n";

    std::cout << "\nPrompt> ";
    std::string prompt;
    if (!std::getline(std::cin, prompt) || prompt.empty()) {
        logger.warning("No prompt was supplied.");
        return 0;
    }
    const std::vector<std::uint32_t> prompt_tokens = tokenizer.encode(prompt);
    if (prompt_tokens.empty()) {
        logger.error("The tokenizer could not encode this prompt.");
        return 1;
    }

    // ── Load model and run inference ──────────────────────────────────────────
    std::string runner_error;
    DeepSeekRunner runner(model_directory, runner_error);
    if (!runner.ready()) {
        logger.error("Failed to load model weights: " + runner_error);
        return 1;
    }

    std::cout << "\nGenerating (5 tokens, streaming)...\n";
    runner.generate(tokenizer, prompt_tokens, /*max_new_tokens=*/5);
    std::cout << "\n";
    return 0;
}

}  // namespace litemind
