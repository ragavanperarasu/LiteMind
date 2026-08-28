#include "Cli.hpp"

#include "Config.hpp"
#include "Gemm.hpp"
#include "ChatTemplate.hpp"
#include "Json.hpp"
#include "Logger.hpp"
#include "SafeTensor.hpp"
#include "Threading.hpp"
#include "Tokenizer.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <system_error>

namespace litemind {
namespace {

[[nodiscard]] std::string format_bytes(const std::uint64_t bytes) {
    constexpr double unit = 1024.0;
    const char* const names[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    double value = static_cast<double>(bytes);
    std::size_t name = 0U;
    while (value >= unit && name + 1U < std::size(names)) {
        value /= unit;
        ++name;
    }
    std::ostringstream text;
    text << std::fixed << std::setprecision(value < 10.0 ? 2 : 1) << value << ' ' << names[name];
    return text.str();
}

[[nodiscard]] std::optional<double> to_number(const std::string& text) {
    try {
        std::size_t consumed = 0U;
        const double value = std::stod(text, &consumed);
        return consumed == text.size() ? std::optional<double>{value} : std::nullopt;
    } catch (const std::exception&) {
        return std::nullopt;
    }
}

/** Reads the value that follows a flag, reporting a clear error when it is missing. */
[[nodiscard]] bool next_value(int argc, char* argv[], int& index, const std::string& flag,
                              std::string& value, std::string& error) {
    if (index + 1 >= argc) {
        error = flag + " needs a value.";
        return false;
    }
    value = argv[++index];
    return true;
}

}  // namespace

void Cli::print_usage(const std::string& program_name) {
    std::cout <<
        "LiteMind - DeepSeek-V2 mixture-of-experts inference on the CPU\n"
        "\n"
        "Usage:\n"
        "  " << program_name << " [model-directory] [options]\n"
        "\n"
        "The model directory holds the files downloaded from Hugging Face:\n"
        "config.json, tokenizer.json and the .safetensors shards. It defaults\n"
        "to ./models.\n"
        "\n"
        "Prompting:\n"
        "  -p, --prompt TEXT       Run one prompt and exit.\n"
        "  -i, --interactive       Keep asking for prompts until you type /exit.\n"
        "  -n, --max-tokens N      Tokens to generate (default 128).\n"
        "      --context N         Prompt plus generated tokens (default 1024).\n"
"                          Each position costs about 450 KB of key/value\n"
"                          cache, so raise it only as far as you need.\n"
        "\n"
        "Sampling (greedy by default, which is reproducible):\n"
        "      --temp T            Sample with temperature T instead of greedily.\n"
        "      --top-k K           Keep only the K most likely tokens (default 40).\n"
        "      --top-p P           Nucleus sampling threshold (default 0.95).\n"
        "      --repeat-penalty R  Penalise tokens already produced (default 1.0).\n"
        "      --seed S            Seed the sampler for a reproducible run.\n"
        "\n"
        "Memory and speed:\n"
        "  -t, --threads N         Worker threads (default: one per core).\n"
        "      --expert-cache GB   Copy routed experts into a bounded GB-gigabyte arena.\n"
        "                          Experts above the cap are returned to the SSD.\n"
        "                          Omit this to let the operating system decide.\n"
        "      --warm              Stream the always-hot weights in before the\n"
        "                          first prompt, so the first token is not slow.\n"
        "\n"
        "Diagnostics:\n"
        "      --config PATH       Read settings from PATH instead of litemind.json.\n"
        "                          Anything else on the command line overrides it.\n"
        "      --no-plan           Skip the summary of what a prompt will cost.\n"
        "      --no-chat           Send prompts raw, without the checkpoint's chat frame.\n"
        "      --inspect           Report what is in the model directory and exit.\n"
        "      --show-tokens       Print the token IDs the prompt encoded to.\n"
        "      --top-logits N      Print the N highest logits predicted after the\n"
        "                          prompt, before sampling. Compare these against\n"
        "                          tools/reference_logits.py to check the maths.\n"
        "  -q, --quiet             Suppress progress output.\n"
        "  -h, --help              Show this message.\n"
        "\n"
        "Examples:\n"
        "  " << program_name << " models --inspect\n"
        "  " << program_name << " models -p \"The capital of France is\" -n 32\n"
        "  " << program_name << " models -i --expert-cache 4 --warm\n";
}

namespace {

/** Groups a number with thin separators so 5928 reads as 5,928. */
[[nodiscard]] std::string with_separators(const std::uint64_t value) {
    std::string digits = std::to_string(value);
    for (std::size_t position = digits.size(); position > 3U;) {
        position -= 3U;
        digits.insert(position, ",");
    }
    return digits;
}

/** Parameter counts, scaled so a 50 K test model reads as usefully as a 15.7 B one. */
[[nodiscard]] std::string format_parameters(const std::uint64_t count) {
    std::ostringstream text;
    text << std::fixed << std::setprecision(2);
    if (count >= 1000000000U) {
        text << (static_cast<double>(count) / 1e9) << " B";
    } else if (count >= 1000000U) {
        text << (static_cast<double>(count) / 1e6) << " M";
    } else if (count >= 1000U) {
        text << (static_cast<double>(count) / 1e3) << " K";
    } else {
        text << std::setprecision(0) << static_cast<double>(count);
    }
    return text.str();
}

/** What one prompt will cost, derived from the architecture rather than measured. */
struct PromptPlan final {
    std::size_t prompt_tokens{};
    std::size_t max_new_tokens{};
    std::size_t forward_passes{};

    std::size_t moe_layers{};
    std::size_t experts_per_layer{};
    std::size_t experts_per_token{};
    std::uint64_t expert_activations{};
    std::uint64_t expert_bytes{};          ///< One routed expert, gate + up + down.
    std::uint64_t weight_traffic_bytes{};  ///< What those activations have to read.

    std::uint64_t total_parameters{};
    std::uint64_t active_parameters{};  ///< Per token.
};

/**
 * Counts the parameters this architecture declares, and how many of them any one
 * token actually runs through. The routed experts are what separates the two:
 * every layer holds n_routed_experts of them and a token uses num_experts_per_tok.
 */
[[nodiscard]] PromptPlan build_plan(const Config& config, const std::size_t prompt_tokens,
                                    const std::size_t max_new_tokens) {
    PromptPlan plan;
    plan.prompt_tokens = prompt_tokens;
    plan.max_new_tokens = max_new_tokens;
    plan.forward_passes = prompt_tokens + max_new_tokens;

    const std::uint64_t hidden = config.hidden_size;
    const std::uint64_t expert_matrix = static_cast<std::uint64_t>(config.moe_intermediate_size) * hidden;
    const std::uint64_t expert_parameters = 3U * expert_matrix;  // gate, up, down

    plan.experts_per_layer = config.n_routed_experts;
    plan.experts_per_token = 0U;
    plan.moe_layers = 0U;

    std::uint64_t total = 0U;
    std::uint64_t active = 0U;

    // Embeddings and the output head. A token reads one embedding row, but the
    // head is multiplied in full to produce logits.
    total += static_cast<std::uint64_t>(config.vocab_size) * hidden;
    active += hidden;
    if (!config.tie_word_embeddings) {
        total += static_cast<std::uint64_t>(config.vocab_size) * hidden;
    }
    active += static_cast<std::uint64_t>(config.vocab_size) * hidden;

    for (std::size_t layer = 0; layer < config.num_hidden_layers; ++layer) {
        // Attention is touched by every token, so it counts towards both.
        std::uint64_t attention = 0U;
        const std::uint64_t heads = config.num_attention_heads;
        const std::uint64_t qk_head = config.qk_head_dim();
        if (config.q_lora_rank == 0U) {
            attention += heads * qk_head * hidden;
        } else {
            attention += static_cast<std::uint64_t>(config.q_lora_rank) * hidden;
            attention += heads * qk_head * static_cast<std::uint64_t>(config.q_lora_rank);
        }
        attention += (static_cast<std::uint64_t>(config.kv_lora_rank) +
                      static_cast<std::uint64_t>(config.qk_rope_head_dim)) * hidden;
        attention += heads *
                     (static_cast<std::uint64_t>(config.qk_nope_head_dim) +
                      static_cast<std::uint64_t>(config.v_head_dim)) *
                     static_cast<std::uint64_t>(config.kv_lora_rank);
        attention += hidden * heads * static_cast<std::uint64_t>(config.v_head_dim);
        total += attention;
        active += attention;

        total += 2U * hidden;  // The two RMS norm vectors.
        active += 2U * hidden;

        if (!config.is_moe_layer(layer)) {
            const std::uint64_t dense = 3U * static_cast<std::uint64_t>(config.intermediate_size) * hidden;
            total += dense;
            active += dense;
            continue;
        }

        ++plan.moe_layers;
        plan.experts_per_token += config.num_experts_per_tok;

        total += static_cast<std::uint64_t>(config.n_routed_experts) * expert_parameters;
        total += static_cast<std::uint64_t>(config.n_shared_experts) * expert_parameters;
        total += static_cast<std::uint64_t>(config.n_routed_experts) * hidden;  // The router gate.

        active += static_cast<std::uint64_t>(config.num_experts_per_tok) * expert_parameters;
        active += static_cast<std::uint64_t>(config.n_shared_experts) * expert_parameters;
        active += static_cast<std::uint64_t>(config.n_routed_experts) * hidden;
    }

    total += hidden;  // The final norm.
    active += hidden;

    plan.total_parameters = total;
    plan.active_parameters = active;
    plan.expert_bytes = expert_parameters * 2U;  // BF16
    plan.expert_activations =
        static_cast<std::uint64_t>(plan.experts_per_token) * plan.forward_passes;
    plan.weight_traffic_bytes = plan.expert_activations * plan.expert_bytes;
    return plan;
}

/** Prints what the prompt implies, before any of it is run. */
void print_plan(const PromptPlan& plan, const Config& config, const CliOptions& options,
                const std::size_t threads) {
    std::cout << "\n  This prompt needs\n"
              << "    Tokens       " << plan.prompt_tokens << " in the prompt, up to "
              << plan.max_new_tokens << " to generate  ("
              << with_separators(plan.forward_passes) << " forward passes)\n"
              << "    Experts      " << config.num_experts_per_tok << " of "
              << plan.experts_per_layer << " per layer across " << plan.moe_layers
              << " MoE layers = " << plan.experts_per_token << " per token\n"
              << "                 " << with_separators(plan.expert_activations)
              << " expert executions, " << format_bytes(plan.expert_bytes) << " each, "
              << format_bytes(plan.weight_traffic_bytes) << " of weight reads\n"
              << "    Parameters   " << format_parameters(plan.total_parameters) << " in the model, "
              << format_parameters(plan.active_parameters) << " active per token ("
              << std::fixed << std::setprecision(1)
              << (100.0 * static_cast<double>(plan.active_parameters) /
                  static_cast<double>(plan.total_parameters))
              << "%)\n"
              << "    Settings     " << threads << " threads, context "
              << options.runner.context_length << ", ";
    if (options.generation.sampling.method == SamplingMethod::Greedy) {
        std::cout << "greedy sampling";
    } else {
        std::cout << "temperature " << std::setprecision(2)
                  << options.generation.sampling.temperature << ", top-k "
                  << options.generation.sampling.top_k << ", top-p " << std::setprecision(2)
                  << options.generation.sampling.top_p;
    }
    if (options.generation.sampling.repetition_penalty != 1.0F) {
        std::cout << ", repeat penalty " << std::setprecision(2)
                  << options.generation.sampling.repetition_penalty;
    }
    if (options.runner.expert_budget_bytes != 0U) {
        std::cout << ", expert arena " << format_bytes(options.runner.expert_budget_bytes);
    } else {
        std::cout << ", experts left to the page cache";
    }
    std::cout << "\n";
}

}  // namespace

bool Cli::apply_config_file(const std::filesystem::path& path, CliOptions& options, bool& present,
                            std::string& error) {
    present = false;
    std::error_code filesystem_error;
    if (!std::filesystem::is_regular_file(path, filesystem_error)) {
        return true;  // Absent is only an error when the caller asked for this file by name.
    }
    present = true;

    Json document;
    if (!Json::parse_file(path.string(), document, error)) {
        error = path.string() + ": " + error;
        return false;
    }
    if (!document.is_object()) {
        error = path.string() + ": the settings file must be a JSON object.";
        return false;
    }

    options.model_directory = document.string_or("model", options.model_directory.string());
    options.generation.max_new_tokens = static_cast<std::size_t>(
        document.unsigned_or("max_tokens", options.generation.max_new_tokens));
    options.runner.context_length = static_cast<std::size_t>(
        document.unsigned_or("context", options.runner.context_length));
    options.runner.threads =
        static_cast<std::size_t>(document.unsigned_or("threads", options.runner.threads));
    options.runner.warm_start = document.boolean_or("warm", options.runner.warm_start);

    const double budget_gigabytes = document.number_or("expert_cache_gb", -1.0);
    if (budget_gigabytes >= 0.0) {
        options.runner.expert_budget_bytes =
            static_cast<std::uint64_t>(budget_gigabytes * 1024.0 * 1024.0 * 1024.0);
    }

    // A temperature is what asks for sampling, exactly as on the command line.
    const double temperature = document.number_or("temperature", -1.0);
    if (temperature > 0.0) {
        options.generation.sampling.temperature = static_cast<float>(temperature);
        options.generation.sampling.method = SamplingMethod::Temperature;
    }
    options.generation.sampling.top_k =
        static_cast<std::size_t>(document.unsigned_or("top_k", options.generation.sampling.top_k));
    options.generation.sampling.top_p =
        static_cast<float>(document.number_or("top_p", options.generation.sampling.top_p));
    options.generation.sampling.repetition_penalty = static_cast<float>(
        document.number_or("repeat_penalty", options.generation.sampling.repetition_penalty));
    options.generation.sampling.seed =
        document.unsigned_or("seed", options.generation.sampling.seed);

    options.chat = document.boolean_or("chat", options.chat);
    options.show_plan = document.boolean_or("show_plan", options.show_plan);
    options.show_tokens = document.boolean_or("show_tokens", options.show_tokens);
    options.prompt = document.string_or("prompt", options.prompt);
    return true;
}

bool Cli::parse(const int argc, char* argv[], CliOptions& options, std::string& error) {
    bool model_directory_set = false;
    bool temperature_set = false;

    // --config is read before anything else, so the file supplies the defaults
    // and any other argument on the line overrides what the file said.
    for (int index = 1; index + 1 < argc; ++index) {
        if (std::string(argv[index]) == "--config") {
            options.config_path = argv[index + 1];
            options.config_required = true;
            break;
        }
    }
    bool config_present = false;
    if (!apply_config_file(options.config_path, options, config_present, error)) {
        return false;
    }
    if (options.config_required && !config_present) {
        error = "The settings file does not exist: " + options.config_path.string();
        return false;
    }
    // A prompt from the settings file is a default; -i must still be able to win.
    const bool prompt_from_config = !options.prompt.empty();

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        std::string value;

        if (argument == "-h" || argument == "--help") {
            options.show_help = true;
            return true;
        }
        if (argument == "--inspect") {
            options.inspect_only = true;
        } else if (argument == "-i" || argument == "--interactive") {
            options.interactive = true;
        } else if (argument == "--show-tokens") {
            options.show_tokens = true;
        } else if (argument == "--top-logits") {
            if (!next_value(argc, argv, index, argument, value, error)) {
                return false;
            }
            const auto number = to_number(value);
            if (!number || *number < 1.0) {
                error = "--top-logits needs a positive whole number.";
                return false;
            }
            options.generation.top_logits = static_cast<std::size_t>(*number);
            options.show_logits = true;
        } else if (argument == "--config") {
            // Already applied above; step over its value so it is not mistaken
            // for the model directory.
            if (!next_value(argc, argv, index, argument, value, error)) {
                return false;
            }
        } else if (argument == "--no-plan") {
            options.show_plan = false;
        } else if (argument == "--no-chat") {
            options.chat = false;
        } else if (argument == "-q" || argument == "--quiet") {
            options.runner.verbose = false;
            options.generation.show_progress = false;
        } else if (argument == "--warm") {
            options.runner.warm_start = true;
        } else if (argument == "-p" || argument == "--prompt") {
            if (!next_value(argc, argv, index, argument, options.prompt, error)) {
                return false;
            }
        } else if (argument == "-n" || argument == "--max-tokens") {
            if (!next_value(argc, argv, index, argument, value, error)) {
                return false;
            }
            const auto number = to_number(value);
            if (!number || *number < 1.0) {
                error = "--max-tokens needs a positive whole number.";
                return false;
            }
            options.generation.max_new_tokens = static_cast<std::size_t>(*number);
        } else if (argument == "--context") {
            if (!next_value(argc, argv, index, argument, value, error)) {
                return false;
            }
            const auto number = to_number(value);
            if (!number || *number < 2.0) {
                error = "--context needs a whole number of at least 2.";
                return false;
            }
            options.runner.context_length = static_cast<std::size_t>(*number);
        } else if (argument == "-t" || argument == "--threads") {
            if (!next_value(argc, argv, index, argument, value, error)) {
                return false;
            }
            const auto number = to_number(value);
            if (!number || *number < 1.0) {
                error = "--threads needs a positive whole number.";
                return false;
            }
            options.runner.threads = static_cast<std::size_t>(*number);
        } else if (argument == "--temp" || argument == "--temperature") {
            if (!next_value(argc, argv, index, argument, value, error)) {
                return false;
            }
            const auto number = to_number(value);
            if (!number || *number < 0.0) {
                error = "--temp needs a number of at least 0.";
                return false;
            }
            options.generation.sampling.temperature = static_cast<float>(*number);
            temperature_set = true;
        } else if (argument == "--top-k") {
            if (!next_value(argc, argv, index, argument, value, error)) {
                return false;
            }
            const auto number = to_number(value);
            if (!number || *number < 0.0) {
                error = "--top-k needs a whole number of at least 0.";
                return false;
            }
            options.generation.sampling.top_k = static_cast<std::size_t>(*number);
        } else if (argument == "--top-p") {
            if (!next_value(argc, argv, index, argument, value, error)) {
                return false;
            }
            const auto number = to_number(value);
            if (!number || *number <= 0.0 || *number > 1.0) {
                error = "--top-p needs a number greater than 0 and at most 1.";
                return false;
            }
            options.generation.sampling.top_p = static_cast<float>(*number);
        } else if (argument == "--repeat-penalty") {
            if (!next_value(argc, argv, index, argument, value, error)) {
                return false;
            }
            const auto number = to_number(value);
            if (!number || *number <= 0.0) {
                error = "--repeat-penalty needs a positive number.";
                return false;
            }
            options.generation.sampling.repetition_penalty = static_cast<float>(*number);
        } else if (argument == "--seed") {
            if (!next_value(argc, argv, index, argument, value, error)) {
                return false;
            }
            const auto number = to_number(value);
            if (!number || *number < 0.0) {
                error = "--seed needs a whole number of at least 0.";
                return false;
            }
            options.generation.sampling.seed = static_cast<std::uint64_t>(*number);
        } else if (argument == "--expert-cache") {
            if (!next_value(argc, argv, index, argument, value, error)) {
                return false;
            }
            const auto number = to_number(value);
            if (!number || *number < 0.0) {
                error = "--expert-cache needs a number of gigabytes of at least 0.";
                return false;
            }
            options.runner.expert_budget_bytes =
                static_cast<std::uint64_t>(*number * 1024.0 * 1024.0 * 1024.0);
        } else if (!argument.empty() && argument.front() == '-') {
            error = "Unrecognised option: " + argument;
            return false;
        } else if (!model_directory_set) {
            options.model_directory = argument;
            model_directory_set = true;
        } else {
            error = "Unexpected extra argument: " + argument;
            return false;
        }
    }

    // Naming a temperature is what asks for sampling; otherwise stay greedy.
    if (temperature_set && options.generation.sampling.temperature > 0.0F) {
        options.generation.sampling.method = SamplingMethod::Temperature;
    }
    // Asking for the interactive loop discards a prompt the settings file
    // supplied; a prompt given on the line is an explicit request and stays.
    if (options.interactive && prompt_from_config) {
        options.prompt.clear();
    }
    if (options.prompt.empty() && !options.inspect_only) {
        options.interactive = true;
    }
    return true;
}

int Cli::inspect(const std::filesystem::path& model_directory) {
    Logger logger;
    std::error_code filesystem_error;
    if (!std::filesystem::is_directory(model_directory, filesystem_error)) {
        logger.error("The model directory does not exist: " + model_directory.string());
        return 1;
    }

    std::cout << "Model directory: " << std::filesystem::absolute(model_directory).string() << "\n\n";

    Config config;
    std::string error;
    if (config.load(model_directory, error)) {
        std::cout << "config.json\n  " << config.summary() << "\n"
                  << "  rms_norm_eps=" << config.rms_norm_eps << " rope_theta=" << config.rope_theta
                  << " softmax_scale=" << config.softmax_scale() << "\n";
        if (config.rope_scaling.enabled) {
            std::cout << "  rope_scaling: " << config.rope_scaling.type
                      << " factor=" << config.rope_scaling.factor
                      << " original_context=" << config.rope_scaling.original_max_position_embeddings
                      << " mscale=" << config.rope_scaling.mscale << "/"
                      << config.rope_scaling.mscale_all_dim << "\n";
        }
        std::cout << "  routing: " << config.topk_method << ", norm_topk_prob="
                  << (config.norm_topk_prob ? "true" : "false")
                  << ", routed_scaling_factor=" << config.routed_scaling_factor << "\n\n";
    } else {
        std::cout << "config.json\n  NOT USABLE: " << error << "\n\n";
    }

    Tokenizer tokenizer;
    if (tokenizer.load(model_directory, error)) {
        std::cout << "tokenizer.json\n  " << tokenizer.vocabulary_size() << " tokens";
        if (tokenizer.has_bos()) {
            std::cout << ", bos=" << tokenizer.bos_token_id();
        }
        std::cout << ", eos=" << tokenizer.eos_token_id() << "\n\n";
    } else {
        std::cout << "tokenizer.json\n  NOT USABLE: " << error << "\n\n";
    }

    std::uint64_t total_bytes = 0U;
    std::uint64_t total_tensors = 0U;
    std::size_t shard_count = 0U;
    std::vector<std::filesystem::path> shard_paths;
    for (const auto& entry : std::filesystem::directory_iterator(model_directory, filesystem_error)) {
        if (entry.is_regular_file(filesystem_error) && entry.path().extension() == ".safetensors") {
            shard_paths.push_back(entry.path());
        }
    }
    std::sort(shard_paths.begin(), shard_paths.end());

    std::cout << "Weight shards\n";
    for (const std::filesystem::path& path : shard_paths) {
        SafeTensor shard;
        if (!shard.open(path, error)) {
            std::cout << "  " << path.filename().string() << "  UNREADABLE: " << error << "\n";
            continue;
        }
        ++shard_count;
        total_tensors += shard.tensors().size();
        total_bytes += shard.file().size();
        std::cout << "  " << path.filename().string() << "  " << format_bytes(shard.file().size())
                  << ", " << shard.tensors().size() << " tensors\n";
    }
    if (shard_paths.empty()) {
        std::cout << "  none found - download the .safetensors files into this directory.\n";
    }
    std::cout << "  total: " << shard_count << " shard(s), " << total_tensors << " tensors, "
              << format_bytes(total_bytes) << "\n\n";

    std::cout << "Host\n  " << ThreadPool::hardware_threads() << " hardware threads, "
              << gemm::kernel_description() << "\n";
    return 0;
}

int Cli::run(const CliOptions& options) const {
    Logger logger;
    if (options.inspect_only) {
        return inspect(options.model_directory);
    }

    std::error_code filesystem_error;
    if (!std::filesystem::is_directory(options.model_directory, filesystem_error)) {
        logger.error("The model directory does not exist: " + options.model_directory.string());
        logger.error("Pass the directory holding config.json, tokenizer.json and the "
                     ".safetensors shards, or run with --help.");
        return 1;
    }

    if (options.runner.verbose) {
        std::cout << "LiteMind 0.2.0 - DeepSeek-V2 mixture-of-experts inference on the CPU\n"
                  << "  " << gemm::kernel_description() << ", "
                  << (options.runner.threads == 0U ? ThreadPool::hardware_threads()
                                                   : options.runner.threads)
                  << " threads\n";
    }

    DeepSeekRunner runner;
    std::string error;
    const auto load_start = std::chrono::steady_clock::now();
    if (!runner.load(options.model_directory, options.runner, error)) {
        logger.error(error);
        return 1;
    }
    const double load_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - load_start).count();

    if (options.runner.verbose) {
        std::cout << "  " << runner.config().summary() << "\n"
                  << "  Ready in " << std::fixed << std::setprecision(1) << load_seconds << " s.\n\n"
                  << runner.memory_report() << "\n";
    }

    // Whether prompts get the conversational frame is decided once, from the
    // checkpoint itself: a base model has no template and is left alone.
    const std::string& stored_template = runner.tokenizer().chat_template();
    const bool recognised = ChatTemplate::recognise(stored_template);
    const bool use_chat_template = options.chat && recognised;
    if (options.runner.verbose) {
        if (use_chat_template) {
            std::cout << "  Chat template: applying the checkpoint's User/Assistant frame.\n\n";
        } else if (!stored_template.empty() && !recognised) {
            logger.warning("This checkpoint has a chat_template LiteMind does not recognise. "
                        "Prompts are being sent unformatted, so replies may wander.");
        } else if (!stored_template.empty() && !options.chat) {
            std::cout << "  Chat template: present but disabled, prompts sent raw.\n\n";
        }
    }

    // A single prompt runs once; otherwise keep reading prompts from the console.
    std::vector<std::string> prompts;
    if (!options.prompt.empty()) {
        prompts.push_back(options.prompt);
    }

    while (true) {
        std::string prompt;
        if (!prompts.empty()) {
            prompt = prompts.front();
            prompts.erase(prompts.begin());
        } else if (options.interactive) {
            std::cout << "Enter prompt (/exit to quit)> " << std::flush;
            if (!std::getline(std::cin, prompt)) {
                std::cout << "\n";
                break;
            }
            if (prompt == "/exit" || prompt == "/quit") {
                break;
            }
            if (prompt.empty()) {
                continue;
            }
        } else {
            break;
        }
        // An instruction-tuned checkpoint expects its conversational frame. Fed a
        // bare fragment it completes the fragment instead of answering it, which
        // reads like a model fault rather than a formatting one.
        const std::string formatted = use_chat_template ? ChatTemplate::apply(prompt) : prompt;
        const std::vector<std::uint32_t> tokens = runner.tokenizer().encode(formatted);
        if (tokens.empty()) {
            logger.error("The tokenizer produced no tokens for this prompt.");
            continue;
        }
        if (options.show_tokens) {
            std::cout << "  " << tokens.size() << " prompt tokens:\n";
            for (const std::uint32_t token : tokens) {
                std::cout << "    " << std::setw(6) << token << "  "
                          << runner.tokenizer().token_text(token) << "\n";
            }
        }

        if (options.show_plan && options.runner.verbose) {
            const PromptPlan plan =
                build_plan(runner.config(), tokens.size(), options.generation.max_new_tokens);
            print_plan(plan, runner.config(), options,
                       options.runner.threads == 0U ? ThreadPool::hardware_threads()
                                                    : options.runner.threads);
        }

        GenerationOptions generation = options.generation;
        if (use_chat_template) {
            generation.stop_sequences = ChatTemplate::stop_sequences();
        }
        // The heading waits for the first fragment, so it lands after the
        // prefill progress rather than above it.
        bool answer_started = false;
        const bool announce_answer = options.show_plan && options.runner.verbose;
        generation.on_text = [&answer_started, announce_answer](const std::string_view fragment) {
            if (announce_answer && !answer_started) {
                std::cout << "\n  Answer\n    ";
                answer_started = true;
            }
            std::cout << fragment << std::flush;
        };

        GenerationResult result;
        if (!runner.generate(tokens, generation, result, error)) {
            logger.error(error);
            continue;
        }

        std::cout << "\n";
        if (options.show_logits && !result.prompt_top_logits.empty()) {
            std::cout << "\n  Top " << result.prompt_top_logits.size()
                      << " logits for the token after the prompt:\n";
            for (const auto& [token, logit] : result.prompt_top_logits) {
                std::cout << "    " << std::setw(6) << token << "  " << std::setw(10)
                          << std::fixed << std::setprecision(4) << logit << "  "
                          << runner.tokenizer().token_text(token) << "\n";
            }
        }
        if (options.runner.verbose) {
            std::cout << "\n  " << result.tokens.size() << " tokens in " << std::fixed
                      << std::setprecision(1) << result.decode_seconds << " s ("
                      << std::setprecision(2) << result.tokens_per_second() << " tok/s), stopped on "
                      << result.stop_reason << ".\n";
            if (runner.expert_cache().enabled()) {
                const ExpertCache& cache = runner.expert_cache();
                const std::uint64_t requests = cache.loads() + cache.hits();
                const double hit_rate =
                    requests == 0U ? 0.0 : 100.0 * static_cast<double>(cache.hits())
                                               / static_cast<double>(requests);
                std::cout << "  Experts: " << requests << " requests, " << std::setprecision(1)
                          << hit_rate << "% already resident, " << format_bytes(cache.bytes_streamed())
                          << " streamed from disk.\n";
            }
            std::cout << "\n";
        }

        if (!options.interactive) {
            break;
        }
    }

    return 0;
}

}  // namespace litemind
