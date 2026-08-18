#include "Cli.hpp"

#include "Config.hpp"
#include "Gemm.hpp"
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
        "      --expert-cache GB   Cap the resident routed experts at GB gigabytes.\n"
        "                          Experts above the cap are returned to the SSD.\n"
        "                          Omit this to let the operating system decide.\n"
        "      --warm              Stream the always-hot weights in before the\n"
        "                          first prompt, so the first token is not slow.\n"
        "\n"
        "Diagnostics:\n"
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

bool Cli::parse(const int argc, char* argv[], CliOptions& options, std::string& error) {
    bool model_directory_set = false;
    bool temperature_set = false;

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
            std::cout << "Prompt (/exit to quit)> " << std::flush;
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
        const std::vector<std::uint32_t> tokens = runner.tokenizer().encode(prompt);
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

        GenerationOptions generation = options.generation;
        generation.on_text = [](const std::string_view fragment) {
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
