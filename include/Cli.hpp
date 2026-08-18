#pragma once

#include "DeepSeekRunner.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace litemind {

/** Everything the command line can set. */
struct CliOptions final {
    std::filesystem::path model_directory{"models"};
    std::string prompt;
    bool interactive{false};
    bool inspect_only{false};
    bool show_tokens{false};
    bool show_logits{false};
    bool show_help{false};

    RunnerOptions runner{};
    GenerationOptions generation{};
};

/**
 * @brief The console front end.
 *
 * Parsing is separated from running so the argument handling can be exercised
 * without a checkpoint on disk.
 */
class Cli final {
public:
    /** Parses argv. Returns false and sets error on a malformed argument. */
    [[nodiscard]] static bool parse(int argc, char* argv[], CliOptions& options, std::string& error);

    /** Prints the usage text. */
    static void print_usage(const std::string& program_name);

    /** Runs the requested action. Returns a process exit code. */
    [[nodiscard]] int run(const CliOptions& options) const;

private:
    /** Prints what is in the model directory without loading any weights. */
    [[nodiscard]] static int inspect(const std::filesystem::path& model_directory);
};

}  // namespace litemind
