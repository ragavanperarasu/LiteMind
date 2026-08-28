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

    /**
     * Settings file read before the arguments, so an argument always wins.
     * Absent by default is not an error; naming one with --config that does not
     * exist is.
     */
    std::filesystem::path config_path{"litemind.json"};
    bool config_required{false};

    /** Print the work a prompt implies before running it. */
    bool show_plan{true};

    /**
     * Wrap prompts in the checkpoint's conversational frame when it has one.
     * A base checkpoint carries no template, so leaving this on costs nothing;
     * turning it off feeds the prompt raw, which is what the logit comparison
     * against a reference implementation needs.
     */
    bool chat{true};

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
    /**
     * Parses argv, applying the settings file first so an argument overrides it.
     * Returns false and sets error on a malformed argument or settings file.
     */
    [[nodiscard]] static bool parse(int argc, char* argv[], CliOptions& options, std::string& error);

    /**
     * Applies a JSON settings file to options. Returns false and sets error when
     * the file is unreadable or malformed; a missing file is reported through
     * present so an absent default can be ignored.
     */
    [[nodiscard]] static bool apply_config_file(const std::filesystem::path& path,
                                                CliOptions& options, bool& present,
                                                std::string& error);

    /** Prints the usage text. */
    static void print_usage(const std::string& program_name);

    /** Runs the requested action. Returns a process exit code. */
    [[nodiscard]] int run(const CliOptions& options) const;

private:
    /** Prints what is in the model directory without loading any weights. */
    [[nodiscard]] static int inspect(const std::filesystem::path& model_directory);
};

}  // namespace litemind
