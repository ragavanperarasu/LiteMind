#pragma once

#include <filesystem>

namespace litemind {

/**
 * @brief Console entry point for inspecting a local LiteMind model package.
 *
 * The CLI is intentionally a metadata and execution-planning tool at this
 * stage. It does not load weights or execute model operations.
 */
class Cli final {
public:
    /** Runs the status view and one prompt-planning interaction. */
    [[nodiscard]] int run(const std::filesystem::path& model_directory) const;
};

}  // namespace litemind
