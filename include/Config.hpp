#pragma once

#include <filesystem>

namespace litemind {

/**
 * @brief Records the location of a model configuration file.
 *
 * Parsing is intentionally outside the initial scaffold. Keeping this boundary
 * now prevents model code from depending on a particular configuration format.
 */
class Config final {
public:
    Config() = default;

    /** Records a configuration file path without reading or parsing it. */
    [[nodiscard]] bool load(const std::filesystem::path& path);

    /** Returns the path most recently supplied to load(). */
    [[nodiscard]] const std::filesystem::path& source_path() const noexcept;

private:
    std::filesystem::path source_path_;
};

}  // namespace litemind
