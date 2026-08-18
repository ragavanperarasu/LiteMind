#pragma once

#include <mutex>
#include <string_view>

namespace litemind {

/** Severity used by the lightweight console logger. */
enum class LogLevel { Debug, Info, Warning, Error };

/**
 * @brief A small thread-safe logger that writes messages to the console.
 *
 * A Logger is an ordinary object instead of a global service, making lifetime
 * and dependencies explicit in applications and tests.
 */
class Logger final {
public:
    explicit Logger(LogLevel minimum_level = LogLevel::Info) noexcept;

    void set_minimum_level(LogLevel level);
    [[nodiscard]] LogLevel minimum_level() const;

    void log(LogLevel level, std::string_view message) const;
    void debug(std::string_view message) const;
    void info(std::string_view message) const;
    void warning(std::string_view message) const;
    void error(std::string_view message) const;

private:
    [[nodiscard]] static std::string_view label(LogLevel level) noexcept;

    LogLevel minimum_level_;
    mutable std::mutex mutex_;
};

}  // namespace litemind
