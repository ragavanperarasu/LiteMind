#include "Logger.hpp"

#include <iostream>

namespace litemind {

Logger::Logger(const LogLevel minimum_level) noexcept : minimum_level_(minimum_level) {}

void Logger::set_minimum_level(const LogLevel level) {
    std::scoped_lock lock(mutex_);
    minimum_level_ = level;
}

LogLevel Logger::minimum_level() const {
    std::scoped_lock lock(mutex_);
    return minimum_level_;
}

void Logger::log(const LogLevel level, const std::string_view message) const {
    std::scoped_lock lock(mutex_);
    if (static_cast<int>(level) < static_cast<int>(minimum_level_)) {
        return;
    }

    std::clog << '[' << label(level) << "] " << message << '\n';
}

void Logger::debug(const std::string_view message) const { log(LogLevel::Debug, message); }
void Logger::info(const std::string_view message) const { log(LogLevel::Info, message); }
void Logger::warning(const std::string_view message) const { log(LogLevel::Warning, message); }
void Logger::error(const std::string_view message) const { log(LogLevel::Error, message); }

std::string_view Logger::label(const LogLevel level) noexcept {
    switch (level) {
        case LogLevel::Debug: return "DEBUG";
        case LogLevel::Info: return "INFO";
        case LogLevel::Warning: return "WARNING";
        case LogLevel::Error: return "ERROR";
    }
    return "UNKNOWN";
}

}  // namespace litemind
