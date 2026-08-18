#include "Config.hpp"

namespace litemind {

bool Config::load(const std::filesystem::path& path) {
    if (path.empty()) {
        return false;
    }

    source_path_ = path;
    return true;
}

const std::filesystem::path& Config::source_path() const noexcept {
    return source_path_;
}

}  // namespace litemind
