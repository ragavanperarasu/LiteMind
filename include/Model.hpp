#pragma once

#include "Config.hpp"
#include "Tensor.hpp"

#include <filesystem>
#include <vector>

namespace litemind {

/**
 * @brief Owns the metadata that identifies a model.
 *
 * Loading remains deliberately separate from this initial ownership model.
 */
class Model final {
public:
    explicit Model(Config config = {}, std::filesystem::path model_path = {});

    [[nodiscard]] const Config& config() const noexcept;
    [[nodiscard]] Config& config() noexcept;
    [[nodiscard]] const std::vector<Tensor>& tensors() const noexcept;
    [[nodiscard]] std::vector<Tensor>& tensors() noexcept;
    [[nodiscard]] const std::filesystem::path& path() const noexcept;

    void set_path(std::filesystem::path model_path);

private:
    Config config_;
    std::vector<Tensor> tensors_;
    std::filesystem::path model_path_;
};

}  // namespace litemind
