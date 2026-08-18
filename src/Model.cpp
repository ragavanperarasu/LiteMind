#include "Model.hpp"

#include <utility>

namespace litemind {

Model::Model(Config config, std::filesystem::path model_path)
    : config_(std::move(config)), model_path_(std::move(model_path)) {}

const Config& Model::config() const noexcept { return config_; }
Config& Model::config() noexcept { return config_; }
const std::vector<Tensor>& Model::tensors() const noexcept { return tensors_; }
std::vector<Tensor>& Model::tensors() noexcept { return tensors_; }
const std::filesystem::path& Model::path() const noexcept { return model_path_; }

void Model::set_path(std::filesystem::path model_path) {
    model_path_ = std::move(model_path);
}

}  // namespace litemind
