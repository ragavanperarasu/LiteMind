#include "Cli.hpp"
#include "Config.hpp"
#include "Logger.hpp"
#include "Model.hpp"

#include <filesystem>
#include <iostream>

int main(int argc, char* argv[]) {
    std::cout << "LiteMind\nVersion 0.1.0\n";

    litemind::Logger logger;
    litemind::Config config;
    litemind::Model model(config);

    // Debug logging is intentionally filtered by Logger's default INFO threshold.
    logger.debug("Architecture initialized.");
    static_cast<void>(model);

    const std::filesystem::path model_directory = argc > 1 ? argv[1] : "models";
    return litemind::Cli{}.run(model_directory);
}
