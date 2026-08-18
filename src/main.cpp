#include "Cli.hpp"

#include <exception>
#include <iostream>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace {

/**
 * Windows consoles default to a legacy code page, which mangles the UTF-8 the
 * tokenizer produces. Switching the output code page makes generated text
 * readable without asking the user to configure their terminal.
 */
void prepare_console() {
#if defined(_WIN32)
    ::SetConsoleOutputCP(CP_UTF8);
    ::SetConsoleCP(CP_UTF8);
#endif
}

}  // namespace

int main(int argc, char* argv[]) {
    prepare_console();
    std::ios::sync_with_stdio(false);

    litemind::CliOptions options;
    std::string error;
    if (!litemind::Cli::parse(argc, argv, options, error)) {
        std::cerr << "[ERROR] " << error << "\n\n";
        litemind::Cli::print_usage(argc > 0 ? argv[0] : "LiteMind");
        return 2;
    }
    if (options.show_help) {
        litemind::Cli::print_usage(argc > 0 ? argv[0] : "LiteMind");
        return 0;
    }

    try {
        return litemind::Cli{}.run(options);
    } catch (const std::exception& failure) {
        std::cerr << "\n[ERROR] " << failure.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "\n[ERROR] An unknown failure stopped the run.\n";
        return 1;
    }
}
