#include "Tokenizer.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Expected path to tokenizer.json.\n";
        return 2;
    }

    litemind::Tokenizer tokenizer;
    std::string error;
    if (!tokenizer.load(std::filesystem::path(argv[1]), error)) {
        std::cerr << error << '\n';
        return 1;
    }

    // These valid UTF-8 code points exercise continuation bytes 0x80–0xBF,
    // including the byte range that the former arithmetic inverse corrupted.
    const std::string input = "Caf\xC3\xA9 \xC2\x80\xC2\x81\xC2\xA0\xC2\xAD \xE4\xB8\x96\xE7\x95\x8C";

    const auto token_ids = tokenizer.encode(input, false);
    const std::string output = tokenizer.decode(token_ids);
    if (output != input) {
        std::size_t first_difference{};
        while (first_difference < input.size() && first_difference < output.size()
               && input[first_difference] == output[first_difference]) {
            ++first_difference;
        }
        std::cerr << "UTF-8 tokenizer round trip failed (" << token_ids.size()
                  << " encoded tokens, " << output.size() << " decoded bytes) at offset " << first_difference
                  << ": expected "
                  << (first_difference < input.size()
                          ? static_cast<unsigned int>(static_cast<unsigned char>(input[first_difference]))
                          : 0U)
                  << ", received "
                  << (first_difference < output.size()
                          ? static_cast<unsigned int>(static_cast<unsigned char>(output[first_difference]))
                          : 0U)
                  << ".\n";
        return 1;
    }

    return 0;
}
