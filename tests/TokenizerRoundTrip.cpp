#include "TestSupport.hpp"
#include "Tokenizer.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

using namespace test_support;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: TokenizerRoundTrip <path to tokenizer.json>\n";
        return 2;
    }

    litemind::Tokenizer tokenizer;
    std::string error;
    if (!tokenizer.load(std::filesystem::path(argv[1]), error)) {
        std::cerr << error << "\n";
        return 1;
    }
    std::cout << "  vocabulary: " << tokenizer.vocabulary_size() << " tokens\n";

    // Byte-level BPE must reproduce its input exactly, whatever the bytes are.
    const std::vector<std::string> inputs{
        "The capital of France is",
        "Caf\xC3\xA9 \xC2\x80\xC2\x81\xC2\xA0\xC2\xAD \xE4\xB8\x96\xE7\x95\x8C",
        "  leading and trailing spaces  ",
        "line one\nline two\r\nline three",
        "numbers 1234567890 and symbols !@#$%^&*()",
        "def add(a, b):\n    return a + b\n",
        "emoji \xF0\x9F\x98\x80 and accents \xC3\xA9\xC3\xA8\xC3\xAA",
        "",
    };

    for (const std::string& input : inputs) {
        const std::vector<std::uint32_t> tokens = tokenizer.encode(input, /*add_bos=*/false);
        const std::string output = tokenizer.decode(tokens);
        check(output == input, "round trip preserves: \"" + input.substr(0U, 40U) + "\"");
    }

    // The beginning-of-sequence token must be added on request and must not
    // contribute any characters when the result is decoded.
    if (tokenizer.has_bos()) {
        const std::vector<std::uint32_t> with_bos = tokenizer.encode("hello", true);
        const std::vector<std::uint32_t> without_bos = tokenizer.encode("hello", false);
        check(with_bos.size() == without_bos.size() + 1U, "add_bos prepends exactly one token");
        check(with_bos.front() == tokenizer.bos_token_id(), "the prepended token is BOS");
        check(tokenizer.decode(with_bos) == "hello", "BOS decodes to nothing");
    }

    // Streaming must produce exactly the same text as decoding in one call,
    // without ever emitting a partial UTF-8 sequence.
    const std::string multibyte = "\xE4\xB8\x96\xE7\x95\x8C \xF0\x9F\x98\x80 caf\xC3\xA9";
    const std::vector<std::uint32_t> tokens = tokenizer.encode(multibyte, false);
    litemind::Tokenizer::StreamDecoder decoder(tokenizer);
    std::string streamed;
    for (const std::uint32_t token : tokens) {
        const std::string fragment = decoder.push(token);
        // Every fragment handed to the console must be valid UTF-8 on its own.
        for (std::size_t index = 0; index < fragment.size();) {
            const auto lead = static_cast<unsigned char>(fragment[index]);
            std::size_t length = 1U;
            if ((lead & 0xE0U) == 0xC0U) length = 2U;
            else if ((lead & 0xF0U) == 0xE0U) length = 3U;
            else if ((lead & 0xF8U) == 0xF0U) length = 4U;
            check(index + length <= fragment.size(), "a streamed fragment is complete UTF-8");
            index += length;
        }
        streamed += fragment;
    }
    streamed += decoder.flush();
    check(streamed == multibyte, "streaming reproduces the full text");

    // Encoding must be stable: the same text always gives the same tokens.
    check(tokenizer.encode("The capital of France is", false)
              == tokenizer.encode("The capital of France is", false),
          "encoding is deterministic");

    return report("TokenizerRoundTrip");
}
