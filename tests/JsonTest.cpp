#include "Json.hpp"
#include "TestSupport.hpp"

using litemind::Json;
using namespace test_support;

int main() {
    std::string error;

    // A config.json shaped document, including the null that DeepSeek-V2-Lite
    // uses for q_lora_rank and the nested rope_scaling object.
    Json document;
    const std::string source = R"({
        "model_type": "deepseek_v2",
        "hidden_size": 2048,
        "q_lora_rank": null,
        "norm_topk_prob": false,
        "routed_scaling_factor": 1.0,
        "rms_norm_eps": 1e-06,
        "eos_token_id": 100001,
        "rope_scaling": {"type": "yarn", "factor": 40, "mscale": 0.707},
        "architectures": ["DeepseekV2ForCausalLM"]
    })";
    check(Json::parse(source, document, error), "a DeepSeek config parses: " + error);
    check(document.is_object(), "the document is an object");
    check(document.string_or("model_type", "") == "deepseek_v2", "model_type reads back");
    check(document.unsigned_or("hidden_size", 0U) == 2048U, "hidden_size reads back");

    // A null must fall through to the caller's default rather than becoming 0.
    check(document.unsigned_or("q_lora_rank", 99U) == 99U, "a null field yields the fallback");
    check(!document.boolean_or("norm_topk_prob", true), "a false boolean reads back");
    check_close(document.number_or("rms_norm_eps", 0.0), 1e-6, 1e-12, "exponent notation parses");
    check(document.unsigned_or("eos_token_id", 0U) == 100001U, "eos_token_id reads back");

    const Json* scaling = document.find("rope_scaling");
    check(scaling != nullptr && scaling->is_object(), "rope_scaling is a nested object");
    if (scaling != nullptr) {
        check_close(scaling->number_or("factor", 0.0), 40.0, 1e-9, "nested factor reads back");
        check_close(scaling->number_or("mscale", 0.0), 0.707, 1e-9, "nested mscale reads back");
    }

    const Json* architectures = document.find("architectures");
    check(architectures != nullptr && architectures->is_array()
              && architectures->elements().size() == 1U,
          "an array member reads back");

    // A key that appears in two objects must resolve by structure, not by the
    // first textual match, which is what broke the previous substring approach.
    Json nested;
    check(Json::parse(R"({"outer": {"size": 7}, "size": 3})", nested, error), "nesting parses");
    check(nested.unsigned_or("size", 0U) == 3U, "the top-level key wins over a nested one");

    // Escapes, including a surrogate pair, must round-trip to UTF-8.
    Json escaped;
    check(Json::parse(R"({"text": "a\\b\"c\né😀"})", escaped, error),
          "escapes parse: " + error);
    check(escaped.string_or("text", "") == "a\\b\"c\n\xC3\xA9\xF0\x9F\x98\x80",
          "escapes and a surrogate pair decode to UTF-8");

    // Malformed input must be rejected, not silently accepted.
    Json broken;
    check(!Json::parse(R"({"a": 1,})", broken, error), "a trailing comma is rejected");
    check(!Json::parse(R"({"a": })", broken, error), "a missing value is rejected");
    check(!Json::parse(R"({"a": 1} extra)", broken, error), "trailing text is rejected");

    return report("JsonTest");
}
