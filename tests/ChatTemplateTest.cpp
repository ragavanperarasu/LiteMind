#include "ChatTemplate.hpp"
#include "TestSupport.hpp"

#include <string>
#include <vector>

namespace {

using litemind::ChatTemplate;
using litemind::StopScanner;
using test_support::check;

/** The template DeepSeek-V2-Lite-Chat ships in tokenizer_config.json. */
constexpr const char* kDeepSeekTemplate =
    "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}"
    "{% endif %}{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}"
    "{{ 'User: ' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'assistant' %}"
    "{{ 'Assistant: ' + message['content'] + eos_token }}{% elif message['role'] == 'system' %}"
    "{{ message['content'] + '\\n\\n' }}{% endif %}{% endfor %}"
    "{% if add_generation_prompt %}{{ 'Assistant:' }}{% endif %}";

void recognition() {
    check(ChatTemplate::recognise(kDeepSeekTemplate), "the DeepSeek template is recognised");
    check(!ChatTemplate::recognise(""), "a base checkpoint has no template");
    // A different family's frame uses different markers and must not be mistaken
    // for DeepSeek's, or prompts would be formatted the wrong way in silence.
    check(!ChatTemplate::recognise("{{ '<|im_start|>user' + message['content'] }}"),
          "a foreign template is refused rather than guessed at");
}

void formatting() {
    // What the template produces for one user turn with add_generation_prompt
    // set. The beginning-of-sequence token belongs to the tokenizer, so it is
    // deliberately absent here.
    check(ChatTemplate::apply("hi") == "User: hi\n\nAssistant:",
          "a user turn ends with the open assistant cue");
    check(ChatTemplate::apply("hi", "Be brief.") == "Be brief.\n\nUser: hi\n\nAssistant:",
          "a system message comes first and carries no marker");
    // Nothing is trimmed or escaped: the prompt reaches the model verbatim.
    check(ChatTemplate::apply("  spaced  ") == "User:   spaced  \n\nAssistant:",
          "prompt text is passed through untouched");
}

void stop_markers() {
    const std::vector<std::string> stops = ChatTemplate::stop_sequences();
    check(StopScanner::longest(stops) == std::string("\nAssistant:").size(),
          "the widest marker is how far back a straddling match can begin");

    // The leading newline anchors a marker to a turn boundary, so a reply that
    // merely mentions the word mid-sentence is not truncated.
    check(StopScanner::find("Ask the User: politely", stops) == std::string::npos,
          "a marker without a turn boundary is not a stop");
    check(StopScanner::find("Hello.\nUser: next", stops) == 6U,
          "a marker at a turn boundary is found at its own start");
    check(StopScanner::find("a\nUser:b\nAssistant:c", stops) == 1U,
          "the earliest of several markers wins");
}

void partial_markers_are_held_back() {
    const std::vector<std::string> stops = ChatTemplate::stop_sequences();

    // The case the scanner exists for: the reply is complete text, but its tail
    // could still become a marker once the next token lands.
    check(StopScanner::unsettled_suffix("done.\nUser", stops) == 5U,
          "a partial marker is held back until it is settled");
    check(StopScanner::unsettled_suffix("done.\n", stops) == 1U,
          "a bare newline could still open a turn");
    check(StopScanner::unsettled_suffix("done.", stops) == 0U,
          "ordinary text is released immediately");

    // A complete marker is the caller's business, not the holder's; reporting it
    // as unsettled would stall the emit rather than cut the reply.
    check(StopScanner::unsettled_suffix("done.\nUser:", stops) == 0U,
          "a complete marker is not treated as a partial one");
}

}  // namespace

int main() {
    recognition();
    formatting();
    stop_markers();
    partial_markers_are_held_back();
    return test_support::report("ChatTemplateTest");
}
