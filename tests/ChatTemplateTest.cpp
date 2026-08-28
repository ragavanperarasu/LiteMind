#include "ChatTemplate.hpp"
#include "TestSupport.hpp"

namespace {

using litemind::ChatTemplate;
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

}  // namespace

int main() {
    recognition();
    formatting();
    return test_support::report("ChatTemplateTest");
}
