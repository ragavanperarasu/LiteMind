#pragma once

#include <string>
#include <string_view>

namespace litemind {

/**
 * @brief Wraps a prompt the way an instruction-tuned checkpoint was trained to see it.
 *
 * A chat checkpoint is fine-tuned on a fixed conversational frame, not on bare
 * text. Handed a bare fragment it has no assistant turn to complete, so it falls
 * back to continuing the fragment as ordinary prose - the same behaviour as the
 * base model, which is what makes a missing template look like a routing fault.
 *
 * Hugging Face stores that frame as a Jinja program in tokenizer_config.json.
 * Interpreting Jinja would mean shipping a template engine, so instead the
 * frames are written out in C++ and the stored program is matched against them
 * by its distinguishing literals. An unrecognised template is refused rather
 * than guessed at: formatting a prompt the wrong way is worse than not
 * formatting it, because it silently degrades output instead of failing.
 */
class ChatTemplate final {
public:
    /**
     * Returns true when template_text is the DeepSeek conversational frame this
     * class implements. Absent or unfamiliar templates return false.
     */
    [[nodiscard]] static bool recognise(std::string_view template_text);

    /**
     * Formats one user turn and the assistant cue the model is meant to
     * complete. The beginning-of-sequence token is left to the tokenizer, which
     * already prepends it, so the result is encoded with add_bos left on.
     */
    [[nodiscard]] static std::string apply(std::string_view user_text,
                                           std::string_view system_text = {});
};

}  // namespace litemind
