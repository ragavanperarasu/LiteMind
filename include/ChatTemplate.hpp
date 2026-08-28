#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

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

    /**
     * The markers that open the next turn, and so mark where a reply ends.
     * Nothing in the frame tells the model to stop after answering, so without
     * these it carries on and plays both sides of the conversation.
     */
    [[nodiscard]] static std::vector<std::string> stop_sequences();
};

/**
 * @brief Finds stop markers in text that arrives a fragment at a time.
 *
 * A marker is produced token by token, so it can straddle two fragments and a
 * partial one can look like ordinary text. Emitting eagerly would leak "\nUser"
 * into a reply whenever the ":" had not arrived yet, so text is released only
 * once it can no longer become part of a marker.
 */
class StopScanner final {
public:
    /**
     * How many trailing bytes of text could still grow into one of stops.
     * Those bytes are not safe to emit yet; everything before them is.
     */
    [[nodiscard]] static std::size_t unsettled_suffix(std::string_view text,
                                                      const std::vector<std::string>& stops);

    /** The earliest complete marker at or after from, or npos when there is none. */
    [[nodiscard]] static std::size_t find(std::string_view text,
                                          const std::vector<std::string>& stops,
                                          std::size_t from = 0U);

    /** The longest marker, which is how far back a straddling match can begin. */
    [[nodiscard]] static std::size_t longest(const std::vector<std::string>& stops);
};

}  // namespace litemind
