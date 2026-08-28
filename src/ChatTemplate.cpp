#include "ChatTemplate.hpp"

#include <algorithm>

namespace litemind {
namespace {

/**
 * The literals that identify DeepSeek's frame. Matching on these rather than on
 * the whole program tolerates the whitespace and ordering differences between
 * checkpoint revisions, while still refusing a genuinely different template:
 * every model family spells its turn markers differently, so the markers are
 * what actually distinguish them.
 */
constexpr std::string_view kUserMarker = "'User: '";
constexpr std::string_view kAssistantMarker = "'Assistant: '";
constexpr std::string_view kGenerationCue = "'Assistant:'";

}  // namespace

bool ChatTemplate::recognise(const std::string_view template_text) {
    return template_text.find(kUserMarker) != std::string_view::npos
        && template_text.find(kAssistantMarker) != std::string_view::npos
        && template_text.find(kGenerationCue) != std::string_view::npos;
}

std::string ChatTemplate::apply(const std::string_view user_text,
                                const std::string_view system_text) {
    std::string formatted;
    formatted.reserve(system_text.size() + user_text.size() + 32U);

    // A system message is emitted bare, with no marker of its own.
    if (!system_text.empty()) {
        formatted.append(system_text);
        formatted.append("\n\n");
    }
    formatted.append("User: ");
    formatted.append(user_text);
    formatted.append("\n\n");

    // No trailing space: the template writes 'Assistant:' exactly, and the
    // leading space of the reply is a token the model is trained to produce.
    formatted.append("Assistant:");
    return formatted;
}

std::vector<std::string> ChatTemplate::stop_sequences() {
    // The leading newline matters: it anchors the marker to a turn boundary, so
    // a reply that merely mentions the word "User:" mid-sentence is not cut off.
    return {"\nUser:", "\nAssistant:"};
}

std::size_t StopScanner::unsettled_suffix(const std::string_view text,
                                          const std::vector<std::string>& stops) {
    std::size_t held = 0U;
    for (const std::string& stop : stops) {
        if (stop.empty()) {
            continue;
        }
        // A full match is the caller's business; only genuine prefixes are held.
        const std::size_t longest_prefix = std::min(stop.size() - 1U, text.size());
        for (std::size_t length = longest_prefix; length > held; --length) {
            if (text.compare(text.size() - length, length, stop, 0U, length) == 0) {
                held = length;
                break;
            }
        }
    }
    return held;
}

std::size_t StopScanner::find(const std::string_view text, const std::vector<std::string>& stops,
                              const std::size_t from) {
    std::size_t earliest = std::string_view::npos;
    for (const std::string& stop : stops) {
        if (stop.empty()) {
            continue;
        }
        const std::size_t at = text.find(stop, from);
        if (at != std::string_view::npos && at < earliest) {
            earliest = at;
        }
    }
    return earliest;
}

std::size_t StopScanner::longest(const std::vector<std::string>& stops) {
    std::size_t widest = 0U;
    for (const std::string& stop : stops) {
        widest = std::max(widest, stop.size());
    }
    return widest;
}

}  // namespace litemind
