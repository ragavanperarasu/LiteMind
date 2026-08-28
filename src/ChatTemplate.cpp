#include "ChatTemplate.hpp"

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

}  // namespace litemind
