#include "Attention.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace litemind {

std::vector<float> Attention::attend(const std::span<const float> query, const KvCache& cache) {
    if (query.size() != cache.head_dimension() || cache.token_count() == 0U) {
        throw std::invalid_argument("Attention requires a query matching a non-empty KV cache.");
    }

    const float scale = 1.0F / std::sqrt(static_cast<float>(cache.head_dimension()));
    std::vector<float> scores(cache.token_count());
    for (std::size_t token = 0; token < cache.token_count(); ++token) {
        const auto key = cache.key(token);
        float dot_product{};
        for (std::size_t dimension = 0; dimension < query.size(); ++dimension) {
            dot_product += query[dimension] * key[dimension];
        }
        scores[token] = dot_product * scale;
    }

    const float maximum = *std::max_element(scores.begin(), scores.end());
    float normalizer{};
    for (float& score : scores) {
        score = std::exp(score - maximum);
        normalizer += score;
    }

    std::vector<float> output(cache.head_dimension(), 0.0F);
    for (std::size_t token = 0; token < cache.token_count(); ++token) {
        const float probability = scores[token] / normalizer;
        const auto value = cache.value(token);
        for (std::size_t dimension = 0; dimension < output.size(); ++dimension) {
            output[dimension] += probability * value[dimension];
        }
    }
    return output;
}

}  // namespace litemind
