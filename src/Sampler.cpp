#include "Sampler.hpp"

#include "CpuTensor.hpp"

#include <stdexcept>

namespace litemind {

std::size_t Sampler::select_next(const std::span<const float> logits, const SamplingMethod method) {
    if (logits.empty()) {
        throw std::invalid_argument("Sampling requires at least one vocabulary logit.");
    }
    switch (method) {
        case SamplingMethod::Greedy: return CpuTensor::argmax(logits);
    }
    throw std::invalid_argument("Unsupported sampling method.");
}

}  // namespace litemind
