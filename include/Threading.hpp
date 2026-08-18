#pragma once

#include <cstddef>
#include <functional>
#include <vector>

namespace litemind {

/**
 * @brief A fixed pool of worker threads with a blocking parallel_for.
 *
 * Decoding one token runs a few hundred matrix-vector products. Spawning
 * threads per product would cost more than the arithmetic, so the workers are
 * created once and parked on a condition variable between calls.
 */
class ThreadPool final {
public:
    /** Creates worker_count workers. Zero means "one per hardware thread". */
    explicit ThreadPool(std::size_t worker_count = 0U);
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /**
     * Runs body(index) for every index in [0, count) and returns once all of
     * them have finished. Runs inline when the pool holds a single worker.
     */
    void parallel_for(std::size_t count, const std::function<void(std::size_t)>& body);

    [[nodiscard]] std::size_t worker_count() const noexcept { return worker_count_; }

    /** The number of hardware threads, with a sane floor when unknown. */
    [[nodiscard]] static std::size_t hardware_threads() noexcept;

private:
    struct State;

    std::size_t worker_count_{1U};
    State* state_{nullptr};
};

}  // namespace litemind
