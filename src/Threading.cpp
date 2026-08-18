#include "Threading.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>

namespace litemind {

struct ThreadPool::State final {
    std::vector<std::thread> workers;
    std::mutex mutex;
    std::condition_variable work_available;
    std::condition_variable work_finished;

    const std::function<void(std::size_t)>* body{nullptr};
    std::atomic<std::size_t> next_index{0U};
    std::size_t total{0U};
    std::size_t active{0U};
    std::uint64_t generation{0U};
    bool stopping{false};
};

std::size_t ThreadPool::hardware_threads() noexcept {
    const unsigned int detected = std::thread::hardware_concurrency();
    return detected == 0U ? 4U : static_cast<std::size_t>(detected);
}

ThreadPool::ThreadPool(const std::size_t worker_count) {
    worker_count_ = worker_count == 0U ? hardware_threads() : worker_count;
    worker_count_ = std::max<std::size_t>(worker_count_, 1U);
    state_ = new State();

    // One thread is the caller, so only worker_count_ - 1 helpers are spawned.
    for (std::size_t index = 1U; index < worker_count_; ++index) {
        state_->workers.emplace_back([this] {
            State& state = *state_;
            std::uint64_t seen_generation = 0U;
            while (true) {
                std::unique_lock<std::mutex> lock(state.mutex);
                state.work_available.wait(lock, [&state, seen_generation] {
                    return state.stopping || state.generation != seen_generation;
                });
                if (state.stopping) {
                    return;
                }
                seen_generation = state.generation;
                const std::function<void(std::size_t)>* body = state.body;
                const std::size_t total = state.total;
                lock.unlock();

                // Dynamic claiming keeps every worker busy even when the chunks
                // differ in cost, which they do across MoE expert sizes.
                while (true) {
                    const std::size_t item = state.next_index.fetch_add(1U, std::memory_order_relaxed);
                    if (item >= total) {
                        break;
                    }
                    (*body)(item);
                }

                lock.lock();
                if (--state.active == 0U) {
                    state.work_finished.notify_all();
                }
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    if (state_ == nullptr) {
        return;
    }
    {
        std::scoped_lock lock(state_->mutex);
        state_->stopping = true;
    }
    state_->work_available.notify_all();
    for (std::thread& worker : state_->workers) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    delete state_;
    state_ = nullptr;
}

void ThreadPool::parallel_for(const std::size_t count, const std::function<void(std::size_t)>& body) {
    if (count == 0U) {
        return;
    }
    if (worker_count_ <= 1U || state_->workers.empty() || count == 1U) {
        for (std::size_t index = 0; index < count; ++index) {
            body(index);
        }
        return;
    }

    State& state = *state_;
    {
        std::scoped_lock lock(state.mutex);
        state.body = &body;
        state.total = count;
        state.next_index.store(0U, std::memory_order_relaxed);
        state.active = state.workers.size();
        ++state.generation;
    }
    state.work_available.notify_all();

    // The calling thread takes work too rather than idling.
    while (true) {
        const std::size_t item = state.next_index.fetch_add(1U, std::memory_order_relaxed);
        if (item >= count) {
            break;
        }
        body(item);
    }

    std::unique_lock<std::mutex> lock(state.mutex);
    state.work_finished.wait(lock, [&state] { return state.active == 0U; });
    state.body = nullptr;
}

}  // namespace litemind
