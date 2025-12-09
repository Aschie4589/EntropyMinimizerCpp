#ifndef CONCURRENT_QUEUE_H_
#define CONCURRENT_QUEUE_H_

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <chrono>

namespace entropy {

/**
 * @brief Thread-safe FIFO queue for work-stealing orchestration
 * 
 * Provides thread-safe push/pop operations with timeout support
 * and graceful shutdown signaling. Designed for producer-consumer
 * patterns in the orchestration layer.
 * 
 * Features:
 * - Thread-safe push/pop operations using mutex and condition variables
 * - Non-blocking tryPop() for polling
 * - Blocking tryPopFor() with timeout for responsive shutdown
 * - signalDone() to wake all waiting threads for graceful termination
 * 
 * Usage:
 * @code
 * ConcurrentQueue<Task> queue;
 * 
 * // Producer thread
 * queue.push(task);
 * 
 * // Consumer thread
 * auto item = queue.tryPopFor(std::chrono::milliseconds(100));
 * if (item) {
 *     process(*item);
 * }
 * 
 * // Shutdown
 * queue.signalDone();
 * @endcode
 * 
 * @tparam T Element type (must be movable)
 */
template<typename T>
class ConcurrentQueue {
public:
    /**
     * @brief Construct empty queue
     */
    ConcurrentQueue() : done_(false) {}
    
    /**
     * @brief Destructor - signals done to wake waiting threads
     */
    ~ConcurrentQueue() {
        signalDone();
    }
    
    // Non-copyable
    ConcurrentQueue(const ConcurrentQueue&) = delete;
    ConcurrentQueue& operator=(const ConcurrentQueue&) = delete;
    
    // Movable
    ConcurrentQueue(ConcurrentQueue&&) = default;
    ConcurrentQueue& operator=(ConcurrentQueue&&) = default;
    
    /**
     * @brief Push item onto queue (thread-safe)
     * 
     * Wakes one waiting consumer thread after successful push.
     * 
     * @param item Item to push (will be moved)
     */
    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            queue_.push(std::move(item));
        }
        cv_.notify_one();
    }
    
    /**
     * @brief Try to pop item without blocking
     * 
     * @return Item if queue non-empty, std::nullopt otherwise
     */
    std::optional<T> tryPop() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }
    
    /**
     * @brief Try to pop item with timeout
     * 
     * Blocks until either:
     * 1. An item becomes available (returns item)
     * 2. Timeout expires (returns std::nullopt)
     * 3. signalDone() is called (returns std::nullopt)
     * 
     * This method is designed for responsive shutdown - workers can
     * periodically check for shutdown signals by using short timeouts.
     * 
     * @param timeout Maximum time to wait for item
     * @return Item if available before timeout/done, std::nullopt otherwise
     */
    std::optional<T> tryPopFor(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Wait until: queue has item, done signaled, or timeout
        bool success = cv_.wait_for(lock, timeout, [this]() {
            return !queue_.empty() || done_;
        });
        
        // If done signaled or timeout, return nullopt
        if (done_ || queue_.empty()) {
            return std::nullopt;
        }
        
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }
    
    /**
     * @brief Signal completion and wake all waiting threads
     * 
     * After calling this, all future and current tryPopFor() calls
     * will return std::nullopt immediately. Used for graceful shutdown.
     * 
     * This operation is idempotent - calling multiple times is safe.
     */
    void signalDone() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            done_ = true;
        }
        cv_.notify_all();
    }
    
    /**
     * @brief Get current queue size (thread-safe)
     * 
     * Note: Size may change immediately after this call returns
     * due to concurrent push/pop operations.
     * 
     * @return Current number of items in queue
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
    /**
     * @brief Check if queue is empty (thread-safe)
     * 
     * Note: Empty status may change immediately after this call returns
     * due to concurrent push/pop operations.
     * 
     * @return true if queue is empty, false otherwise
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    
    /**
     * @brief Check if done signal has been raised
     * 
     * @return true if signalDone() has been called
     */
    bool isDone() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return done_;
    }

private:
    mutable std::mutex mutex_;           ///< Protects queue_ and done_
    std::condition_variable cv_;         ///< Signals when items available or done
    std::queue<T> queue_;                ///< Underlying FIFO queue
    bool done_;                          ///< Shutdown signal flag
};

} // namespace entropy

#endif // CONCURRENT_QUEUE_H_
