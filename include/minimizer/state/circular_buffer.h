#ifndef CIRCULAR_BUFFER_H
#define CIRCULAR_BUFFER_H

#include <vector>
#include <stdexcept>
#include <numeric>

namespace entropy {

/**
 * @brief Fixed-size circular buffer for efficient history storage
 * 
 * @tparam T Element type (typically double for entropy values)
 * 
 * This buffer maintains a fixed-capacity ring buffer that overwrites
 * the oldest elements when full. Optimized for entropy history tracking
 * in minimization runs where millions of iterations occur.
 */
template<typename T>
class CircularBuffer {
public:
    /**
     * @brief Construct a circular buffer with fixed capacity
     * @param capacity Maximum number of elements to store
     * @throws std::invalid_argument if capacity is 0
     */
    explicit CircularBuffer(size_t capacity);

    /**
     * @brief Add element to buffer (overwrites oldest if full)
     * @param value Element to add
     */
    void push(const T& value);

    /**
     * @brief Get element at index (0 = oldest, size-1 = newest)
     * @param index Index from oldest element
     * @return Element at index
     * @throws std::out_of_range if index >= size
     */
    T get(size_t index) const;

    /**
     * @brief Get most recently added element
     * @return Newest element
     * @throws std::logic_error if buffer is empty
     */
    T newest() const;

    /**
     * @brief Get oldest element in buffer
     * @return Oldest element
     * @throws std::logic_error if buffer is empty
     */
    T oldest() const;

    /**
     * @brief Compute average of all elements in buffer
     * @return Average value
     * @throws std::logic_error if buffer is empty
     */
    T average() const;

    /**
     * @brief Get current number of elements
     * @return Number of elements in buffer
     */
    size_t size() const;

    /**
     * @brief Get buffer capacity
     * @return Maximum number of elements
     */
    size_t capacity() const;

    /**
     * @brief Check if buffer is full
     * @return true if size == capacity
     */
    bool full() const;

    /**
     * @brief Check if buffer is empty
     * @return true if size == 0
     */
    bool empty() const;

    /**
     * @brief Clear all elements
     */
    void clear();

private:
    std::vector<T> buffer_;
    size_t capacity_;
    size_t head_;  // Next write position
    size_t size_;  // Current number of elements
};

// Template implementation
template<typename T>
CircularBuffer<T>::CircularBuffer(size_t capacity)
    : capacity_(capacity), head_(0), size_(0) {
    if (capacity == 0) {
        throw std::invalid_argument("CircularBuffer capacity must be > 0");
    }
    buffer_.resize(capacity);
}

template<typename T>
void CircularBuffer<T>::push(const T& value) {
    buffer_[head_] = value;
    head_ = (head_ + 1) % capacity_;
    if (size_ < capacity_) {
        ++size_;
    }
}

template<typename T>
T CircularBuffer<T>::get(size_t index) const {
    if (index >= size_) {
        throw std::out_of_range("CircularBuffer index out of range");
    }
    
    size_t oldest_pos = (head_ + capacity_ - size_) % capacity_;
    size_t actual_pos = (oldest_pos + index) % capacity_;
    return buffer_[actual_pos];
}

template<typename T>
T CircularBuffer<T>::newest() const {
    if (size_ == 0) {
        throw std::logic_error("CircularBuffer is empty");
    }
    size_t newest_pos = (head_ + capacity_ - 1) % capacity_;
    return buffer_[newest_pos];
}

template<typename T>
T CircularBuffer<T>::oldest() const {
    if (size_ == 0) {
        throw std::logic_error("CircularBuffer is empty");
    }
    size_t oldest_pos = (head_ + capacity_ - size_) % capacity_;
    return buffer_[oldest_pos];
}

template<typename T>
T CircularBuffer<T>::average() const {
    if (size_ == 0) {
        throw std::logic_error("Cannot compute average of empty buffer");
    }
    
    T sum = T(0);
    size_t oldest_pos = (head_ + capacity_ - size_) % capacity_;
    
    for (size_t i = 0; i < size_; ++i) {
        size_t pos = (oldest_pos + i) % capacity_;
        sum += buffer_[pos];
    }
    
    return sum / static_cast<T>(size_);
}

template<typename T>
size_t CircularBuffer<T>::size() const {
    return size_;
}

template<typename T>
size_t CircularBuffer<T>::capacity() const {
    return capacity_;
}

template<typename T>
bool CircularBuffer<T>::full() const {
    return size_ == capacity_;
}

template<typename T>
bool CircularBuffer<T>::empty() const {
    return size_ == 0;
}

template<typename T>
void CircularBuffer<T>::clear() {
    head_ = 0;
    size_ = 0;
}

} // namespace entropy

#endif // CIRCULAR_BUFFER_H
