#include <gtest/gtest.h>
#include "minimizer/orchestration/concurrent_queue.h"
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <algorithm>

using namespace entropy;
using namespace std::chrono_literals;

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST(ConcurrentQueueTest, PushAndTryPop) {
    ConcurrentQueue<int> queue;
    
    // Initially empty
    EXPECT_TRUE(queue.empty());
    EXPECT_EQ(queue.size(), 0);
    
    // Push items
    queue.push(1);
    queue.push(2);
    queue.push(3);
    
    EXPECT_FALSE(queue.empty());
    EXPECT_EQ(queue.size(), 3);
    
    // Pop items in FIFO order
    auto item1 = queue.tryPop();
    ASSERT_TRUE(item1.has_value());
    EXPECT_EQ(*item1, 1);
    
    auto item2 = queue.tryPop();
    ASSERT_TRUE(item2.has_value());
    EXPECT_EQ(*item2, 2);
    
    auto item3 = queue.tryPop();
    ASSERT_TRUE(item3.has_value());
    EXPECT_EQ(*item3, 3);
    
    // Empty after popping all
    EXPECT_TRUE(queue.empty());
    auto item4 = queue.tryPop();
    EXPECT_FALSE(item4.has_value());
}

TEST(ConcurrentQueueTest, TryPopEmpty) {
    ConcurrentQueue<int> queue;
    
    auto item = queue.tryPop();
    EXPECT_FALSE(item.has_value());
}

// ============================================================================
// Timeout Tests
// ============================================================================

TEST(ConcurrentQueueTest, TryPopForTimeout) {
    ConcurrentQueue<int> queue;
    
    auto start = std::chrono::steady_clock::now();
    auto item = queue.tryPopFor(100ms);
    auto end = std::chrono::steady_clock::now();
    
    EXPECT_FALSE(item.has_value());
    
    // Should have waited approximately 100ms
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    EXPECT_GE(duration.count(), 90);  // Allow some variance
    EXPECT_LE(duration.count(), 200); // But not too much
}

TEST(ConcurrentQueueTest, TryPopForSuccess) {
    ConcurrentQueue<int> queue;
    
    // Push from another thread after short delay
    std::thread producer([&queue]() {
        std::this_thread::sleep_for(50ms);
        queue.push(42);
    });
    
    auto start = std::chrono::steady_clock::now();
    auto item = queue.tryPopFor(200ms);
    auto end = std::chrono::steady_clock::now();
    
    producer.join();
    
    ASSERT_TRUE(item.has_value());
    EXPECT_EQ(*item, 42);
    
    // Should have returned before timeout (around 50ms)
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    EXPECT_LT(duration.count(), 150);
}

// ============================================================================
// Signal Done Tests
// ============================================================================

TEST(ConcurrentQueueTest, SignalDoneWakesWaitingThreads) {
    ConcurrentQueue<int> queue;
    std::atomic<int> woken_count{0};
    
    // Start 4 threads waiting on empty queue
    std::vector<std::thread> waiters;
    for (int i = 0; i < 4; ++i) {
        waiters.emplace_back([&queue, &woken_count]() {
            auto item = queue.tryPopFor(5000ms); // Long timeout
            if (!item.has_value()) {
                woken_count++;
            }
        });
    }
    
    // Let threads start waiting
    std::this_thread::sleep_for(50ms);
    
    // Signal done - should wake all threads
    queue.signalDone();
    
    // Wait for all threads
    for (auto& t : waiters) {
        t.join();
    }
    
    // All 4 threads should have been woken with nullopt
    EXPECT_EQ(woken_count, 4);
    EXPECT_TRUE(queue.isDone());
}

TEST(ConcurrentQueueTest, SignalDonePreventsWaiting) {
    ConcurrentQueue<int> queue;
    
    queue.signalDone();
    
    // tryPopFor should return immediately after done
    auto start = std::chrono::steady_clock::now();
    auto item = queue.tryPopFor(1000ms);
    auto end = std::chrono::steady_clock::now();
    
    EXPECT_FALSE(item.has_value());
    
    // Should return very quickly (< 50ms)
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    EXPECT_LT(duration.count(), 50);
}

TEST(ConcurrentQueueTest, SignalDoneWithItemsInQueue) {
    ConcurrentQueue<int> queue;
    
    queue.push(1);
    queue.push(2);
    queue.signalDone();
    
    // Existing items should still be retrievable
    auto item1 = queue.tryPop();
    ASSERT_TRUE(item1.has_value());
    EXPECT_EQ(*item1, 1);
    
    // But tryPopFor returns nullopt after done
    auto item2 = queue.tryPopFor(100ms);
    EXPECT_FALSE(item2.has_value());
}

// ============================================================================
// Single Producer, Single Consumer
// ============================================================================

TEST(ConcurrentQueueTest, SingleProducerSingleConsumer) {
    ConcurrentQueue<int> queue;
    const int num_items = 1000;
    std::vector<int> consumed;
    
    // Producer thread
    std::thread producer([&queue, num_items]() {
        for (int i = 0; i < num_items; ++i) {
            queue.push(i);
        }
    });
    
    // Consumer thread
    std::thread consumer([&queue, &consumed, num_items]() {
        int count = 0;
        while (count < num_items) {
            auto item = queue.tryPopFor(100ms);
            if (item.has_value()) {
                consumed.push_back(*item);
                count++;
            }
        }
    });
    
    producer.join();
    consumer.join();
    
    // Verify all items consumed in order
    EXPECT_EQ(consumed.size(), num_items);
    for (int i = 0; i < num_items; ++i) {
        EXPECT_EQ(consumed[i], i);
    }
}

// ============================================================================
// Multiple Producers, Multiple Consumers
// ============================================================================

TEST(ConcurrentQueueTest, MultipleProducersMultipleConsumers) {
    ConcurrentQueue<int> queue;
    const int num_producers = 4;
    const int num_consumers = 4;
    const int items_per_producer = 250;
    const int total_items = num_producers * items_per_producer;
    
    std::vector<int> consumed;
    std::mutex consumed_mutex;
    
    // Start producers
    std::vector<std::thread> producers;
    for (int p = 0; p < num_producers; ++p) {
        producers.emplace_back([&queue, p, items_per_producer]() {
            int base = p * items_per_producer;
            for (int i = 0; i < items_per_producer; ++i) {
                queue.push(base + i);
            }
        });
    }
    
    // Start consumers
    std::atomic<int> consumed_count{0};
    std::vector<std::thread> consumers;
    for (int c = 0; c < num_consumers; ++c) {
        consumers.emplace_back([&queue, &consumed, &consumed_mutex, &consumed_count, total_items]() {
            while (consumed_count < total_items) {
                auto item = queue.tryPopFor(100ms);
                if (item.has_value()) {
                    {
                        std::lock_guard<std::mutex> lock(consumed_mutex);
                        consumed.push_back(*item);
                    }
                    consumed_count++;
                }
            }
        });
    }
    
    // Wait for all producers
    for (auto& t : producers) {
        t.join();
    }
    
    // Wait for all consumers
    for (auto& t : consumers) {
        t.join();
    }
    
    // Verify all items consumed
    EXPECT_EQ(consumed.size(), total_items);
    
    // Verify no duplicates (sort and check uniqueness)
    std::sort(consumed.begin(), consumed.end());
    auto last = std::unique(consumed.begin(), consumed.end());
    EXPECT_EQ(last, consumed.end()) << "Duplicates found in consumed items";
    
    // Verify all expected values present
    for (int i = 0; i < total_items; ++i) {
        EXPECT_TRUE(std::binary_search(consumed.begin(), consumed.end(), i))
            << "Missing item: " << i;
    }
}

// ============================================================================
// Thread Safety Stress Test
// ============================================================================

TEST(ConcurrentQueueTest, StressTestThreadSafety) {
    ConcurrentQueue<int> queue;
    const int num_threads = 4;
    const int pushes_per_thread = 1000;
    const int total_items = num_threads * pushes_per_thread;
    
    std::atomic<int> push_count{0};
    std::atomic<int> pop_count{0};
    
    // Pusher threads
    std::vector<std::thread> pushers;
    for (int i = 0; i < num_threads; ++i) {
        pushers.emplace_back([&queue, &push_count, pushes_per_thread]() {
            for (int j = 0; j < pushes_per_thread; ++j) {
                queue.push(j);
                push_count++;
            }
        });
    }
    
    // Popper threads
    std::vector<std::thread> poppers;
    for (int i = 0; i < num_threads; ++i) {
        poppers.emplace_back([&queue, &pop_count, total_items]() {
            while (pop_count < total_items) {
                auto item = queue.tryPopFor(50ms);
                if (item.has_value()) {
                    pop_count++;
                }
            }
        });
    }
    
    // Wait for all pushers
    for (auto& t : pushers) {
        t.join();
    }
    
    // Wait for all poppers
    for (auto& t : poppers) {
        t.join();
    }
    
    // Verify counts
    EXPECT_EQ(push_count, total_items);
    EXPECT_EQ(pop_count, total_items);
    EXPECT_TRUE(queue.empty());
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST(ConcurrentQueueTest, MovableTypes) {
    struct MoveOnlyType {
        int value;
        MoveOnlyType(int v) : value(v) {}
        MoveOnlyType(const MoveOnlyType&) = delete;
        MoveOnlyType& operator=(const MoveOnlyType&) = delete;
        MoveOnlyType(MoveOnlyType&&) = default;
        MoveOnlyType& operator=(MoveOnlyType&&) = default;
    };
    
    ConcurrentQueue<MoveOnlyType> queue;
    
    queue.push(MoveOnlyType(42));
    
    auto item = queue.tryPop();
    ASSERT_TRUE(item.has_value());
    EXPECT_EQ(item->value, 42);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(ConcurrentQueueTest, MultipleSignalDone) {
    ConcurrentQueue<int> queue;
    
    // Multiple calls to signalDone should be safe
    queue.signalDone();
    queue.signalDone();
    queue.signalDone();
    
    EXPECT_TRUE(queue.isDone());
}

TEST(ConcurrentQueueTest, PushAfterSignalDone) {
    ConcurrentQueue<int> queue;
    
    queue.signalDone();
    
    // Push should still work (though tryPopFor won't wait)
    queue.push(42);
    
    // tryPop should still work
    auto item = queue.tryPop();
    ASSERT_TRUE(item.has_value());
    EXPECT_EQ(*item, 42);
}

TEST(ConcurrentQueueTest, SizeAndEmptyThreadSafety) {
    ConcurrentQueue<int> queue;
    std::atomic<bool> stop{false};
    
    // Thread that modifies queue
    std::thread modifier([&queue, &stop]() {
        while (!stop) {
            queue.push(1);
            std::this_thread::sleep_for(1ms);
            queue.tryPop();
        }
    });
    
    // Thread that queries size/empty
    std::thread observer([&queue, &stop]() {
        for (int i = 0; i < 100; ++i) {
            size_t s = queue.size();
            bool e = queue.empty();
            // Just verify no crashes or hangs
            (void)s;
            (void)e;
            std::this_thread::sleep_for(1ms);
        }
        stop = true;
    });
    
    modifier.join();
    observer.join();
    
    // No assertion - just verify no crashes
}
