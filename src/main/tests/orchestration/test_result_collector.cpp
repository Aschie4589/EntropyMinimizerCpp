#include <gtest/gtest.h>
#include "minimizer/orchestration/result_collector.h"
#include <thread>
#include <vector>
#include <chrono>
#include <atomic>
#include <random>

using namespace entropy;

// ============================================================================
// Helper Functions
// ============================================================================

HostVector makeTestVector(int dimension, double value = 1.0) {
    std::vector<std::complex<double>> data(dimension, std::complex<double>(value, 0.0));
    return HostVector(data);
}

RunResult makeSuccessResult(int run_id, double entropy, int iterations = 1000, double runtime = 1.5) {
    RunResult result;
    result.run_id = run_id;
    result.final_entropy = entropy;
    result.final_vector = makeTestVector(5, entropy);
    result.iterations_taken = iterations;
    result.runtime_seconds = runtime;
    result.error_type = RunErrorType::NONE;
    result.error_message = "";
    return result;
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST(ResultCollectorTest, InitialState) {
    ResultCollector collector;
    
    EXPECT_TRUE(collector.empty());
    EXPECT_EQ(collector.numCompleted(), 0);
    EXPECT_EQ(collector.numErrors(), 0);
    EXPECT_EQ(collector.numTotal(), 0);
}

TEST(ResultCollectorTest, AddSingleResult) {
    ResultCollector collector;
    
    auto result = makeSuccessResult(1, 0.5);
    collector.addResult(1, result);
    
    EXPECT_FALSE(collector.empty());
    EXPECT_EQ(collector.numCompleted(), 1);
    EXPECT_EQ(collector.numErrors(), 0);
    EXPECT_EQ(collector.numTotal(), 1);
}

TEST(ResultCollectorTest, AddSingleError) {
    ResultCollector collector;
    
    collector.addError(1, "Device failed");
    
    EXPECT_FALSE(collector.empty());
    EXPECT_EQ(collector.numCompleted(), 0);
    EXPECT_EQ(collector.numErrors(), 1);
    EXPECT_EQ(collector.numTotal(), 1);
}

TEST(ResultCollectorTest, AddMixedResults) {
    ResultCollector collector;
    
    collector.addResult(1, makeSuccessResult(1, 0.5));
    collector.addError(2, "Error in run 2");
    collector.addResult(3, makeSuccessResult(3, 0.3));
    
    EXPECT_EQ(collector.numCompleted(), 2);
    EXPECT_EQ(collector.numErrors(), 1);
    EXPECT_EQ(collector.numTotal(), 3);
}

// ============================================================================
// GetMinimum Tests
// ============================================================================

TEST(ResultCollectorTest, GetMinimumSingleResult) {
    ResultCollector collector;
    
    auto result = makeSuccessResult(1, 0.5);
    collector.addResult(1, result);
    
    auto minimum = collector.getMinimum();
    EXPECT_EQ(minimum.run_id, 1);
    EXPECT_DOUBLE_EQ(minimum.final_entropy, 0.5);
}

TEST(ResultCollectorTest, GetMinimumMultipleResults) {
    ResultCollector collector;
    
    collector.addResult(1, makeSuccessResult(1, 0.8));
    collector.addResult(2, makeSuccessResult(2, 0.3));  // Minimum
    collector.addResult(3, makeSuccessResult(3, 0.5));
    
    auto minimum = collector.getMinimum();
    EXPECT_EQ(minimum.run_id, 2);
    EXPECT_DOUBLE_EQ(minimum.final_entropy, 0.3);
}

TEST(ResultCollectorTest, GetMinimumWithErrors) {
    ResultCollector collector;
    
    collector.addResult(1, makeSuccessResult(1, 0.8));
    collector.addError(2, "Error");
    collector.addResult(3, makeSuccessResult(3, 0.2));  // Minimum
    collector.addError(4, "Another error");
    
    auto minimum = collector.getMinimum();
    EXPECT_EQ(minimum.run_id, 3);
    EXPECT_DOUBLE_EQ(minimum.final_entropy, 0.2);
}

TEST(ResultCollectorTest, GetMinimumEmptyThrows) {
    ResultCollector collector;
    
    EXPECT_THROW(collector.getMinimum(), std::runtime_error);
}

TEST(ResultCollectorTest, GetMinimumOnlyErrorsThrows) {
    ResultCollector collector;
    
    collector.addError(1, "Error 1");
    collector.addError(2, "Error 2");
    
    EXPECT_THROW(collector.getMinimum(), std::runtime_error);
}

// ============================================================================
// GetAllResults Tests
// ============================================================================

TEST(ResultCollectorTest, GetAllResultsEmpty) {
    ResultCollector collector;
    
    auto all = collector.getAllResults();
    EXPECT_TRUE(all.empty());
}

TEST(ResultCollectorTest, GetAllResultsMixed) {
    ResultCollector collector;
    
    collector.addResult(1, makeSuccessResult(1, 0.5));
    collector.addError(2, "Error message");
    collector.addResult(3, makeSuccessResult(3, 0.3));
    
    auto all = collector.getAllResults();
    EXPECT_EQ(all.size(), 3);
    
    // Verify we got both types
    int success_count = 0;
    int error_count = 0;
    for (const auto& r : all) {
        if (r.isSuccess()) {
            success_count++;
        } else {
            error_count++;
        }
    }
    EXPECT_EQ(success_count, 2);
    EXPECT_EQ(error_count, 1);
}

// ============================================================================
// Statistics Tests
// ============================================================================

TEST(ResultCollectorTest, StatisticsSingleResult) {
    ResultCollector collector;
    
    collector.addResult(1, makeSuccessResult(1, 0.5, 1000, 2.0));
    
    auto stats = collector.getStatistics();
    EXPECT_DOUBLE_EQ(stats.mean_entropy, 0.5);
    EXPECT_DOUBLE_EQ(stats.std_entropy, 0.0);
    EXPECT_DOUBLE_EQ(stats.min_entropy, 0.5);
    EXPECT_DOUBLE_EQ(stats.max_entropy, 0.5);
    EXPECT_DOUBLE_EQ(stats.mean_iterations, 1000.0);
    EXPECT_DOUBLE_EQ(stats.mean_runtime, 2.0);
    EXPECT_EQ(stats.num_successful, 1);
    EXPECT_EQ(stats.num_failed, 0);
    EXPECT_DOUBLE_EQ(stats.success_rate, 1.0);
}

TEST(ResultCollectorTest, StatisticsMultipleResults) {
    ResultCollector collector;
    
    collector.addResult(1, makeSuccessResult(1, 0.2, 1000, 1.0));
    collector.addResult(2, makeSuccessResult(2, 0.4, 2000, 2.0));
    collector.addResult(3, makeSuccessResult(3, 0.6, 3000, 3.0));
    
    auto stats = collector.getStatistics();
    EXPECT_DOUBLE_EQ(stats.mean_entropy, 0.4);
    EXPECT_DOUBLE_EQ(stats.min_entropy, 0.2);
    EXPECT_DOUBLE_EQ(stats.max_entropy, 0.6);
    EXPECT_DOUBLE_EQ(stats.mean_iterations, 2000.0);
    EXPECT_DOUBLE_EQ(stats.mean_runtime, 2.0);
    EXPECT_EQ(stats.num_successful, 3);
    EXPECT_EQ(stats.num_failed, 0);
    EXPECT_DOUBLE_EQ(stats.success_rate, 1.0);
    
    // Standard deviation: sqrt(((0.2-0.4)^2 + (0.4-0.4)^2 + (0.6-0.4)^2) / 3)
    // = sqrt((0.04 + 0 + 0.04) / 3) = sqrt(0.08/3) ≈ 0.163299
    EXPECT_NEAR(stats.std_entropy, 0.163299, 1e-5);
}

TEST(ResultCollectorTest, StatisticsWithErrors) {
    ResultCollector collector;
    
    collector.addResult(1, makeSuccessResult(1, 0.3));
    collector.addError(2, "Error");
    collector.addResult(3, makeSuccessResult(3, 0.5));
    collector.addError(4, "Another error");
    
    auto stats = collector.getStatistics();
    EXPECT_EQ(stats.num_successful, 2);
    EXPECT_EQ(stats.num_failed, 2);
    EXPECT_DOUBLE_EQ(stats.success_rate, 0.5);
    EXPECT_DOUBLE_EQ(stats.mean_entropy, 0.4);
}

TEST(ResultCollectorTest, StatisticsEmptyThrows) {
    ResultCollector collector;
    
    EXPECT_THROW(collector.getStatistics(), std::runtime_error);
}

TEST(ResultCollectorTest, StatisticsOnlyErrorsThrows) {
    ResultCollector collector;
    
    collector.addError(1, "Error");
    
    EXPECT_THROW(collector.getStatistics(), std::runtime_error);
}

// ============================================================================
// Clear Tests
// ============================================================================

TEST(ResultCollectorTest, Clear) {
    ResultCollector collector;
    
    collector.addResult(1, makeSuccessResult(1, 0.5));
    collector.addError(2, "Error");
    
    EXPECT_EQ(collector.numTotal(), 2);
    
    collector.clear();
    
    EXPECT_TRUE(collector.empty());
    EXPECT_EQ(collector.numCompleted(), 0);
    EXPECT_EQ(collector.numErrors(), 0);
}

// ============================================================================
// RunResult Tests
// ============================================================================

TEST(RunResultTest, SuccessResult) {
    auto vec = makeTestVector(5);
    RunResult result;
    result.run_id = 1;
    result.final_entropy = 0.5;
    result.final_vector = vec;
    result.iterations_taken = 1000;
    result.runtime_seconds = 1.5;
    
    EXPECT_TRUE(result.isSuccess());
    EXPECT_EQ(result.run_id, 1);
    EXPECT_DOUBLE_EQ(result.final_entropy, 0.5);
    EXPECT_EQ(result.iterations_taken, 1000);
    EXPECT_DOUBLE_EQ(result.runtime_seconds, 1.5);
    EXPECT_TRUE(result.error_message.empty());
}

TEST(RunResultTest, ErrorResult) {
    RunResult result;
    result.run_id = 2;
    result.error_type = RunErrorType::DEVICE_ERROR;
    result.error_message = "Device failure";
    
    EXPECT_FALSE(result.isSuccess());
    EXPECT_EQ(result.run_id, 2);
    EXPECT_EQ(result.error_message, "Device failure");
    EXPECT_TRUE(std::isinf(result.final_entropy));
}

TEST(RunResultTest, DefaultConstructor) {
    RunResult result;
    
    EXPECT_EQ(result.run_id, -1);
    EXPECT_TRUE(std::isinf(result.final_entropy));
    EXPECT_EQ(result.iterations_taken, 0);
}

// ============================================================================
// Thread Safety Tests - Multiple Threads Adding Results
// ============================================================================

TEST(ResultCollectorTest, ThreadSafetyMultipleAdds) {
    ResultCollector collector;
    const int num_threads = 10;
    const int results_per_thread = 100;
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&collector, t, results_per_thread]() {
            for (int i = 0; i < results_per_thread; ++i) {
                int run_id = t * results_per_thread + i;
                double entropy = 0.1 + (run_id % 10) * 0.1;
                collector.addResult(run_id, makeSuccessResult(run_id, entropy));
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(collector.numCompleted(), num_threads * results_per_thread);
    EXPECT_EQ(collector.numErrors(), 0);
}

TEST(ResultCollectorTest, ThreadSafetyMixedOperations) {
    ResultCollector collector;
    const int num_threads = 8;
    const int ops_per_thread = 50;
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&collector, t, ops_per_thread]() {
            for (int i = 0; i < ops_per_thread; ++i) {
                int run_id = t * ops_per_thread + i;
                
                // Mix of results and errors
                if (i % 3 == 0) {
                    collector.addError(run_id, "Error " + std::to_string(run_id));
                } else {
                    double entropy = 0.1 + (i % 10) * 0.1;
                    collector.addResult(run_id, makeSuccessResult(run_id, entropy));
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(collector.numTotal(), num_threads * ops_per_thread);
    
    // Verify we can safely call other methods
    EXPECT_NO_THROW(collector.getMinimum());
    EXPECT_NO_THROW(collector.getStatistics());
    EXPECT_NO_THROW(collector.getAllResults());
}

TEST(ResultCollectorTest, ThreadSafetyConcurrentReads) {
    ResultCollector collector;
    
    // Pre-populate with data
    for (int i = 0; i < 100; ++i) {
        collector.addResult(i, makeSuccessResult(i, 0.1 + i * 0.01));
    }
    
    std::atomic<int> read_count{0};
    const int num_reader_threads = 8;
    const int reads_per_thread = 100;
    
    std::vector<std::thread> readers;
    
    for (int t = 0; t < num_reader_threads; ++t) {
        readers.emplace_back([&collector, &read_count, reads_per_thread]() {
            for (int i = 0; i < reads_per_thread; ++i) {
                // Mix of read operations
                if (i % 4 == 0) {
                    auto min = collector.getMinimum();
                    (void)min;
                } else if (i % 4 == 1) {
                    auto stats = collector.getStatistics();
                    (void)stats;
                } else if (i % 4 == 2) {
                    auto all = collector.getAllResults();
                    (void)all;
                } else {
                    size_t count = collector.numCompleted();
                    (void)count;
                }
                read_count++;
            }
        });
    }
    
    for (auto& reader : readers) {
        reader.join();
    }
    
    EXPECT_EQ(read_count, num_reader_threads * reads_per_thread);
}

TEST(ResultCollectorTest, ThreadSafetyConcurrentReadWrite) {
    ResultCollector collector;
    const int num_writers = 4;
    const int num_readers = 4;
    const int writes_per_thread = 100;
    const int reads_per_thread = 100;
    
    std::vector<std::thread> threads;
    std::atomic<bool> stop_readers{false};
    
    // Writer threads
    for (int t = 0; t < num_writers; ++t) {
        threads.emplace_back([&collector, t, writes_per_thread]() {
            for (int i = 0; i < writes_per_thread; ++i) {
                int run_id = t * writes_per_thread + i;
                double entropy = 0.1 + (run_id % 100) * 0.01;
                collector.addResult(run_id, makeSuccessResult(run_id, entropy));
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        });
    }
    
    // Reader threads
    for (int t = 0; t < num_readers; ++t) {
        threads.emplace_back([&collector, &stop_readers, reads_per_thread]() {
            int reads = 0;
            while (!stop_readers && reads < reads_per_thread) {
                try {
                    if (!collector.empty()) {
                        auto min = collector.getMinimum();
                        (void)min;
                    }
                } catch (const std::runtime_error&) {
                    // Expected if called before any results added
                }
                reads++;
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }
        });
    }
    
    // Wait for writers to finish
    for (int i = 0; i < num_writers; ++i) {
        threads[i].join();
    }
    
    // Signal readers to stop
    stop_readers = true;
    
    // Wait for readers
    for (int i = num_writers; i < num_writers + num_readers; ++i) {
        threads[i].join();
    }
    
    EXPECT_EQ(collector.numCompleted(), num_writers * writes_per_thread);
}

// ============================================================================
// Stress Test - 100 Threads Adding Results
// ============================================================================

TEST(ResultCollectorTest, StressTest100Threads) {
    ResultCollector collector;
    const int num_threads = 100;
    const int results_per_thread = 10;
    
    std::vector<std::thread> threads;
    
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back([&collector, t, results_per_thread]() {
            std::mt19937 rng(t);  // Thread-local RNG
            std::uniform_real_distribution<double> entropy_dist(0.0, 1.0);
            
            for (int i = 0; i < results_per_thread; ++i) {
                int run_id = t * results_per_thread + i;
                double entropy = entropy_dist(rng);
                collector.addResult(run_id, makeSuccessResult(run_id, entropy));
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(collector.numCompleted(), num_threads * results_per_thread);
    
    // Verify all results are accessible
    auto all = collector.getAllResults();
    EXPECT_EQ(all.size(), num_threads * results_per_thread);
    
    // Verify statistics computation
    auto stats = collector.getStatistics();
    EXPECT_EQ(stats.num_successful, num_threads * results_per_thread);
    EXPECT_GE(stats.mean_entropy, 0.0);
    EXPECT_LE(stats.mean_entropy, 1.0);
}
