#include <gtest/gtest.h>
#include "minimizer/orchestration/resource_monitor.h"
#include <thread>
#include <chrono>
#include <atomic>
#include <fstream>
#include <filesystem>
#include <cstdlib>

using namespace entropy;
using namespace std::chrono_literals;

// ============================================================================
// Test Fixture with Temp File Management
// ============================================================================

class ResourceMonitorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create unique temp file for this test
        temp_file_ = "/tmp/entropy_resource_test_" + std::to_string(getpid()) + ".yml";
        
        // Clear any existing environment variables
        unsetenv("ENTROPY_MAX_GPUS");
        unsetenv("ENTROPY_MAX_CPUS");
    }
    
    void TearDown() override {
        // Clean up temp file
        if (std::filesystem::exists(temp_file_)) {
            std::filesystem::remove(temp_file_);
        }
        
        // Clear environment variables
        unsetenv("ENTROPY_MAX_GPUS");
        unsetenv("ENTROPY_MAX_CPUS");
    }
    
    void writeConfigFile(int max_gpus, int max_cpus, int poll_interval_ms = 5000) {
        std::ofstream file(temp_file_);
        file << "resources:\n";
        file << "  max_gpus: " << max_gpus << "\n";
        file << "  max_cpus: " << max_cpus << "\n";
        file << "  poll_interval: " << poll_interval_ms << "\n";
        file.close();
    }
    
    void writeNestedConfigFile(int max_gpus, int max_cpus) {
        std::ofstream file(temp_file_);
        file << "orchestration:\n";
        file << "  resources:\n";
        file << "    max_gpus: " << max_gpus << "\n";
        file << "    max_cpus: " << max_cpus << "\n";
        file.close();
    }
    
    std::string temp_file_;
};

// ============================================================================
// Basic Functionality Tests
// ============================================================================

TEST_F(ResourceMonitorTest, ConstructorSetsInitialLimits) {
    ResourceLimits limits;
    limits.max_gpus = 2;
    limits.max_cpus = 4;
    
    ResourceMonitor monitor(limits);
    
    auto current = monitor.getCurrentLimits();
    EXPECT_EQ(current.max_gpus, 2);
    EXPECT_EQ(current.max_cpus, 4);
}

TEST_F(ResourceMonitorTest, InitiallyNotRunning) {
    ResourceLimits limits;
    ResourceMonitor monitor(limits);
    
    EXPECT_FALSE(monitor.isRunning());
}

TEST_F(ResourceMonitorTest, StartSetsRunning) {
    ResourceLimits limits;
    ResourceMonitor monitor(limits);
    
    monitor.start([](const ResourceLimits&) {});
    
    EXPECT_TRUE(monitor.isRunning());
    
    monitor.stop();
}

TEST_F(ResourceMonitorTest, StopClearsRunning) {
    ResourceLimits limits;
    ResourceMonitor monitor(limits);
    
    monitor.start([](const ResourceLimits&) {});
    EXPECT_TRUE(monitor.isRunning());
    
    monitor.stop();
    EXPECT_FALSE(monitor.isRunning());
}

TEST_F(ResourceMonitorTest, MultipleStopCallsSafe) {
    ResourceLimits limits;
    ResourceMonitor monitor(limits);
    
    monitor.start([](const ResourceLimits&) {});
    monitor.stop();
    monitor.stop();  // Should not crash
    monitor.stop();  // Should not crash
    
    EXPECT_FALSE(monitor.isRunning());
}

TEST_F(ResourceMonitorTest, MultipleStartCallsIgnored) {
    ResourceLimits limits;
    limits.poll_interval = 100ms;
    ResourceMonitor monitor(limits);
    
    std::atomic<int> callback_count{0};
    
    monitor.start([&callback_count](const ResourceLimits&) {
        callback_count++;
    });
    
    // Second start should be ignored
    monitor.start([](const ResourceLimits&) {
        FAIL() << "Second callback should not be registered";
    });
    
    EXPECT_TRUE(monitor.isRunning());
    monitor.stop();
}

// ============================================================================
// ResourceLimits Tests
// ============================================================================

TEST(ResourceLimitsTest, EqualityOperator) {
    ResourceLimits a, b;
    a.max_gpus = 2;
    a.max_cpus = 4;
    b.max_gpus = 2;
    b.max_cpus = 4;
    
    EXPECT_EQ(a, b);
    
    b.max_gpus = 3;
    EXPECT_NE(a, b);
}

TEST(ResourceLimitsTest, ToStringFormat) {
    ResourceLimits limits;
    limits.max_gpus = 2;
    limits.max_cpus = 4;
    limits.poll_interval = 5000ms;
    limits.config_file = "/path/to/config.yml";
    
    std::string str = limits.to_string();
    EXPECT_NE(str.find("max_gpus=2"), std::string::npos);
    EXPECT_NE(str.find("max_cpus=4"), std::string::npos);
    EXPECT_NE(str.find("5000ms"), std::string::npos);
}

// ============================================================================
// File Polling Tests
// ============================================================================

TEST_F(ResourceMonitorTest, PollFileDetectsChanges) {
    // Write initial config
    writeConfigFile(2, 4);
    
    ResourceLimits limits;
    limits.max_gpus = 2;
    limits.max_cpus = 4;
    limits.config_file = temp_file_;
    limits.poll_interval = 100ms;
    
    std::atomic<int> callback_count{0};
    ResourceLimits changed_limits;
    std::mutex callback_mutex;
    
    ResourceMonitor monitor(limits);
    monitor.start([&](const ResourceLimits& new_limits) {
        std::lock_guard<std::mutex> lock(callback_mutex);
        callback_count++;
        changed_limits = new_limits;
    });
    
    // Wait for initial poll
    std::this_thread::sleep_for(150ms);
    
    // Update config file
    writeConfigFile(4, 8);
    
    // Wait for change detection
    std::this_thread::sleep_for(250ms);
    
    monitor.stop();
    
    EXPECT_GT(callback_count, 0);
    EXPECT_EQ(changed_limits.max_gpus, 4);
    EXPECT_EQ(changed_limits.max_cpus, 8);
}

TEST_F(ResourceMonitorTest, NoCallbackOnIdenticalConfig) {
    // Write initial config
    writeConfigFile(2, 4);
    
    ResourceLimits limits;
    limits.max_gpus = 2;
    limits.max_cpus = 4;
    limits.config_file = temp_file_;
    limits.poll_interval = 50ms;
    
    std::atomic<int> callback_count{0};
    
    ResourceMonitor monitor(limits);
    monitor.start([&](const ResourceLimits&) {
        callback_count++;
    });
    
    // Wait for several poll cycles
    std::this_thread::sleep_for(200ms);
    
    monitor.stop();
    
    // Should not trigger callback since values match initial
    EXPECT_EQ(callback_count, 0);
}

TEST_F(ResourceMonitorTest, NestedYamlStructure) {
    // Write nested config
    writeNestedConfigFile(3, 6);
    
    ResourceLimits limits;
    limits.max_gpus = 1;
    limits.max_cpus = 2;
    limits.config_file = temp_file_;
    limits.poll_interval = 100ms;
    
    std::atomic<bool> callback_triggered{false};
    ResourceLimits changed_limits;
    
    ResourceMonitor monitor(limits);
    monitor.start([&](const ResourceLimits& new_limits) {
        callback_triggered = true;
        changed_limits = new_limits;
    });
    
    std::this_thread::sleep_for(250ms);
    monitor.stop();
    
    EXPECT_TRUE(callback_triggered);
    EXPECT_EQ(changed_limits.max_gpus, 3);
    EXPECT_EQ(changed_limits.max_cpus, 6);
}

TEST_F(ResourceMonitorTest, MissingFileDoesNotCrash) {
    ResourceLimits limits;
    limits.max_gpus = 2;
    limits.max_cpus = 4;
    limits.config_file = "/nonexistent/file.yml";
    limits.poll_interval = 50ms;
    
    ResourceMonitor monitor(limits);
    monitor.start([](const ResourceLimits&) {
        FAIL() << "Callback should not be triggered for missing file";
    });
    
    std::this_thread::sleep_for(150ms);
    
    // Should not crash
    EXPECT_NO_THROW(monitor.stop());
}

TEST_F(ResourceMonitorTest, InvalidYamlDoesNotCrash) {
    // Write invalid YAML
    std::ofstream file(temp_file_);
    file << "invalid: yaml: content:\n";
    file << "  - unclosed:\n";
    file << "    bracket [\n";
    file.close();
    
    ResourceLimits limits;
    limits.max_gpus = 2;
    limits.max_cpus = 4;
    limits.config_file = temp_file_;
    limits.poll_interval = 50ms;
    
    ResourceMonitor monitor(limits);
    monitor.start([](const ResourceLimits&) {});
    
    std::this_thread::sleep_for(150ms);
    
    // Should not crash
    EXPECT_NO_THROW(monitor.stop());
}

// ============================================================================
// Environment Variable Tests
// ============================================================================

TEST_F(ResourceMonitorTest, ReadFromEnvironmentVariables) {
    setenv("ENTROPY_MAX_GPUS", "3", 1);
    setenv("ENTROPY_MAX_CPUS", "6", 1);
    
    ResourceLimits limits;
    limits.max_gpus = 1;
    limits.max_cpus = 2;
    limits.poll_interval = 100ms;
    
    std::atomic<bool> callback_triggered{false};
    ResourceLimits changed_limits;
    
    ResourceMonitor monitor(limits);
    monitor.start([&](const ResourceLimits& new_limits) {
        callback_triggered = true;
        changed_limits = new_limits;
    });
    
    std::this_thread::sleep_for(250ms);
    monitor.stop();
    
    EXPECT_TRUE(callback_triggered);
    EXPECT_EQ(changed_limits.max_gpus, 3);
    EXPECT_EQ(changed_limits.max_cpus, 6);
}

TEST_F(ResourceMonitorTest, EnvironmentOverridesFile) {
    // Write file with one set of values
    writeConfigFile(2, 4);
    
    // Set environment variables with different values
    setenv("ENTROPY_MAX_GPUS", "5", 1);
    setenv("ENTROPY_MAX_CPUS", "10", 1);
    
    ResourceLimits limits;
    limits.max_gpus = 1;
    limits.max_cpus = 2;
    limits.config_file = temp_file_;
    limits.poll_interval = 100ms;
    
    ResourceLimits changed_limits;
    
    ResourceMonitor monitor(limits);
    monitor.start([&](const ResourceLimits& new_limits) {
        changed_limits = new_limits;
    });
    
    std::this_thread::sleep_for(250ms);
    monitor.stop();
    
    // Environment variables should take precedence
    EXPECT_EQ(changed_limits.max_gpus, 5);
    EXPECT_EQ(changed_limits.max_cpus, 10);
}

TEST_F(ResourceMonitorTest, InvalidEnvironmentVariableIgnored) {
    setenv("ENTROPY_MAX_GPUS", "not_a_number", 1);
    setenv("ENTROPY_MAX_CPUS", "also_invalid", 1);
    
    ResourceLimits limits;
    limits.max_gpus = 2;
    limits.max_cpus = 4;
    limits.poll_interval = 50ms;
    
    ResourceMonitor monitor(limits);
    monitor.start([](const ResourceLimits&) {
        FAIL() << "Should not trigger callback for invalid env vars";
    });
    
    std::this_thread::sleep_for(150ms);
    monitor.stop();
    
    // Should keep original values
    auto current = monitor.getCurrentLimits();
    EXPECT_EQ(current.max_gpus, 2);
    EXPECT_EQ(current.max_cpus, 4);
}

// ============================================================================
// Rapid Change Tests
// ============================================================================

TEST_F(ResourceMonitorTest, MultipleRapidChanges) {
    writeConfigFile(1, 2);
    
    ResourceLimits limits;
    limits.max_gpus = 1;
    limits.max_cpus = 2;
    limits.config_file = temp_file_;
    limits.poll_interval = 50ms;
    
    std::atomic<int> callback_count{0};
    std::vector<ResourceLimits> all_changes;
    std::mutex changes_mutex;
    
    ResourceMonitor monitor(limits);
    monitor.start([&](const ResourceLimits& new_limits) {
        callback_count++;
        std::lock_guard<std::mutex> lock(changes_mutex);
        all_changes.push_back(new_limits);
    });
    
    // Make several rapid changes
    std::this_thread::sleep_for(60ms);
    writeConfigFile(2, 4);
    
    std::this_thread::sleep_for(60ms);
    writeConfigFile(3, 6);
    
    std::this_thread::sleep_for(60ms);
    writeConfigFile(4, 8);
    
    std::this_thread::sleep_for(100ms);
    monitor.stop();
    
    // Should have detected at least some changes
    EXPECT_GT(callback_count, 0);
    EXPECT_GT(all_changes.size(), 0);
}

// ============================================================================
// Graceful Shutdown Tests
// ============================================================================

TEST_F(ResourceMonitorTest, GracefulShutdownWhilePolling) {
    writeConfigFile(2, 4);
    
    ResourceLimits limits;
    limits.max_gpus = 2;
    limits.max_cpus = 4;
    limits.config_file = temp_file_;
    limits.poll_interval = 1000ms;  // Long interval
    
    ResourceMonitor monitor(limits);
    monitor.start([](const ResourceLimits&) {});
    
    // Stop immediately - should terminate quickly despite long poll interval
    auto start = std::chrono::steady_clock::now();
    monitor.stop();
    auto end = std::chrono::steady_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Should stop much faster than the 1000ms poll interval
    EXPECT_LT(duration.count(), 500);
    EXPECT_FALSE(monitor.isRunning());
}

TEST_F(ResourceMonitorTest, DestructorStopsMonitoring) {
    writeConfigFile(2, 4);
    
    ResourceLimits limits;
    limits.max_gpus = 2;
    limits.max_cpus = 4;
    limits.config_file = temp_file_;
    limits.poll_interval = 100ms;
    
    {
        ResourceMonitor monitor(limits);
        monitor.start([](const ResourceLimits&) {});
        EXPECT_TRUE(monitor.isRunning());
        // Destructor should stop cleanly
    }
    
    // No crash = success
}

// ============================================================================
// Thread Safety Tests
// ============================================================================

TEST_F(ResourceMonitorTest, ConcurrentGetCurrentLimits) {
    writeConfigFile(2, 4);
    
    ResourceLimits limits;
    limits.max_gpus = 2;
    limits.max_cpus = 4;
    limits.config_file = temp_file_;
    limits.poll_interval = 50ms;
    
    ResourceMonitor monitor(limits);
    monitor.start([](const ResourceLimits&) {});
    
    std::atomic<bool> stop_readers{false};
    std::vector<std::thread> readers;
    
    // Start multiple reader threads
    for (int i = 0; i < 4; ++i) {
        readers.emplace_back([&monitor, &stop_readers]() {
            while (!stop_readers) {
                auto limits = monitor.getCurrentLimits();
                (void)limits;  // Just verify no crashes
                std::this_thread::sleep_for(10ms);
            }
        });
    }
    
    // Let them read for a bit
    std::this_thread::sleep_for(200ms);
    
    stop_readers = true;
    for (auto& t : readers) {
        t.join();
    }
    
    monitor.stop();
}

TEST_F(ResourceMonitorTest, GetCurrentLimitsReturnsLatest) {
    writeConfigFile(1, 2);
    
    ResourceLimits limits;
    limits.max_gpus = 1;
    limits.max_cpus = 2;
    limits.config_file = temp_file_;
    limits.poll_interval = 50ms;
    
    ResourceMonitor monitor(limits);
    monitor.start([](const ResourceLimits&) {});
    
    std::this_thread::sleep_for(100ms);
    writeConfigFile(5, 10);
    std::this_thread::sleep_for(150ms);
    
    auto current = monitor.getCurrentLimits();
    EXPECT_EQ(current.max_gpus, 5);
    EXPECT_EQ(current.max_cpus, 10);
    
    monitor.stop();
}
