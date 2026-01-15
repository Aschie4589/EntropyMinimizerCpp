#include <gtest/gtest.h>
#include "minimizer/orchestration/worker_thread_pool.h"
#include "minimizer/orchestration/device_pool.h"
#include "minimizer/orchestration/device_registry.h"
#include "minimizer/orchestration/concurrent_queue.h"
#include "minimizer/orchestration/result_collector.h"
#include "minimizer/config/resource_config.h"
#include "minimizer/config/minimizer_config.h"
#include "minimizer/algorithm/minimization_strategy.h"
#include <thread>
#include <chrono>

using namespace entropy;
using namespace std::chrono_literals;

// ============================================================================
// Test Fixture
// ============================================================================

class WorkerThreadPoolTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup default config
        setupDefaultConfig();
        
        // Create Kraus operators (simple 2-operator channel for testing)
        kraus_count_ = 2;
        input_dim_ = 5;
        output_dim_ = 5;
        
        std::vector<std::complex<double>> kraus_data(
            kraus_count_ * input_dim_ * output_dim_,
            std::complex<double>(0.0, 0.0)
        );
        
        // K_0: Projector onto first basis state
        kraus_data[0] = 1.0;
        
        // K_1: Projector onto second basis state
        kraus_data[input_dim_ * output_dim_ + input_dim_ + 1] = 1.0;
        
        kraus_ops_ = HostKrausOperators::fromDouble(
            kraus_data, kraus_count_, input_dim_, output_dim_
        );
        
        // Create config vector
        configs_.push_back(config_);
        
        // Create device pool (CPU-only for deterministic testing)
        ResourceConfig resource_config = ResourceConfig::createCPUOnly(4);
        device_pool_ = std::make_unique<DevicePool>(resource_config);
        DeviceRegistry& registry = DeviceRegistry::instance();
        device_pool_->initializeFromRegistry(registry);
    }
    
    void setupDefaultConfig() {
        // Algorithm config
        config_.algorithm.epsilon = 0.01;
        config_.algorithm.precision = PrecisionType::DOUBLE;
        
        // Stopping config - use small iteration count for fast tests
        config_.stopping.max_iterations = 10;
        config_.stopping.convergence_window = 5;
        config_.stopping.convergence_tolerance = 1e-6;
    }
    
    HostVector createRandomVector(int dim) {
        std::vector<std::complex<double>> vec(dim);
        double norm = 0.0;
        
        for (int i = 0; i < dim; ++i) {
            vec[i] = std::complex<double>(
                static_cast<double>(rand()) / RAND_MAX,
                static_cast<double>(rand()) / RAND_MAX
            );
            norm += std::norm(vec[i]);
        }
        
        norm = std::sqrt(norm);
        for (int i = 0; i < dim; ++i) {
            vec[i] /= norm;
        }
        
        return HostVector(vec);
    }
    
    MinimizerConfig config_;
    std::vector<MinimizerConfig> configs_;
    HostKrausOperators kraus_ops_;
    int kraus_count_;
    int input_dim_;
    int output_dim_;
    std::unique_ptr<DevicePool> device_pool_;
};

// ============================================================================
// Constructor and Basic Tests
// ============================================================================

TEST_F(WorkerThreadPoolTest, Constructor_Valid) {
    ConcurrentQueue<RunTask> work_queue;
    ResultCollector result_collector;
    
    WorkerThreadPool pool(
        *device_pool_,
        work_queue,
        result_collector,
        configs_,
        kraus_ops_,
        input_dim_
    );
    
    EXPECT_EQ(pool.getWorkerCount(), 0);
    EXPECT_EQ(pool.getActiveWorkerCount(), 0);
}

TEST_F(WorkerThreadPoolTest, StartAndAddInitialWorkers) {
    ConcurrentQueue<RunTask> work_queue;
    ResultCollector result_collector;
    
    WorkerThreadPool pool(
        *device_pool_,
        work_queue,
        result_collector,
        configs_,
        kraus_ops_,
        input_dim_
    );
    
    pool.start();
    pool.addWorkers(2);
    
    // Give workers time to start
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    EXPECT_EQ(pool.getWorkerCount(), 2);
    
    pool.stop();
    pool.waitAll();
}

// ============================================================================
// Dynamic Worker Scaling Tests (Phase 3 API)
// ============================================================================

TEST_F(WorkerThreadPoolTest, AddWorkersWhileRunning) {
    ConcurrentQueue<RunTask> work_queue;
    ResultCollector result_collector;
    
    WorkerThreadPool pool(
        *device_pool_,
        work_queue,
        result_collector,
        configs_,
        kraus_ops_,
        input_dim_
    );
    
    pool.start();
    pool.addWorkers(2);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    EXPECT_EQ(pool.getWorkerCount(), 2);
    
    // Add 2 more workers dynamically
    pool.addWorkers(2);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    EXPECT_EQ(pool.getWorkerCount(), 4);
    
    pool.stop();
    pool.waitAll();
}

TEST_F(WorkerThreadPoolTest, RemoveWorkersGracefully) {
    ConcurrentQueue<RunTask> work_queue;
    ResultCollector result_collector;
    
    WorkerThreadPool pool(
        *device_pool_,
        work_queue,
        result_collector,
        configs_,
        kraus_ops_,
        input_dim_
    );
    
    pool.start();
    pool.addWorkers(4);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    EXPECT_EQ(pool.getWorkerCount(), 4);
    
    // Remove 2 workers
    pool.removeWorkers(2);
    
    // Wait for workers to finish current tasks and exit
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    EXPECT_LE(pool.getWorkerCount(), 2);
    
    pool.stop();
    pool.waitAll();
}

TEST_F(WorkerThreadPoolTest, AddWorkersIncreasesCapacity) {
    ConcurrentQueue<RunTask> work_queue;
    ResultCollector result_collector;
    
    WorkerThreadPool pool(
        *device_pool_,
        work_queue,
        result_collector,
        configs_,
        kraus_ops_,
        input_dim_
    );
    
    pool.start();
    pool.addWorkers(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    int initial_count = pool.getWorkerCount();
    
    // Add workers multiple times
    pool.addWorkers(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    pool.addWorkers(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    EXPECT_GT(pool.getWorkerCount(), initial_count);
    
    pool.stop();
    pool.waitAll();
}

// ============================================================================
// Task Processing Tests
// ============================================================================

TEST_F(WorkerThreadPoolTest, ProcessSingleTask) {
    ConcurrentQueue<RunTask> work_queue;
    ResultCollector result_collector;
    
    WorkerThreadPool pool(
        *device_pool_,
        work_queue,
        result_collector,
        configs_,
        kraus_ops_,
        input_dim_
    );
    
    // Create a simple task
    RunTask task;
    task.run_id = 0;
    task.config_id = 0;
    task.initial_vector = createRandomVector(input_dim_);
    
    work_queue.push(std::move(task));
    work_queue.signalDone();
    
    pool.start();
    pool.addWorkers(1);
    
    // Wait for task completion
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    pool.stop();
    pool.waitAll();
    
    // Check result was collected
    EXPECT_GT(result_collector.numCompleted() + result_collector.numErrors(), 0);
}

TEST_F(WorkerThreadPoolTest, ProcessMultipleTasks) {
    ConcurrentQueue<RunTask> work_queue;
    ResultCollector result_collector;
    
    WorkerThreadPool pool(
        *device_pool_,
        work_queue,
        result_collector,
        configs_,
        kraus_ops_,
        input_dim_
    );
    
    // Create 10 tasks
    int num_tasks = 10;
    for (int i = 0; i < num_tasks; ++i) {
        RunTask task;
        task.run_id = i;
        task.config_id = 0;
        task.initial_vector = createRandomVector(input_dim_);
        work_queue.push(std::move(task));
    }
    work_queue.signalDone();
    
    pool.start();
    pool.addWorkers(2);
    
    // Wait for all tasks to complete
    while (result_collector.numCompleted() + result_collector.numErrors() < static_cast<size_t>(num_tasks)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    pool.stop();
    pool.waitAll();
    
    EXPECT_EQ(result_collector.numCompleted() + result_collector.numErrors(), num_tasks);
}

TEST_F(WorkerThreadPoolTest, ConcurrentWorkersDistributeTasks) {
    ConcurrentQueue<RunTask> work_queue;
    ResultCollector result_collector;
    
    WorkerThreadPool pool(
        *device_pool_,
        work_queue,
        result_collector,
        configs_,
        kraus_ops_,
        input_dim_
    );
    
    // Create 20 tasks
    int num_tasks = 20;
    for (int i = 0; i < num_tasks; ++i) {
        RunTask task;
        task.run_id = i;
        task.config_id = 0;
        task.initial_vector = createRandomVector(input_dim_);
        work_queue.push(std::move(task));
    }
    work_queue.signalDone();
    
    // Start with 4 workers
    pool.start();
    pool.addWorkers(4);
    
    // Wait for completion
    while (result_collector.numCompleted() + result_collector.numErrors() < static_cast<size_t>(num_tasks)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    pool.stop();
    pool.waitAll();
    
    EXPECT_EQ(result_collector.numCompleted() + result_collector.numErrors(), num_tasks);
}

// ============================================================================
// Worker State Query Tests
// ============================================================================

TEST_F(WorkerThreadPoolTest, GetWorkerCount) {
    ConcurrentQueue<RunTask> work_queue;
    ResultCollector result_collector;
    
    WorkerThreadPool pool(
        *device_pool_,
        work_queue,
        result_collector,
        configs_,
        kraus_ops_,
        input_dim_
    );
    
    EXPECT_EQ(pool.getWorkerCount(), 0);
    
    pool.start();
    pool.addWorkers(3);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    EXPECT_EQ(pool.getWorkerCount(), 3);
    
    pool.addWorkers(2);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    EXPECT_EQ(pool.getWorkerCount(), 5);
    
    pool.stop();
    pool.waitAll();
}

TEST_F(WorkerThreadPoolTest, GetActiveWorkerCount) {
    ConcurrentQueue<RunTask> work_queue;
    ResultCollector result_collector;
    
    WorkerThreadPool pool(
        *device_pool_,
        work_queue,
        result_collector,
        configs_,
        kraus_ops_,
        input_dim_
    );
    
    pool.start();
    pool.addWorkers(2);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Initially no active workers (waiting for tasks)
    int active = pool.getActiveWorkerCount();
    EXPECT_GE(active, 0);
    EXPECT_LE(active, 2);
    
    pool.stop();
    pool.waitAll();
}

// ============================================================================
// Lifecycle Tests
// ============================================================================

TEST_F(WorkerThreadPoolTest, StopAndWaitAll) {
    ConcurrentQueue<RunTask> work_queue;
    ResultCollector result_collector;
    
    WorkerThreadPool pool(
        *device_pool_,
        work_queue,
        result_collector,
        configs_,
        kraus_ops_,
        input_dim_
    );
    
    pool.start();
    pool.addWorkers(3);
    
    // Add some tasks
    for (int i = 0; i < 5; ++i) {
        RunTask task;
        task.run_id = i;
        task.config_id = 0;
        task.initial_vector = createRandomVector(input_dim_);
        work_queue.push(std::move(task));
    }
    
    // Stop gracefully
    pool.stop();
    pool.waitAll();
    
    // Should complete without hanging
    SUCCEED();
}

TEST_F(WorkerThreadPoolTest, MultipleStartStopCycles) {
    ConcurrentQueue<RunTask> work_queue;
    ResultCollector result_collector;
    
    WorkerThreadPool pool(
        *device_pool_,
        work_queue,
        result_collector,
        configs_,
        kraus_ops_,
        input_dim_
    );
    
    // Cycle 1
    pool.start();
    pool.addWorkers(2);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    pool.stop();
    pool.waitAll();
    
    // Cycle 2
    pool.start();
    pool.addWorkers(3);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    pool.stop();
    pool.waitAll();
    
    SUCCEED();
}
