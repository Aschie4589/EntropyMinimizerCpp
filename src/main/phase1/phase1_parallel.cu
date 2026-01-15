#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <chrono>
#include <filesystem>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>

#include "channel/generator/random_generator.h"
#include "utilities/messaging/message_handler.h"
#include "utilities/messaging/printer.h"
#include "utilities/gpu/gpu_resource_manager.h"
#include "yaml-cpp/yaml.h"
#include "minimizer/analysis/tensor_entropy.h"

/*
    Parallel Phase 1 - Multi-threaded channel generation and analysis
    
    Architecture:
    - Task queue pattern with thread pool
    - GPU resource management for dynamic allocation
    - Three stages: generation → entropy estimation → (future) minimization
    - Safe concurrent file I/O (each channel has unique directory)
*/

#define PHASE1_CONFIG_FILE "/home/tommasoa/EntropyMinimizerCpp/configs/phase1.yml"

// ============================================================================
// Data Structures
// ============================================================================

struct ChannelTask {
    int N;                          // Hilbert space dimension
    int d;                          // Number of Kraus operators
    int channel_index;              // Channel ID within (N,d) pair
    std::string unique_id;          // Unique identifier
    std::string save_folder;        // Output directory
    double eps;                     // Epsilon for entropy estimation
    double tensor_eps;              // Epsilon for tensor entropy
    
    // Task state
    enum Stage { GENERATION, ENTROPY, MINIMIZATION, COMPLETE };
    std::atomic<Stage> current_stage{GENERATION};
    
    // Results storage
    std::vector<std::complex<double>> kraus_operators;
    double estimated_entropy = 0.0;
    double estimated_error = 0.0;
    
    ChannelTask(int n, int d_val, int ch_idx, const std::string& out_folder, 
                double e, double te) 
        : N(n), d(d_val), channel_index(ch_idx), eps(e), tensor_eps(te)
    {
        // Generate unique ID
        auto t = std::time(nullptr);
        auto tm = *std::localtime(&t);
        std::ostringstream oss;
        oss << "kraus_N_" << N << "_d_" << d << "_ch_" << channel_index << "_"
            << std::put_time(&tm, "%Y_%m_%d-%H_%M_%S") << "_"
            << std::this_thread::get_id();
        unique_id = oss.str();
        
        save_folder = out_folder + "/" + unique_id;
        
        // Pre-allocate Kraus operators
        kraus_operators.resize(d * N * N);
    }
};

// ============================================================================
// Thread-Safe Task Queue
// ============================================================================

template<typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_; // Mutual exclusive, i.e. the lock: gets locked before critical sections and unlocked after so that you don't have concurrent access
    std::condition_variable cond_var_; // Condition variable; used to block threads until notified
    std::atomic<bool> shutdown_{false};

public:
    void push(T item) {
        {
            std::lock_guard<std::mutex> lock(mutex_); // This is similar to just locking and then unlocking: the mutex_ is locked, then undlocked when lock goes out of scope
            queue_.push(std::move(item)); // std::move to avoid copying
        }
        cond_var_.notify_one(); // Wake up one thread that is waiting on the condition variable
    }
    
    bool pop(T& item, int timeout_ms = -1) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (timeout_ms < 0) {
            // Wait indefinitely
            cond_var_.wait(lock, [this]{ 
                return !queue_.empty() || shutdown_.load(); 
            });
        } else {
            // Wait with timeout
            if (!cond_var_.wait_for(lock, std::chrono::milliseconds(timeout_ms), // This is this part: wait for the condition variable to signal an available task; if that does not happen within timeout_ms milliseconds, return false
                [this]{ return !queue_.empty() || shutdown_.load(); })) {
                return false; // Timeout
            }
        }
        
        if (shutdown_.load() && queue_.empty()) {
            return false;
        }
        
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
    void shutdown() {
        shutdown_.store(true);
        cond_var_.notify_all();
    }
    
    bool is_shutdown() const {
        return shutdown_.load();
    }
};

// ============================================================================
// Stage Functions
// ============================================================================

void stage_generate_channel(std::shared_ptr<ChannelTask> task, 
                            GPUResourceManager& gpu_mgr,
                            utils::MessageHandler* msg_handler) 
{
    auto start = std::chrono::high_resolution_clock::now();
    
    // Acquire GPU
    int gpu_id = gpu_mgr.waitForGPU(300); // 5 min timeout
    if (gpu_id == -1) {
        std::cerr << "Failed to acquire GPU for channel " << task->unique_id << std::endl;
        return;
    }
    
    cudaSetDevice(gpu_id);
    
    std::cout << "[GPU " << gpu_id << "] Generating channel " << task->unique_id << std::endl;
    
    // Create directory
    std::filesystem::create_directories(task->save_folder);
    
    // Configure generator
    channel::RandomGeneratorConfig rg_config;
    rg_config.kraus_number = task->d;
    rg_config.kraus_in_dimension = task->N;
    rg_config.kraus_out_dimension = task->N;
    
    // Generate Kraus operators on GPU
    channel::RandomGenerator rg(rg_config, *msg_handler);
    rg.generate(&task->kraus_operators);
    
    // Release GPU early (file I/O doesn't need it)
    gpu_mgr.releaseGPU(gpu_id);
    
    // Save to file (CPU-bound I/O, GPU released)
    std::string kraus_filename = task->save_folder + "/kraus.dat";
    std::ofstream kraus_file(kraus_filename, std::ios::binary);
    if (kraus_file.is_open()) {
        int kraus_size = task->d * task->N * task->N;
        kraus_file.write(reinterpret_cast<const char*>(task->kraus_operators.data()), 
                        sizeof(std::complex<double>) * kraus_size);
        kraus_file.close();
    } else {
        std::cerr << "Error: Failed to write Kraus operators for " << task->unique_id << std::endl;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "[DONE] Generated " << task->unique_id << " in " 
              << duration.count() << " ms" << std::endl;
    
    task->current_stage.store(ChannelTask::ENTROPY);
}

void stage_compute_entropy(std::shared_ptr<ChannelTask> task,
                          GPUResourceManager& gpu_mgr,
                          utils::MessageHandler* msg_handler)
{
    auto start = std::chrono::high_resolution_clock::now();
    
    // Acquire GPU
    int gpu_id = gpu_mgr.waitForGPU(300);
    if (gpu_id == -1) {
        std::cerr << "Failed to acquire GPU for entropy computation: " << task->unique_id << std::endl;
        return;
    }
    
    cudaSetDevice(gpu_id);
    
    std::cout << "[GPU " << gpu_id << "] Computing tensor entropy for " << task->unique_id << std::endl;
    
    // Create estimator (could be cached per-thread if expensive)
    TensorEntropyEstimator<double> tensor_estimator;
    
    // Compute entropy
    tensor_estimator.computeEntropy(
        reinterpret_cast<cuDoubleComplex*>(task->kraus_operators.data()),
        task->d, task->N, task->N, task->tensor_eps
    );
    
    task->estimated_entropy = tensor_estimator.getEstimatedEntropy();
    task->estimated_error = tensor_estimator.getEstimatedError();
    
    gpu_mgr.releaseGPU(gpu_id);
    
    // Save results to YAML (CPU-bound I/O)
    std::string yaml_filename = task->save_folder + "/info.yml";
    YAML::Node info_node;
    info_node["channel"]["N"] = task->N;
    info_node["channel"]["d"] = task->d;
    info_node["channel"]["kraus_number"] = task->d;
    info_node["channel"]["kraus_in_dimension"] = task->N;
    info_node["channel"]["kraus_out_dimension"] = task->N;
    info_node["run"]["channel_index"] = task->channel_index;
    info_node["run"]["unique_string"] = task->unique_id;
    info_node["analysis"]["tensor_entropy_estimation"]["eps"] = task->tensor_eps;
    info_node["analysis"]["tensor_entropy_estimation"]["estimated_entropy"] = task->estimated_entropy;
    info_node["analysis"]["tensor_entropy_estimation"]["estimated_error"] = task->estimated_error;
    
    double theoretical_entropy = std::log(static_cast<double>(task->d)) * 
                                 (2.0 - 1.0 / static_cast<double>(task->d));
    info_node["analysis"]["theoretical_entropy"] = theoretical_entropy;
    
    std::ofstream yaml_file(yaml_filename);
    if (yaml_file.is_open()) {
        yaml_file << info_node;
        yaml_file.close();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    std::cout << "[DONE] Entropy for " << task->unique_id << ": " 
              << task->estimated_entropy << " +/- " << task->estimated_error
              << " (theoretical: " << theoretical_entropy << ") in "
              << duration.count() << " ms" << std::endl;
    
    task->current_stage.store(ChannelTask::COMPLETE);
}

// ============================================================================
// Worker Thread
// ============================================================================

void worker_thread(int worker_id, 
                  ThreadSafeQueue<std::shared_ptr<ChannelTask>>& gen_queue,
                  ThreadSafeQueue<std::shared_ptr<ChannelTask>>& entropy_queue,
                  GPUResourceManager& gpu_mgr,
                  utils::MessageHandler* msg_handler,
                  std::atomic<int>& tasks_completed)
{
    std::cout << "Worker " << worker_id << " started" << std::endl;
    
    while (true) {
        std::shared_ptr<ChannelTask> task;
        
        // Try to get a generation task first
        if (gen_queue.pop(task, 100)) {
            stage_generate_channel(task, gpu_mgr, msg_handler);
            entropy_queue.push(task);
            continue;
        }
        
        // Try entropy task
        if (entropy_queue.pop(task, 100)) {
            stage_compute_entropy(task, gpu_mgr, msg_handler);
            tasks_completed.fetch_add(1);
            continue;
        }
        
        // Check if we should exit
        if (gen_queue.is_shutdown() && entropy_queue.is_shutdown()) {
            if (gen_queue.size() == 0 && entropy_queue.size() == 0) {
                break;
            }
        }
    }
    
    std::cout << "Worker " << worker_id << " exiting" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Determine config file
    std::string config_file = PHASE1_CONFIG_FILE;
    if (argc > 1) {
        config_file = argv[1];
        std::cout << "Using config file: " << config_file << std::endl;
    } else {
        std::cout << "Using default config file: " << config_file << std::endl;
    }
    
    // Load configuration
    YAML::Node config = YAML::LoadFile(config_file);
    
    std::vector<int> channel_ns = config["config"]["params"]["N"].as<std::vector<int>>();
    std::vector<int> channel_ds = config["config"]["params"]["d"].as<std::vector<int>>();
    std::string out_folder = config["config"]["output"]["output-directory"].as<std::string>();
    int num_channels = config["config"]["params"]["channel-gen"]["num-channels"].as<int>();
    double eps = config["config"]["params"]["eps"].as<double>();
    double tensor_eps = config["config"]["params"]["moe-tensor-estimation"]["eps"].as<double>();
    
    // Get valid (N,d) pairs
    std::vector<std::pair<int,int>> nd_pairs;
    for (int n : channel_ns) {
        for (int d : channel_ds) {
            if (d*d <= n) {
                nd_pairs.push_back(std::make_pair(n, d));
            }
        }
    }
    
    std::cout << "Phase 1 Parallel Execution" << std::endl;
    std::cout << "Valid (N,d) pairs: " << nd_pairs.size() << std::endl;
    std::cout << "Channels per pair: " << num_channels << std::endl;
    std::cout << "Total channels: " << (nd_pairs.size() * num_channels) << std::endl;
    
    // Initialize GPU resource manager
    GPUResourceManager gpu_mgr("../configs/gpu_config.txt", 5ULL * 1024 * 1024 * 1024, 10);
    int num_available_gpus = gpu_mgr.getAvailableGPUCount();
    std::cout << "\n" << gpu_mgr.getStatusString() << std::endl;
    
    // Determine number of worker threads (typically num_gpus * 2 for overlap)
    int num_workers = std::max(1, num_available_gpus * 2);
    std::cout << "Spawning " << num_workers << " worker threads" << std::endl;
    
    // Create message handler
    auto msg_handler = std::make_unique<utils::MessageHandler>();
    msg_handler->addSink(std::make_unique<utils::Printer>(0, true));
    
    // Create task queues
    ThreadSafeQueue<std::shared_ptr<ChannelTask>> generation_queue;
    ThreadSafeQueue<std::shared_ptr<ChannelTask>> entropy_queue;
    
    // Enqueue all tasks
    int total_tasks = 0;
    for (const auto& nd : nd_pairs) {
        for (int ch = 0; ch < num_channels; ch++) {
            auto task = std::make_shared<ChannelTask>(
                nd.first, nd.second, ch, out_folder, eps, tensor_eps
            );
            generation_queue.push(task);
            total_tasks++;
        }
    }
    
    std::cout << "\nEnqueued " << total_tasks << " channel generation tasks" << std::endl;
    
    // Launch worker threads
    std::vector<std::thread> workers;
    std::atomic<int> tasks_completed{0};
    
    for (int i = 0; i < num_workers; i++) {
        workers.emplace_back(worker_thread, i, 
                           std::ref(generation_queue),
                           std::ref(entropy_queue),
                           std::ref(gpu_mgr),
                           msg_handler.get(),
                           std::ref(tasks_completed));
    }
    
    // Monitor progress
    std::cout << "\n=== Processing started ===" << std::endl;
    while (tasks_completed.load() < total_tasks) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        std::cout << "Progress: " << tasks_completed.load() << "/" << total_tasks 
                  << " channels completed" << std::endl;
    }
    
    // Signal shutdown
    generation_queue.shutdown();
    entropy_queue.shutdown();
    
    // Wait for all workers
    for (auto& worker : workers) {
        worker.join();
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);
    
    std::cout << "\n=== Phase 1 Complete ===" << std::endl;
    std::cout << "Total time: " << total_duration.count() << " seconds" << std::endl;
    std::cout << "Average time per channel: " 
              << (total_duration.count() / static_cast<double>(total_tasks)) 
              << " seconds" << std::endl;
    std::cout << "\nFinal GPU status:" << std::endl;
    std::cout << gpu_mgr.getStatusString() << std::endl;
    
    return 0;
}
