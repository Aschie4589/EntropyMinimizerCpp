#include "minimizer/algorithm/algorithm_manager.h"
#include <random>
#include <cmath>
#include <stdexcept>
#include <sstream>

// DEBUG LOGGING
#include "utilities/messaging/DEBUG_LOGGER.h"

namespace entropy {

AlgorithmManager::AlgorithmManager(
    IComputeDevice& device,
    EntropyManager& entropy_manager,
    const HostKrausOperators& kraus,
    double epsilon,
    PrecisionType precision
)
    : device_(device)
    , entropy_manager_(entropy_manager)
    , kraus_count_(kraus.kraus_count)
    , input_dim_(kraus.input_dim)
    , output_dim_(kraus.output_dim)
    , epsilon_(epsilon)
    , precision_(precision)
    , host_kraus_(kraus)
    , initialized_(false)
{
    DEBUG_LOG("Constructing AlgorithmManager", "algorithm_manager_log.txt");
    // Validate epsilon
    if (epsilon <= 0.0 || epsilon >= 1.0) {
        throw std::invalid_argument("Epsilon must be in range (0, 1)");
    }
    
    // Validate dimensions
    if (kraus_count_ <= 0) {
        throw std::invalid_argument("Kraus count must be positive");
    }
    if (input_dim_ <= 0 || output_dim_ <= 0) {
        throw std::invalid_argument("Dimensions must be positive");
    }
    
    // Validate Kraus data size
    size_t expected_size = kraus_count_ * input_dim_ * output_dim_;
    if (precision == PrecisionType::FLOAT) {
        expected_size *= sizeof(std::complex<float>);
    } else {
        expected_size *= sizeof(std::complex<double>);
    }
    
    if (kraus.data.size() != expected_size) {
        std::ostringstream oss;
        oss << "Kraus data size mismatch: expected " << expected_size 
            << " bytes, got " << kraus.data.size();
        throw std::invalid_argument(oss.str());
    }
    DEBUG_LOG("AlgorithmManager constructed successfully", "algorithm_manager_log.txt");
}

void AlgorithmManager::initialize(const std::vector<std::complex<double>>& initial_vector) {
    DEBUG_LOG("AlgorithmManager: Initializing AlgorithmManager", "algorithm_manager_log.txt");
    if (!strategy_) {
        throw std::runtime_error("Strategy must be set before initialization");
    }
    
    DEBUG_LOG("AlgorithmManager: Validating initial vector size", "algorithm_manager_log.txt");
    // Validate vector size
    if (static_cast<int>(initial_vector.size()) != input_dim_) {
        std::ostringstream oss;
        oss << "Initial vector size mismatch: expected " << input_dim_ 
            << ", got " << initial_vector.size();
        throw std::invalid_argument(oss.str());
    }
    
    DEBUG_LOG("AlgorithmManager: Allocating and uploading Kraus operators to device", "algorithm_manager_log.txt");
    // Allocate and upload Kraus operators
    size_t kraus_bytes = kraus_count_ * input_dim_ * output_dim_;
    if (precision_ == PrecisionType::FLOAT) {
        kraus_bytes *= sizeof(std::complex<float>);
    } else {
        kraus_bytes *= sizeof(std::complex<double>);
    }
    DEBUG_LOG("AlgorithmManager: Calculated Kraus operators byte size: " + std::to_string(kraus_bytes), "algorithm_manager_log.txt");

    // Debug info about the device
    DEBUG_LOG("Device Name: " + device_.getName(), "algorithm_manager_log.txt");
    DEBUG_LOG("Device Backend: " + std::to_string(static_cast<int>(device_.getBackend())), "algorithm_manager_log.txt");
    DEBUG_LOG("Device ID: " + std::to_string(device_.getDeviceID()), "algorithm_manager_log.txt");
    d_kraus_ = device_.allocate(kraus_bytes);
    DEBUG_LOG("AlgorithmManager: Allocated device memory for Kraus operators", "algorithm_manager_log.txt");
    d_kraus_->copyFromHost(host_kraus_.data.data(), kraus_bytes);
    DEBUG_LOG("AlgorithmManager: Uploaded Kraus operators to device", "algorithm_manager_log.txt");

    DEBUG_LOG("AlgorithmManager: Allocated and uploaded Kraus operators", "algorithm_manager_log.txt");
    
    DEBUG_LOG("AlgorithmManager: Allocating and uploading current vector", "algorithm_manager_log.txt");
    // Allocate and upload current vector
    size_t vector_bytes;
    if (precision_ == PrecisionType::FLOAT) {
        vector_bytes = input_dim_ * sizeof(std::complex<float>);
        // Convert double to float if needed
        std::vector<std::complex<float>> vec_float(input_dim_);
        for (int i = 0; i < input_dim_; ++i) {
            vec_float[i] = std::complex<float>(
                static_cast<float>(initial_vector[i].real()),
                static_cast<float>(initial_vector[i].imag())
            );
        }
        d_current_vector_ = device_.allocate(vector_bytes);
        d_current_vector_->copyFromHost(vec_float.data(), vector_bytes);
    } else {
        vector_bytes = input_dim_ * sizeof(std::complex<double>);
        d_current_vector_ = device_.allocate(vector_bytes);
        d_current_vector_->copyFromHost(initial_vector.data(), vector_bytes);
    }
    
    DEBUG_LOG("AlgorithmManager: Allocated and uploaded current vector", "algorithm_manager_log.txt");

    DEBUG_LOG("AlgorithmManager: Initializing strategy", "algorithm_manager_log.txt");
    // Initialize strategy with dimensions
    strategy_->initialize(kraus_count_, input_dim_, output_dim_, epsilon_, precision_);
    DEBUG_LOG("AlgorithmManager: Strategy initialized", "algorithm_manager_log.txt");
    initialized_ = true;

    DEBUG_LOG("AlgorithmManager: Initialization complete", "algorithm_manager_log.txt");
}

void AlgorithmManager::initializeRandom() {
    if (!strategy_) {
        throw std::runtime_error("Strategy must be set before initialization");
    }
    
    // Generate random normalized vector
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> dist(0.0, 1.0);
    
    std::vector<std::complex<double>> random_vec(input_dim_);
    double norm_sq = 0.0;
    
    for (int i = 0; i < input_dim_; ++i) {
        double real = dist(gen);
        double imag = dist(gen);
        random_vec[i] = std::complex<double>(real, imag);
        norm_sq += real * real + imag * imag;
    }
    
    // Normalize
    double norm = std::sqrt(norm_sq);
    for (int i = 0; i < input_dim_; ++i) {
        random_vec[i] /= norm;
    }
    
    initialize(random_vec);
}

void AlgorithmManager::stepOnce() {
    if (!initialized_) {
        throw std::runtime_error("AlgorithmManager must be initialized before stepOnce()");
    }
    
    // Validate device memory ownership
    validateDeviceMemory(d_kraus_.get());
    validateDeviceMemory(d_current_vector_.get());
    
    // Call strategy (updates d_current_vector_ in-place, returns epsilon entropy)
    double epsilon_entropy = strategy_->stepOnce(d_kraus_.get(), d_current_vector_.get());
    
    // Update entropy manager
    entropy_manager_.updateEpsilonEntropy(epsilon_entropy);
}

std::vector<std::complex<double>> AlgorithmManager::getCurrentVector() const {
    if (!initialized_) {
        throw std::runtime_error("AlgorithmManager not initialized");
    }
    
    std::vector<std::complex<double>> result(input_dim_);
    
    if (precision_ == PrecisionType::FLOAT) {
        // Copy as float, then convert to double
        std::vector<std::complex<float>> vec_float(input_dim_);
        size_t bytes = input_dim_ * sizeof(std::complex<float>);
        d_current_vector_->copyToHost(vec_float.data(), bytes);
        
        for (int i = 0; i < input_dim_; ++i) {
            result[i] = std::complex<double>(
                static_cast<double>(vec_float[i].real()),
                static_cast<double>(vec_float[i].imag())
            );
        }
    } else {
        // Copy directly as double
        size_t bytes = input_dim_ * sizeof(std::complex<double>);
        d_current_vector_->copyToHost(result.data(), bytes);
    }
    
    return result;
}

void AlgorithmManager::setCurrentVector(const std::vector<std::complex<double>>& vec) {
    if (!initialized_) {
        throw std::runtime_error("AlgorithmManager not initialized");
    }
    
    if (static_cast<int>(vec.size()) != input_dim_) {
        std::ostringstream oss;
        oss << "Vector size mismatch: expected " << input_dim_ 
            << ", got " << vec.size();
        throw std::invalid_argument(oss.str());
    }
    
    if (precision_ == PrecisionType::FLOAT) {
        // Convert double to float
        std::vector<std::complex<float>> vec_float(input_dim_);
        for (int i = 0; i < input_dim_; ++i) {
            vec_float[i] = std::complex<float>(
                static_cast<float>(vec[i].real()),
                static_cast<float>(vec[i].imag())
            );
        }
        size_t bytes = input_dim_ * sizeof(std::complex<float>);
        d_current_vector_->copyFromHost(vec_float.data(), bytes);
    } else {
        // Copy directly as double
        size_t bytes = input_dim_ * sizeof(std::complex<double>);
        d_current_vector_->copyFromHost(vec.data(), bytes);
    }
}

double AlgorithmManager::getEpsilonEntropy() const {
    return entropy_manager_.getEpsilonEntropy();
}

double AlgorithmManager::getEstimatedEntropy() const {
    return entropy_manager_.getEstimatedEntropy();
}

std::pair<double, double> AlgorithmManager::getEntropyBounds() const {
    return entropy_manager_.getEntropyBounds();
}

void AlgorithmManager::setStrategy(std::unique_ptr<IMinimizationStrategy> strategy) {
    if (!strategy) {
        throw std::invalid_argument("Strategy cannot be null");
    }
    strategy_ = std::move(strategy);
}

void AlgorithmManager::validateDeviceMemory(const IDeviceMemory* memory) const {
    if (!memory) {
        throw std::invalid_argument("Device memory pointer is null");
    }
    
    // Check that memory backend matches our device backend
    if (memory->getBackend() != device_.getBackend()) {
        std::ostringstream oss;
        oss << "Device memory backend mismatch: strategy uses "
            << static_cast<int>(memory->getBackend())
            << " but AlgorithmManager device is "
            << static_cast<int>(device_.getBackend());
        throw std::invalid_argument(oss.str());
    }
}

} // namespace entropy
