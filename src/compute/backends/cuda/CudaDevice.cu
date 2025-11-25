// CUDA compute device implementation

#include <compute/backends/cuda/CudaDevice.h>
#include <compute/backends/cuda/CudaSVDSolver.h>
#include <compute/backends/cuda/CudaMemory.h>
#include <compute/backends/cuda/CudaLinearAlgebra.h>
#include <compute/backends/cuda/CudaSolver.h>
#include <compute/backends/cuda/CudaRandom.h>
#include <compute/backends/cuda/CudaStream.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <complex>

namespace compute {

// ============================================================================
// DeviceGuard Implementation
// ============================================================================

CudaDevice::DeviceGuard::DeviceGuard(int device_id)
    : previous_device_(-1), need_restore_(false) {
    // Get current device
    cudaError_t err = cudaGetDevice(&previous_device_);
    if (err != cudaSuccess) {
        // If we can't get current device, don't try to restore
        need_restore_ = false;
        previous_device_ = -1;
    } else {
        need_restore_ = true;
    }
    
    // Set to target device
    if (previous_device_ != device_id) {
        err = cudaSetDevice(device_id);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("DeviceGuard: Failed to set device ") + 
                std::to_string(device_id) + ": " + cudaGetErrorString(err)
            );
        }
    }
}

CudaDevice::DeviceGuard::~DeviceGuard() {
    // Restore previous device (don't throw in destructor)
    if (need_restore_ && previous_device_ >= 0) {
        cudaSetDevice(previous_device_);
    }
}

// ============================================================================
// Precision Conversion Kernels
// ============================================================================

// CUDA kernel for precision conversion: double -> float
__global__ void convertDoubleToFloatKernel(
    const cuDoubleComplex* src,
    cuFloatComplex* dst,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = make_cuFloatComplex(
            static_cast<float>(cuCreal(src[idx])),
            static_cast<float>(cuCimag(src[idx]))
        );
    }
}

// CUDA kernel for precision conversion: float -> double
__global__ void convertFloatToDoubleKernel(
    const cuFloatComplex* src,
    cuDoubleComplex* dst,
    size_t count
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        dst[idx] = make_cuDoubleComplex(
            static_cast<double>(cuCrealf(src[idx])),
            static_cast<double>(cuCimagf(src[idx]))
        );
    }
}

CudaDevice::CudaDevice(int device_id, size_t device_scratch_size, size_t host_scratch_size)
    : device_id_(device_id)
{
    // Set CUDA device
    cudaError_t cuda_err = cudaSetDevice(device_id_);
    if (cuda_err != cudaSuccess) {
        throw std::runtime_error(
            std::string("Failed to set CUDA device ") + std::to_string(device_id_) +
            ": " + cudaGetErrorString(cuda_err)
        );
    }

    // Create scratch memory pools
    device_scratch_ = std::make_unique<CudaScratchMemory>(device_scratch_size, device_id_);
    host_scratch_ = std::make_unique<CpuScratchMemory>(host_scratch_size);

    std::cout << "CudaDevice initialized (device " << device_id_ << ")" << std::endl;
    std::cout << "  Device scratch: " << (device_scratch_size / 1024 / 1024) << " MB" << std::endl;
    std::cout << "  Host scratch: " << (host_scratch_size / 1024 / 1024) << " MB" << std::endl;
}

CudaDevice::~CudaDevice() {
    // Scratch pools cleaned up automatically via unique_ptr
}

CudaDevice::CudaDevice(CudaDevice&& other) noexcept
    : device_id_(other.device_id_)
    , device_scratch_(std::move(other.device_scratch_))
    , host_scratch_(std::move(other.host_scratch_))
{
}

CudaDevice& CudaDevice::operator=(CudaDevice&& other) noexcept {
    if (this != &other) {
        // Move resources
        device_id_ = other.device_id_;
        device_scratch_ = std::move(other.device_scratch_);
        host_scratch_ = std::move(other.host_scratch_);
    }
    return *this;
}

std::string CudaDevice::getName() const {
    DeviceGuard guard(device_id_);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id_);
    return std::string(prop.name);
}

std::unique_ptr<IDeviceMemory> CudaDevice::allocate(size_t bytes) {
    DeviceGuard guard(device_id_);
    return std::make_unique<CudaMemory>(bytes, device_id_);
}

std::unique_ptr<IStream> CudaDevice::createStream() {
    DeviceGuard guard(device_id_);
    return std::make_unique<CudaStream>(device_id_);
}

ILinearAlgebra* CudaDevice::getLinearAlgebra() {
    DeviceGuard guard(device_id_);
    
    if (!linalg_) {
        linalg_ = std::make_unique<CudaLinearAlgebra>(device_id_);
    }
    return linalg_.get();
}

ISolver* CudaDevice::getSolver() {
    DeviceGuard guard(device_id_);
    
    if (!solver_) {
        solver_ = std::make_unique<CudaSolver>(device_id_);
    }
    return solver_.get();
}

IRandomGenerator* CudaDevice::getRandomGenerator() {
    DeviceGuard guard(device_id_);
    
    if (!random_) {
        random_ = std::make_unique<CudaRandom>(device_id_);
    }
    return random_.get();
}

std::unique_ptr<ISVDSolver> CudaDevice::createSVDSolver(
    int m, int n,
    const SVDSpec& spec,
    PrecisionType precision
) {
    DeviceGuard guard(device_id_);
    
    // Create new CudaSVDSolver (owns its own handles)
    return std::unique_ptr<ISVDSolver>(new CudaSVDSolver(
        this,
        m, n,
        spec,
        precision
    ));
}

void CudaDevice::convertPrecision(
    const IDeviceMemory* src,
    IDeviceMemory* dst,
    PrecisionType src_type,
    PrecisionType dst_type,
    size_t count
) {
    DeviceGuard guard(device_id_);
    
    if (src_type == dst_type) {
        // No conversion needed, just copy
        dst->copyFrom(src, count * (src_type == PrecisionType::DOUBLE ? sizeof(cuDoubleComplex) : sizeof(cuFloatComplex)));
        return;
    }
    
    const void* src_ptr = src->data();
    void* dst_ptr = dst->data();
    
    // Launch kernel with 256 threads per block
    const int blockSize = 256;
    const int numBlocks = (count + blockSize - 1) / blockSize;
    
    if (src_type == PrecisionType::DOUBLE && dst_type == PrecisionType::FLOAT) {
        // Convert double -> float
        convertDoubleToFloatKernel<<<numBlocks, blockSize>>>(
            static_cast<const cuDoubleComplex*>(src_ptr),
            static_cast<cuFloatComplex*>(dst_ptr),
            count
        );
    } else if (src_type == PrecisionType::FLOAT && dst_type == PrecisionType::DOUBLE) {
        // Convert float -> double
        convertFloatToDoubleKernel<<<numBlocks, blockSize>>>(
            static_cast<const cuFloatComplex*>(src_ptr),
            static_cast<cuDoubleComplex*>(dst_ptr),
            count
        );
    } else {
        throw std::invalid_argument("Invalid precision conversion types");
    }
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA precision conversion kernel failed: ") + cudaGetErrorString(err)
        );
    }
}

void CudaDevice::synchronize() {
    DeviceGuard guard(device_id_);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA synchronization failed: ") + cudaGetErrorString(err)
        );
    }
}

} // namespace compute
