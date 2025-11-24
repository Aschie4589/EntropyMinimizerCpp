// CUDA compute device implementation

#include <compute/backends/cuda/CudaDevice.h>
#include <compute/backends/cuda/CudaSVDSolver.h>
#include <compute/backends/cuda/CudaMemory.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

namespace compute {

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
    device_scratch_ = std::make_unique<CudaScratchMemory>(device_scratch_size);
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
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id_);
    return std::string(prop.name);
}

std::unique_ptr<IDeviceMemory> CudaDevice::allocate(size_t bytes) {
    return std::make_unique<CudaMemory>(bytes);
}

std::unique_ptr<IStream> CudaDevice::createStream() {
    throw std::runtime_error("CudaDevice::createStream not yet implemented");
}

ILinearAlgebra* CudaDevice::getLinearAlgebra() {
    throw std::runtime_error("CudaDevice::getLinearAlgebra not yet implemented");
}

ISolver* CudaDevice::getSolver() {
    throw std::runtime_error("CudaDevice::getSolver not yet implemented");
}

IRandomGenerator* CudaDevice::getRandomGenerator() {
    throw std::runtime_error("CudaDevice::getRandomGenerator not yet implemented");
}

std::unique_ptr<ISVDSolver> CudaDevice::createSVDSolver(
    int m, int n,
    const SVDSpec& spec,
    PrecisionType precision
) {
    // Create new CudaSVDSolver (owns its own handles)
    return std::unique_ptr<ISVDSolver>(new CudaSVDSolver(
        this,
        m, n,
        spec,
        precision
    ));
}

void CudaDevice::convertPrecision(
    const IDeviceMemory*,
    IDeviceMemory*,
    PrecisionType,
    PrecisionType,
    size_t
) {
    throw std::runtime_error("CudaDevice::convertPrecision not yet implemented");
}

void CudaDevice::synchronize() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA synchronization failed: ") + cudaGetErrorString(err)
        );
    }
}

} // namespace compute
