// CUDA compute device implementation
// Provides CUDA backend for compute operations

#ifndef CUDA_DEVICE_H_
#define CUDA_DEVICE_H_

#include <compute/device/IComputeDevice.h>
#include <compute/memory/CudaScratchMemory.h>
#include <compute/memory/CpuScratchMemory.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <memory>

namespace compute {

/**
 * @brief CUDA implementation of compute device
 * 
 * Manages CUDA resources including:
 * - cuSOLVER and cuBLAS handles
 * - Device and host scratch memory pools
 * - Solver and SVD solver creation
 */
class CudaDevice : public IComputeDevice {
public:
    /**
     * @brief Construct CUDA device
     * @param device_id CUDA device ID (default: 0)
     * @param device_scratch_size Initial device scratch pool size in bytes (default: 64MB)
     * @param host_scratch_size Initial host scratch pool size in bytes (default: 16MB)
     * 
     * Note: Does NOT create cuSOLVER/cuBLAS handles. Those are owned by individual solvers.
     */
    explicit CudaDevice(
        int device_id = 0,
        size_t device_scratch_size = 64 * 1024 * 1024,
        size_t host_scratch_size = 16 * 1024 * 1024
    );

    ~CudaDevice() override;

    // Prevent copying
    CudaDevice(const CudaDevice&) = delete;
    CudaDevice& operator=(const CudaDevice&) = delete;

    // Allow moving
    CudaDevice(CudaDevice&&) noexcept;
    CudaDevice& operator=(CudaDevice&&) noexcept;

    // IComputeDevice interface
    DeviceBackend getBackend() const override { return DeviceBackend::CUDA; }
    int getDeviceID() const override { return device_id_; }
    std::string getName() const override;
    
    std::unique_ptr<IDeviceMemory> allocate(size_t bytes) override;
    void synchronize() override;
    
    std::unique_ptr<IStream> createStream() override;
    
    ILinearAlgebra* getLinearAlgebra() override;
    ISolver* getSolver() override;
    IRandomGenerator* getRandomGenerator() override;
    
    IScratchMemory* getDeviceScratch() override { return device_scratch_.get(); }
    IScratchMemory* getHostScratch() override { return host_scratch_.get(); }
    
    std::unique_ptr<ISVDSolver> createSVDSolver(
        int m, int n,
        const SVDSpec& spec,
        PrecisionType precision
    ) override;
    
    void convertPrecision(
        const IDeviceMemory* src,
        IDeviceMemory* dst,
        PrecisionType src_type,
        PrecisionType dst_type,
        size_t count
    ) override;

    /**
     * @brief Get CUDA device ID
     */
    int getDeviceId() const { return device_id_; }

private:
    int device_id_;
    
    std::unique_ptr<CudaScratchMemory> device_scratch_;
    std::unique_ptr<CpuScratchMemory> host_scratch_;
    
    // Note: solver_ and linalg_ would be implemented in full version
    // For now, returning nullptr from createSolver/getLinearAlgebra
};

} // namespace compute

#endif // CUDA_DEVICE_H_
