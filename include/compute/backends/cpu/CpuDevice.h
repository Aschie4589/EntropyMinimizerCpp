// CPU compute device implementation
// Provides CPU backend for compute operations

#ifndef CPU_DEVICE_H_
#define CPU_DEVICE_H_

#include <compute/device/IComputeDevice.h>
#include <compute/memory/CpuScratchMemory.h>
#include <memory>

namespace compute {

/**
 * @brief CPU implementation of compute device
 * 
 * Manages CPU resources including:
 * - Host scratch memory pool
 * - Solver and SVD solver creation using LAPACK
 */
class CpuDevice : public IComputeDevice {
public:
    /**
     * @brief Construct CPU device
     * @param device_id Logical device ID (default: 0, typically ignored for CPU)
     * @param scratch_size Initial scratch pool size in bytes (default: 64MB)
     */
    explicit CpuDevice(
        int device_id = 0,
        size_t scratch_size = 64 * 1024 * 1024
    );

    ~CpuDevice() override;

    // Prevent copying
    CpuDevice(const CpuDevice&) = delete;
    CpuDevice& operator=(const CpuDevice&) = delete;

    // Allow moving
    CpuDevice(CpuDevice&&) noexcept;
    CpuDevice& operator=(CpuDevice&&) noexcept;

    // IComputeDevice interface
    DeviceBackend getBackend() const override { return DeviceBackend::CPU; }
    int getDeviceID() const override { return device_id_; }
    std::string getName() const override;
    
    std::unique_ptr<IDeviceMemory> allocate(size_t bytes) override;
    void synchronize() override;
    
    std::unique_ptr<IStream> createStream() override;
    
    ILinearAlgebra* getLinearAlgebra() override;
    ISolver* getSolver() override;
    IRandomGenerator* getRandomGenerator() override;
    
    IScratchMemory* getDeviceScratch() override { return scratch_.get(); }
    IScratchMemory* getHostScratch() override { return scratch_.get(); }
    
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

private:
    int device_id_;
    std::unique_ptr<CpuScratchMemory> scratch_;
};

} // namespace compute

#endif // CPU_DEVICE_H_
