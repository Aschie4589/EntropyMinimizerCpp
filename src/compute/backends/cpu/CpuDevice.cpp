#include "compute/backends/cpu/CpuDevice.h"
#include "compute/backends/cpu/CpuSVDSolver.h"
#include "compute/backends/cpu/CpuMemory.h"
#include <stdexcept>
#include <sstream>
#include <iostream>

namespace compute {

CpuDevice::CpuDevice(int device_id, size_t scratch_size)
    : device_id_(device_id)
    , scratch_(std::make_unique<CpuScratchMemory>(scratch_size))
{
    std::cout << "CpuDevice initialized (device " << device_id_ << ")" << std::endl;
    std::cout << "  Scratch: " << (scratch_size / (1024 * 1024)) << " MB" << std::endl;
}

CpuDevice::~CpuDevice() = default;

CpuDevice::CpuDevice(CpuDevice&&) noexcept = default;
CpuDevice& CpuDevice::operator=(CpuDevice&&) noexcept = default;

std::string CpuDevice::getName() const {
    std::ostringstream oss;
    oss << "CPU Device " << device_id_;
    return oss.str();
}

std::unique_ptr<IDeviceMemory> CpuDevice::allocate(size_t bytes) {
    return std::make_unique<CpuMemory>(bytes);
}

void CpuDevice::synchronize() {
    // CPU operations are synchronous by default
}

std::unique_ptr<IStream> CpuDevice::createStream() {
    // CPU doesn't support asynchronous streams in this implementation
    return nullptr;
}

ILinearAlgebra* CpuDevice::getLinearAlgebra() {
    // TODO: Implement CpuLinearAlgebra
    return nullptr;
}

ISolver* CpuDevice::getSolver() {
    // TODO: Implement CpuSolver
    return nullptr;
}

IRandomGenerator* CpuDevice::getRandomGenerator() {
    // TODO: Implement CpuRandomGenerator
    return nullptr;
}

std::unique_ptr<ISVDSolver> CpuDevice::createSVDSolver(
    int m, int n,
    const SVDSpec& spec,
    PrecisionType precision
) {
    return std::make_unique<CpuSVDSolver>(this, m, n, spec, precision);
}

void CpuDevice::convertPrecision(
    const IDeviceMemory* src,
    IDeviceMemory* dst,
    PrecisionType src_type,
    PrecisionType dst_type,
    size_t count
) {
    // TODO: Implement precision conversion for CPU
    throw std::runtime_error("CPU precision conversion not yet implemented");
}

} // namespace compute
