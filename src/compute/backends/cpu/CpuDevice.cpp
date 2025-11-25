#include "compute/backends/cpu/CpuDevice.h"
#include "compute/backends/cpu/CpuSVDSolver.h"
#include "compute/backends/cpu/CpuMemory.h"
#include "compute/backends/cpu/CpuLinearAlgebra.h"
#include "compute/backends/cpu/CpuSolver.h"
#include "compute/backends/cpu/CpuRandom.h"
#include "compute/backends/cpu/CpuStream.h"
#include <stdexcept>
#include <sstream>
#include <iostream>
#include <complex>

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
    // CPU streams are no-op but we return a valid instance for consistency
    return std::make_unique<CpuStream>();
}

ILinearAlgebra* CpuDevice::getLinearAlgebra() {
    if (!linalg_) {
        linalg_ = std::make_unique<CpuLinearAlgebra>();
    }
    return linalg_.get();
}

ISolver* CpuDevice::getSolver() {
    if (!solver_) {
        solver_ = std::make_unique<CpuSolver>();
    }
    return solver_.get();
}

IRandomGenerator* CpuDevice::getRandomGenerator() {
    if (!random_) {
        random_ = std::make_unique<CpuRandom>();
    }
    return random_.get();
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
    if (src_type == dst_type) {
        // No conversion needed, just copy
        dst->copyFrom(src, count * (src_type == PrecisionType::DOUBLE ? sizeof(std::complex<double>) : sizeof(std::complex<float>)));
        return;
    }
    
    const void* src_ptr = src->data();
    void* dst_ptr = dst->data();
    
    if (src_type == PrecisionType::DOUBLE && dst_type == PrecisionType::FLOAT) {
        // Convert double -> float
        const auto* s = static_cast<const std::complex<double>*>(src_ptr);
        auto* d = static_cast<std::complex<float>*>(dst_ptr);
        for (size_t i = 0; i < count; ++i) {
            d[i] = std::complex<float>(
                static_cast<float>(s[i].real()),
                static_cast<float>(s[i].imag())
            );
        }
    } else if (src_type == PrecisionType::FLOAT && dst_type == PrecisionType::DOUBLE) {
        // Convert float -> double
        const auto* s = static_cast<const std::complex<float>*>(src_ptr);
        auto* d = static_cast<std::complex<double>*>(dst_ptr);
        for (size_t i = 0; i < count; ++i) {
            d[i] = std::complex<double>(
                static_cast<double>(s[i].real()),
                static_cast<double>(s[i].imag())
            );
        }
    } else {
        throw std::invalid_argument("Invalid precision conversion types");
    }
}

} // namespace compute
