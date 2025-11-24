#include "compute/backends/cpu/CpuStream.h"

void CpuStream::synchronize() {
    // CPU operations are synchronous - nothing to do
}

bool CpuStream::isComplete() const {
    // CPU operations complete immediately
    return true;
}

void* CpuStream::getNativeHandle() {
    // CPU has no native stream handle
    return nullptr;
}