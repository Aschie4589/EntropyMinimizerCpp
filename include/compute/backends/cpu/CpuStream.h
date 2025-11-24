#ifndef CPU_STREAM_H_
#define CPU_STREAM_H_

#include "compute/stream/IComputeStream.h"

class CpuStream : public IStream {
public:
    CpuStream() = default;
    ~CpuStream() override = default;
    
    void synchronize() override;
    bool isComplete() const override;
    void* getNativeHandle() override;
};

#endif // CPU_STREAM_H_