#ifndef ICOMPUTESTREAM_H_
#define ICOMPUTESTREAM_H_

/*
Interface for compute streams
*/
class IStream {
public:
    virtual ~IStream() = default;
    
    // Stream control
    virtual void synchronize() = 0;
    virtual bool isComplete() const = 0;
    
    // Get native handle (for low-level ops)
    virtual void* getNativeHandle() = 0;
    
    // Non-copyable (move-only)
    IStream(const IStream&) = delete;
    IStream& operator=(const IStream&) = delete;
    
protected:
    IStream() = default;
};

#endif // ICOMPUTESTREAM_H_