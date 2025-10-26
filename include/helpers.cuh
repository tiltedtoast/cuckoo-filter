#pragma once

#include <cstdio>
#include <iostream>

constexpr bool powerOfTwo(size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

__host__ __device__ __forceinline__ uint32_t globalThreadId() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

constexpr size_t nextPowerOfTwo(size_t n) {
    if (powerOfTwo(n))
        return n;

    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n |= n >> 32;
    n++;

    return n;
}

template <typename T>
size_t countOnes(T* data, size_t n) {
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (data[i]) {
            count++;
        }
    }
    return count;
}

#define SDIV(x, y) (((x) + (y) - 1) / (y))

#define CUDA_CALL(err)                                                      \
    do {                                                                    \
        cudaError_t err_ = (err);                                           \
        if (err_ == cudaSuccess) [[likely]] {                               \
            break;                                                          \
        }                                                                   \
        printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(err_);                                                         \
    } while (0)

struct MemoryUsage {
    size_t before = 0;
    size_t after = 0;

    void snapshotBefore() {
        size_t free;
        size_t total;
        CUDA_CALL(cudaMemGetInfo(&free, &total));
        before = total - free;
    }

    void snapshotAfter() {
        size_t free;
        size_t total;
        CUDA_CALL(cudaMemGetInfo(&free, &total));
        after = total - free;
    }

    [[nodiscard]] size_t used() const {
        return after - before;
    }
};