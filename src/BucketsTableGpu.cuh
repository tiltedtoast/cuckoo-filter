#pragma once

#include <cstdint>
#include <ctime>
#include <cuco/hash_functions.cuh>
#include <cuda/std/atomic>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <iostream>
#include <vector>
#include "common.cuh"

struct SpinLock {
    int lock_flag = 0;

    __device__ void lock() {
        while (atomicCAS(&lock_flag, 0, 1) != 0) {
            // Spin until lock is acquired
        }
    }

    __device__ void unlock() {
        atomicExch(&lock_flag, 0);
    }
};

template <
    typename T,
    size_t bitsPerTag,
    size_t bucketSize,
    size_t maxEvictions,
    size_t blockSize>
class BucketsTableGpu;

template <
    typename T,
    size_t bitsPerTag,
    size_t bucketSize,
    size_t maxEvictions,
    size_t blockSize>
__global__ void insertKernel(
    const T* keys,
    size_t n,
    typename BucketsTableGpu<
        T,
        bitsPerTag,
        bucketSize,
        maxEvictions,
        blockSize>::DeviceTableView table_view
);

template <
    typename T,
    size_t bitsPerTag,
    size_t bucketSize,
    size_t maxEvictions,
    size_t blockSize>
__global__ void containsKernel(
    const T* keys,
    bool* output,
    size_t n,
    typename BucketsTableGpu<
        T,
        bitsPerTag,
        bucketSize,
        maxEvictions,
        blockSize>::DeviceTableView table_view
);

template <
    typename T,
    size_t bitsPerTag,
    size_t bucketSize = 32,
    size_t maxEvictions = 500,
    size_t blockSize = 256>
class BucketsTableGpu {
    static_assert(bitsPerTag <= 32, "The tag cannot be larger than 32 bits");
    static_assert(bitsPerTag >= 1, "The tag must be at least 1 bit");
    static_assert(
        bitsPerTag <= 8 * sizeof(T),
        "The tag cannot be larger than the size of the type"
    );

    static_assert(bucketSize > 0, "Bucket size must be greater than 0");

    size_t numBuckets;

   public:
    using TagType = typename std::conditional<
        bitsPerTag <= 8,
        uint8_t,
        typename std::conditional<bitsPerTag <= 16, uint16_t, uint32_t>::type>::
        type;

    static constexpr TagType EMPTY = 0;
    static constexpr size_t tagMask = (1ULL << bitsPerTag) - 1;

    struct __align__(alignof(TagType)) Bucket {
        TagType tags[bucketSize];

        __forceinline__ __device__ int findSlot(TagType tag, TagType target) {
            uint32_t idx = tag & (bucketSize - 1);
            for (size_t i = 0; i < bucketSize; ++i) {
                if (tags[idx] == target) {
                    return static_cast<int>(idx);
                }
                idx = (idx + 1) & (bucketSize - 1);
            }
            return -1;
        }

        __device__ int findEmptySlot(TagType tag) {
            return findSlot(tag, EMPTY);
        }

        __device__ bool contains(TagType tag) {
            return findSlot(tag, tag) != -1;
        }

        __device__ void insertAt(size_t slot, TagType tag) {
            tags[slot] = tag;
        }

        __device__ bool remove(TagType tag) {
            size_t idx = findSlot(tag, tag);
            if (idx != -1) {
                tags[idx] = EMPTY;
                return true;
            }
            return false;
        }

        __device__ TagType getTagAt(size_t slot) const {
            return tags[slot];
        }
    };

   private:
    Bucket* d_buckets;
    SpinLock* d_locks{};

    size_t* d_numOccupied{};
    size_t h_numOccupied = 0;

   public:
    template <typename H>
    static __host__ __device__ uint32_t hash(const H& key) {
        auto bytes = reinterpret_cast<const cuda::std::byte*>(&key);
        cuco::xxhash_32<H> hasher;
        return hasher.compute_hash(bytes, sizeof(H));
    }

    static __host__ __device__ TagType fingerprint(const T& key) {
        uint32_t hash_val = hash(key);
        auto fp = static_cast<TagType>(hash_val & tagMask);
        return fp == 0 ? 1 : fp;
    }

    static __host__ __device__ cuda::std::tuple<size_t, size_t, TagType>
    getCandidateBuckets(const T& key, size_t numBuckets) {
        TagType fp = fingerprint(key);
        size_t i1 = hash(key) & (numBuckets - 1);
        size_t i2 = i1 ^ (hash(fp) & (numBuckets - 1));
        return {i1, i2, fp};
    }

    static __host__ __device__ size_t
    getAlternateBucket(size_t bucket, TagType fp, size_t numBuckets) {
        return bucket ^ (hash(fp) & (numBuckets - 1));
    }

   public:
    explicit BucketsTableGpu(size_t numBuckets) : numBuckets(numBuckets) {
        CUDA_CALL(cudaMalloc(&d_buckets, numBuckets * sizeof(Bucket)));
        CUDA_CALL(cudaMalloc(&d_locks, numBuckets * sizeof(SpinLock)));
        CUDA_CALL(
            cudaMalloc(&d_numOccupied, sizeof(cuda::std::atomic<size_t>))
        );

        assert(
            powerOfTwo(numBuckets) && "Number of buckets must be a power of 2"
        );

        clear();
    }

    ~BucketsTableGpu() {
        if (d_buckets) {
            CUDA_CALL(cudaFree(d_buckets));
        }
        if (d_locks) {
            CUDA_CALL(cudaFree(d_locks));
        }
        if (d_numOccupied) {
            CUDA_CALL(cudaFree(d_numOccupied));
        }
    }

    size_t insertMany(const T* keys, const size_t n) {
        T* d_keys;
        CUDA_CALL(cudaMalloc(&d_keys, n * sizeof(T)));

        constexpr size_t numStreams = 4;
        const size_t chunkSize = SDIV(n, numStreams);
        cudaStream_t streams[numStreams];

        for (auto& stream : streams) {
            CUDA_CALL(cudaStreamCreate(&stream));
        }

        for (size_t i = 0; i < numStreams; ++i) {
            size_t offset = i * chunkSize;
            size_t currentChunkSize = std::min(chunkSize, n - offset);

            if (currentChunkSize > 0) {
                CUDA_CALL(cudaMemcpyAsync(
                    d_keys + offset,
                    keys + offset,
                    currentChunkSize * sizeof(T),
                    cudaMemcpyHostToDevice,
                    streams[i]
                ));

                size_t numBlocks = SDIV(currentChunkSize, blockSize);
                insertKernel<T, bitsPerTag, bucketSize, maxEvictions, blockSize>
                    <<<numBlocks, blockSize, 0, streams[i]>>>(
                        d_keys + offset, currentChunkSize, get_device_view()
                    );
            }
        }

        CUDA_CALL(cudaDeviceSynchronize());

        for (auto& stream : streams) {
            CUDA_CALL(cudaStreamDestroy(stream));
        }

        CUDA_CALL(cudaFree(d_keys));

        CUDA_CALL(cudaMemcpy(
            &h_numOccupied,
            d_numOccupied,
            sizeof(size_t),
            cudaMemcpyDeviceToHost
        ));

        return h_numOccupied;
    }

    void containsMany(const T* keys, const size_t n, bool* output) {
        T* d_keys;
        bool* d_output;
        CUDA_CALL(cudaMalloc(&d_keys, n * sizeof(T)));
        CUDA_CALL(cudaMalloc(&d_output, n * sizeof(bool)));

        constexpr size_t numStreams = 4;
        const size_t chunkSize = SDIV(n, numStreams);
        cudaStream_t streams[numStreams];

        for (auto& stream : streams) {
            CUDA_CALL(cudaStreamCreate(&stream));
        }

        for (size_t i = 0; i < numStreams; ++i) {
            size_t offset = i * chunkSize;
            size_t currentChunkSize = std::min(chunkSize, n - offset);

            if (currentChunkSize > 0) {
                CUDA_CALL(cudaMemcpyAsync(
                    d_keys + offset,
                    keys + offset,
                    currentChunkSize * sizeof(T),
                    cudaMemcpyHostToDevice,
                    streams[i]
                ));

                size_t numBlocks = SDIV(currentChunkSize, blockSize);
                containsKernel<
                    T,
                    bitsPerTag,
                    bucketSize,
                    maxEvictions,
                    blockSize><<<numBlocks, blockSize, 0, streams[i]>>>(
                    d_keys + offset,
                    d_output + offset,
                    currentChunkSize,
                    get_device_view()
                );

                CUDA_CALL(cudaMemcpyAsync(
                    output + offset,
                    d_output + offset,
                    currentChunkSize * sizeof(bool),
                    cudaMemcpyDeviceToHost,
                    streams[i]
                ));
            }
        }

        CUDA_CALL(cudaDeviceSynchronize());

        for (auto& stream : streams) {
            CUDA_CALL(cudaStreamDestroy(stream));
        }

        CUDA_CALL(cudaFree(d_keys));
        CUDA_CALL(cudaFree(d_output));
    }
    void clear() {
        CUDA_CALL(cudaMemset(d_buckets, 0, numBuckets * sizeof(Bucket)));
        CUDA_CALL(cudaMemset(d_locks, 0, numBuckets * sizeof(SpinLock)));
        CUDA_CALL(
            cudaMemset(d_numOccupied, 0, sizeof(cuda::std::atomic<size_t>))
        );
        h_numOccupied = 0;
    }

    float loadFactor() {
        CUDA_CALL(cudaMemcpy(
            &h_numOccupied,
            d_numOccupied,
            sizeof(size_t),
            cudaMemcpyDeviceToHost
        ));
        return static_cast<float>(h_numOccupied) / (numBuckets * bucketSize);
    }

    [[nodiscard]] double expectedFalsePositiveRate() const {
        return (2.0 * bucketSize) / (1ULL << bitsPerTag);
    }

    struct DeviceTableView {
        Bucket* d_buckets;
        SpinLock* d_locks;
        size_t* d_numOccupied;
        size_t& numBuckets;

        __device__ bool tryInsertAtBucket(size_t bucketIdx, TagType tag) {
            d_locks[bucketIdx].lock();

            int slot = d_buckets[bucketIdx].findEmptySlot(tag);
            if (slot != -1) {
                d_buckets[bucketIdx].insertAt(slot, tag);
                atomicAdd(
                    reinterpret_cast<unsigned long long*>(d_numOccupied), 1ULL
                );
                d_locks[bucketIdx].unlock();
                return true;
            }
            d_locks[bucketIdx].unlock();
            return false;
        }

        __device__ bool insertWithEviction(TagType fp, size_t startBucket) {
            TagType currentFp = fp;
            size_t currentBucket = startBucket;

            for (size_t evictions = 0; evictions < maxEvictions; ++evictions) {
                d_locks[currentBucket].lock();

                int slot = d_buckets[currentBucket].findEmptySlot(fp);
                if (slot != -1) {
                    d_buckets[currentBucket].insertAt(slot, currentFp);
                    atomicAdd(
                        reinterpret_cast<unsigned long long*>(d_numOccupied),
                        1ULL
                    );
                    d_locks[currentBucket].unlock();
                    return true;
                }

                auto evictSlot = (currentFp + evictions) & (bucketSize - 1);
                TagType evictedFp =
                    d_buckets[currentBucket].getTagAt(evictSlot);
                d_buckets[currentBucket].insertAt(evictSlot, currentFp);

                d_locks[currentBucket].unlock();

                currentFp = evictedFp;
                currentBucket = BucketsTableGpu::getAlternateBucket(
                    currentBucket, evictedFp, numBuckets
                );
            }
            return false;
        }

        __device__ bool insert(const T& key) {
            auto [i1, i2, fp] =
                BucketsTableGpu::getCandidateBuckets(key, numBuckets);

            if (tryInsertAtBucket(i1, fp) || tryInsertAtBucket(i2, fp)) {
                return true;
            }

            if (insertWithEviction(fp, i1)) {
                if (d_buckets[i1].contains(fp) || d_buckets[i2].contains(fp)) {
                    return true;
                }
                return false;
            }
            return false;
        }

        __device__ bool contains(const T& key) const {
            auto [i1, i2, fp] =
                BucketsTableGpu::getCandidateBuckets(key, numBuckets);
            return d_buckets[i1].contains(fp) || d_buckets[i2].contains(fp);
        }
    };

    DeviceTableView get_device_view() {
        return DeviceTableView{
            d_buckets,
            d_locks,
            d_numOccupied,
            numBuckets,
        };
    }
};

template <
    typename T,
    size_t bitsPerTag,
    size_t bucketSize,
    size_t maxEvictions,
    size_t blockSize>
__global__ void insertKernel(
    const T* keys,
    size_t n,
    typename BucketsTableGpu<
        T,
        bitsPerTag,
        bucketSize,
        maxEvictions,
        blockSize>::DeviceTableView table_view
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        for (int i = 0; i < 3; ++i) {
            if (table_view.insert(keys[idx])) {
                return;
            }
        }
        // We kind of have a big problem at this point, since it means our item
        // has been evicted and reinserted multiple times. May as well buy a
        // lottery ticket I suppose.
    }
}

template <
    typename T,
    size_t bitsPerTag,
    size_t bucketSize,
    size_t maxEvictions,
    size_t blockSize>
__global__ void containsKernel(
    const T* keys,
    bool* output,
    size_t n,
    typename BucketsTableGpu<
        T,
        bitsPerTag,
        bucketSize,
        maxEvictions,
        blockSize>::DeviceTableView table_view
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = table_view.contains(keys[idx]);
    }
}