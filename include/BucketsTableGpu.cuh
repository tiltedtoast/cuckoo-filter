#pragma once

#include <cmath>
#include <cstdint>
#include <ctime>
#include <cuco/hash_functions.cuh>
#include <cuda/std/atomic>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <iostream>
#include <vector>
#include "helpers.cuh"

template <
    typename T,
    size_t bitsPerTag_,
    size_t maxEvictions_,
    size_t blockSize_,
    size_t maxBucketBytes_>
struct CuckooConfig {
    using KeyType = T;
    static constexpr size_t bitsPerTag = bitsPerTag_;
    static constexpr size_t maxEvictions = maxEvictions_;
    static constexpr size_t blockSize = blockSize_;
    static constexpr size_t maxBucketBytes = maxBucketBytes_;
};

template <typename Config>
class BucketsTableGpu;

template <typename Config>
__global__ void insertKernel(
    const typename Config::KeyType* keys,
    size_t n,
    typename BucketsTableGpu<Config>::DeviceTableView tableView
);

template <typename Config>
__global__ void containsKernel(
    const typename Config::KeyType* keys,
    bool* output,
    size_t n,
    typename BucketsTableGpu<Config>::DeviceTableView tableView
);

template <typename Config>
class BucketsTableGpu {
    using T = typename Config::KeyType;
    static constexpr size_t bitsPerTag = Config::bitsPerTag;

    using TagType = typename std::conditional<
        bitsPerTag <= 8,
        uint8_t,
        typename std::conditional<bitsPerTag <= 16, uint16_t, uint32_t>::type>::
        type;

    static constexpr size_t tagEntryBytes = sizeof(TagType);

    static constexpr size_t maxEntriesByBytes =
        (Config::maxBucketBytes) / tagEntryBytes;

    static constexpr size_t maxEvictions = Config::maxEvictions;
    static constexpr size_t blockSize = Config::blockSize;
    static_assert(bitsPerTag <= 32, "The tag cannot be larger than 32 bits");
    static_assert(bitsPerTag >= 1, "The tag must be at least 1 bit");
    static_assert(
        bitsPerTag <= 8 * sizeof(T),
        "The tag cannot be larger than the size of the type"
    );

   public:
    static constexpr size_t bucketSize = []() constexpr {
        size_t v = 1;
        while ((v << 1) <= maxEntriesByBytes) {
            v <<= 1;
        }
        return v;
    }();

    static_assert(bucketSize > 0, "Bucket size must be greater than 0");

    struct PackedTag {
        static_assert(
            sizeof(TagType) * 8 <= 64,
            "TagType must not be larger than 64 bits"
        );

        TagType value;

        // Lower bits = fingerprint
        // Next bit = bucket type where key lives (1 for primary, 0 secondary)
        // Upper bits = bucket index where key lives
        static constexpr size_t fpBits = bitsPerTag;
        static constexpr size_t totalBits = sizeof(TagType) * 8;
        static constexpr size_t bucketIdxBits = totalBits - fpBits - 1;
        static_assert(
            fpBits < totalBits - 1,
            "fpBits must leave at least 1 bit for bucketType and 1 "
            "for bucketIdx"
        );

        static constexpr TagType fpMask = TagType((1ULL << fpBits) - 1ULL);

        static constexpr TagType bucketTypeMask = TagType(1ULL << fpBits);

        static constexpr TagType bucketIdxMask =
            TagType(((1ULL << bucketIdxBits) - 1ULL) << (fpBits + 1));

        __host__ __device__
        PackedTag(TagType fp, uint64_t bucketIdx, bool isPrimary)
            : value(0) {
            setFingerprint(fp);
            setBucketIdx(bucketIdx);
            setBucketType(isPrimary);
        }

        __host__ __device__ TagType getFingerprint() const {
            return value & fpMask;
        }

        __host__ __device__ uint64_t getBucketIndex() const {
            return uint64_t((value & bucketIdxMask) >> (fpBits + 1));
        }

        __host__ __device__ bool isPrimary() const {
            return (value & bucketTypeMask) != 0;
        }

        __host__ __device__ bool isSecondary() const {
            return !isPrimary();
        }

        __host__ __device__ void setFingerprint(TagType fp) {
            value = (value & ~fpMask) | (fp & fpMask);
        }

        __host__ __device__ void setBucketIdx(size_t bucketIdx) {
            TagType v = TagType(bucketIdx) << (fpBits + 1);

            value = (value & ~bucketIdxMask) | v;
        }

        __host__ __device__ void setBucketType(bool primary) {
            if (primary) {
                value |= bucketTypeMask;
            } else {
                value &= ~bucketTypeMask;
            }
        }

        __host__ __device__ void setPrimary() {
            setBucketType(true);
        }

        __host__ __device__ void setSecondary() {
            setBucketType(false);
        }
    };

    static constexpr TagType EMPTY = 0;
    static constexpr size_t tagMask = (1ULL << bitsPerTag) - 1;

    struct __align__(128) Bucket {
        cuda::std::atomic<TagType> tags[bucketSize];

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

        __device__ bool contains(TagType tag) {
            return findSlot(tag, tag) != -1;
        }

        __device__ bool tryInsertAt(size_t slot, TagType tag) {
            TagType expected = EMPTY;
            return tags[slot].compare_exchange_strong(expected, tag);
        }

        __device__ bool tryRemoveAt(size_t slot, TagType tag) {
            TagType expected = tag;
            return tags[slot].compare_exchange_strong(expected, EMPTY);
        }

        __device__ TagType getTagAt(size_t slot) const {
            return tags[slot];
        }
    };

   private:
    size_t numBuckets;
    Bucket* d_buckets;
    cuda::std::atomic<size_t>* d_numOccupied{};
    size_t h_numOccupied = 0;

   public:
    template <typename H>
    static __host__ __device__ uint32_t hash(const H& key) {
        auto bytes = reinterpret_cast<const cuda::std::byte*>(&key);
        cuco::xxhash_32<H> hasher;
        return hasher.compute_hash(bytes, sizeof(H));
    }

    static __host__ __device__ TagType fingerprint(const T& key) {
        return static_cast<TagType>(hash(key) & tagMask) + 1;
    }

    static __host__ __device__ cuda::std::tuple<size_t, size_t, TagType>
    getCandidateBuckets(const T& key, size_t numBuckets) {
        TagType fp = fingerprint(key);
        size_t i1 = hash(key) & (numBuckets - 1);
        size_t i2 = getAlternateBucket(i1, fp, numBuckets);
        return {i1, i2, fp};
    }

    static __host__ __device__ size_t
    getAlternateBucket(size_t bucket, TagType fp, size_t numBuckets) {
        return bucket ^ (hash(fp) & (numBuckets - 1));
    }

   public:
    BucketsTableGpu(const BucketsTableGpu&) = delete;
    BucketsTableGpu& operator=(const BucketsTableGpu&) = delete;

    explicit BucketsTableGpu(size_t numBuckets) : numBuckets(numBuckets) {
        CUDA_CALL(cudaMalloc(&d_buckets, numBuckets * sizeof(Bucket)));
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
        if (d_numOccupied) {
            CUDA_CALL(cudaFree(d_numOccupied));
        }
    }

    size_t insertMany(const T* keys, const size_t n) {
        T* d_keys;
        CUDA_CALL(cudaMalloc(&d_keys, n * sizeof(T)));

        const size_t numStreams =
            std::clamp(n / blockSize, size_t(1), size_t(12));

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
                insertKernel<Config><<<numBlocks, blockSize, 0, streams[i]>>>(
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

        const size_t numStreams =
            std::clamp(n / blockSize, size_t(1), size_t(12));
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
                containsKernel<Config><<<numBlocks, blockSize, 0, streams[i]>>>(
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

    struct DeviceTableView {
        Bucket* d_buckets;
        cuda::std::atomic<size_t>* d_numOccupied;
        size_t numBuckets;

        __device__ bool tryRemoveAtBucket(size_t bucketIdx, TagType tag) {
            uint32_t idx = tag & (bucketSize - 1);
            for (size_t i = 0; i < bucketSize; ++i) {
                if (d_buckets[bucketIdx].tryRemoveAt(idx, tag)) {
                    d_numOccupied->fetch_sub(1);
                    return true;
                }
                idx = (idx + 1) & (bucketSize - 1);
            }
            return false;
        }

        __device__ bool tryInsertAtBucket(size_t bucketIdx, TagType tag) {
            uint32_t idx = tag & (bucketSize - 1);
            for (size_t i = 0; i < bucketSize; ++i) {
                if (d_buckets[bucketIdx].tryInsertAt(idx, tag)) {
                    d_numOccupied->fetch_add(1);
                    return true;
                }
                idx = (idx + 1) & (bucketSize - 1);
            }
            return false;
        }

        __device__ bool insertWithEviction(TagType fp, size_t startBucket) {
            TagType currentFp = fp;
            size_t currentBucket = startBucket;

            for (size_t evictions = 0; evictions < maxEvictions; ++evictions) {
                if (tryInsertAtBucket(currentBucket, currentFp)) {
                    return true;
                }

                auto evictSlot = (currentFp + evictions) & (bucketSize - 1);

                TagType evictedFp =
                    d_buckets[currentBucket].tags[evictSlot].exchange(
                        currentFp
                    );

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

            return insertWithEviction(fp, i1);
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
            d_numOccupied,
            numBuckets,
        };
    }
};

template <typename Config>
__global__ void insertKernel(
    const typename Config::KeyType* keys,
    size_t n,
    typename BucketsTableGpu<Config>::DeviceTableView tableView
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        tableView.insert(keys[idx]);
    }
}

template <typename Config>
__global__ void containsKernel(
    const typename Config::KeyType* keys,
    bool* output,
    size_t n,
    typename BucketsTableGpu<Config>::DeviceTableView tableView
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = tableView.contains(keys[idx]);
    }
}