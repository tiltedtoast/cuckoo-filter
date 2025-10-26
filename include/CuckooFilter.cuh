#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <cub/cub.cuh>
#include <cuco/hash_functions.cuh>
#include <cuda/std/atomic>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <iostream>
#include <vector>
#include "helpers.cuh"

#if __has_include(<thrust/device_vector.h>)
    #include <thrust/device_vector.h>
    #define CUCKOO_FILTER_HAS_THRUST 1
#endif

namespace cg = cooperative_groups;

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
class CuckooFilter;

template <typename Config>
__global__ void insertKernel(
    const typename Config::KeyType* keys,
    size_t n,
    typename CuckooFilter<Config>::DeviceView view
);

template <typename Config>
__global__ void insertKernelSorted(
    const typename Config::KeyType* keys,
    const typename CuckooFilter<Config>::PackedTagType* packedTags,
    size_t n,
    typename CuckooFilter<Config>::DeviceView view
);

template <typename Config>
__global__ void computePackedTagsKernel(
    const typename Config::KeyType* keys,
    typename CuckooFilter<Config>::PackedTagType* packedTags,
    size_t n,
    size_t numBuckets
);

template <typename Config>
__global__ void containsKernel(
    const typename Config::KeyType* keys,
    bool* output,
    size_t n,
    typename CuckooFilter<Config>::DeviceView view
);

template <typename Config>
__global__ void deleteKernel(
    const typename Config::KeyType* keys,
    bool* output,
    size_t n,
    typename CuckooFilter<Config>::DeviceView view
);

template <typename Config>
class CuckooFilter {
   public:
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

    static constexpr size_t bucketSize = []() constexpr {
        size_t v = 1;
        while ((v << 1) <= maxEntriesByBytes) {
            v <<= 1;
        }
        return v;
    }();

    static_assert(bucketSize > 0, "Bucket size must be greater than 0");

    using PackedTagType = uint64_t;

    struct PackedTag {
        PackedTagType value;

        // Lower bits = fingerprint
        // Upper bits = bucket index
        static constexpr size_t fpBits = bitsPerTag;
        static constexpr size_t totalBits = sizeof(PackedTagType) * 8;
        static constexpr size_t bucketIdxBits = totalBits - fpBits;

        static_assert(
            fpBits < totalBits,
            "fpBits must leave at least some bits for bucketIdx"
        );

        static constexpr PackedTagType fpMask =
            PackedTagType((1ULL << fpBits) - 1ULL);

        static constexpr PackedTagType bucketIdxMask =
            PackedTagType(((1ULL << bucketIdxBits) - 1ULL) << fpBits);

        __host__ __device__ PackedTag() : value(0) {
        }

        __host__ __device__ PackedTag(TagType fp, uint64_t bucketIdx)
            : value(0) {
            setFingerprint(fp);
            setBucketIdx(bucketIdx);
        }

        __host__ __device__ TagType getFingerprint() const {
            return static_cast<TagType>(value & fpMask);
        }

        __host__ __device__ uint64_t getBucketIndex() const {
            return uint64_t((value & bucketIdxMask) >> fpBits);
        }

        __host__ __device__ void setFingerprint(TagType fp) {
            value =
                (value & ~fpMask) | (static_cast<PackedTagType>(fp) & fpMask);
        }

        __host__ __device__ void setBucketIdx(size_t bucketIdx) {
            PackedTagType v = static_cast<PackedTagType>(bucketIdx) << fpBits;
            value = (value & ~bucketIdxMask) | v;
        }
    };

    static constexpr TagType EMPTY = 0;
    static constexpr size_t tagMask = (1ULL << bitsPerTag) - 1;

    struct __align__(128) Bucket {
        cuda::std::atomic<TagType> tags[bucketSize];

        __forceinline__ __device__ int findSlot(TagType tag, TagType target) {
            uint32_t idx = tag & (bucketSize - 1);
            for (size_t i = 0; i < bucketSize; ++i) {
                TagType current = tags[idx].load(cuda::memory_order_relaxed);
                if (current == target) {
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
            return tags[slot].compare_exchange_strong(
                expected,
                tag,
                cuda::memory_order_relaxed,
                cuda::memory_order_relaxed
            );
        }

        __device__ bool tryRemoveAt(size_t slot, TagType tag) {
            TagType expected = tag;
            return tags[slot].compare_exchange_strong(
                expected,
                EMPTY,
                cuda::memory_order_relaxed,
                cuda::memory_order_relaxed
            );
        }

        __device__ TagType getTagAt(size_t slot) const {
            return tags[slot].load(cuda::memory_order_relaxed);
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
        return cuco::xxhash_32<H>()(key);
    }

    template <typename H>
    static __host__ __device__ uint64_t hash64(const H& key) {
        return cuco::xxhash_64<H>()(key);
    }

    static __host__ __device__ cuda::std::tuple<size_t, size_t, TagType>
    getCandidateBuckets(const T& key, size_t numBuckets) {
        const uint64_t h = hash64(key);

        // Upper 32 bits for the fingerprint
        const uint32_t h_fp = h >> 32;
        const TagType fp = static_cast<TagType>(h_fp & tagMask) + 1;

        // Lower 32 bits for the bucket indices
        const uint32_t h_bucket = h & 0xFFFFFFFF;
        const size_t i1 = h_bucket & (numBuckets - 1);
        const size_t i2 = getAlternateBucket(i1, fp, numBuckets);

        return {i1, i2, fp};
    }

    static __host__ __device__ size_t
    getAlternateBucket(size_t bucket, TagType fp, size_t numBuckets) {
        return bucket ^ (hash(fp) & (numBuckets - 1));
    }

    static size_t
    calculateNumBuckets(size_t capacity, double targetLoadFactor) {
        double itemsPerBucket = bucketSize * targetLoadFactor;

        auto requiredBuckets = static_cast<size_t>(
            std::ceil(static_cast<double>(capacity) / itemsPerBucket)
        );

        return nextPowerOfTwo(requiredBuckets);
    }

   public:
    CuckooFilter(const CuckooFilter&) = delete;
    CuckooFilter& operator=(const CuckooFilter&) = delete;

    explicit CuckooFilter(size_t capacity, double targetLoadFactor = 0.95)
        : numBuckets(calculateNumBuckets(capacity, targetLoadFactor)) {
        assert(
            targetLoadFactor > 0.0 && targetLoadFactor <= 1.0 &&
            "Target load factor must be in range (0, 1]"
        );

        CUDA_CALL(cudaMalloc(&d_buckets, numBuckets * sizeof(Bucket)));
        CUDA_CALL(
            cudaMalloc(&d_numOccupied, sizeof(cuda::std::atomic<size_t>))
        );

        assert(
            powerOfTwo(numBuckets) && "Number of buckets must be a power of 2"
        );

        clear();
    }

    ~CuckooFilter() {
        if (d_buckets) {
            CUDA_CALL(cudaFree(d_buckets));
        }
        if (d_numOccupied) {
            CUDA_CALL(cudaFree(d_numOccupied));
        }
    }

    size_t insertMany(const T* d_keys, const size_t n) {
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
                size_t numBlocks = SDIV(currentChunkSize, blockSize);
                insertKernel<Config><<<numBlocks, blockSize, 0, streams[i]>>>(
                    d_keys + offset, currentChunkSize, getDeviceView()
                );
            }
        }

        CUDA_CALL(cudaDeviceSynchronize());

        for (auto& stream : streams) {
            CUDA_CALL(cudaStreamDestroy(stream));
        }

        CUDA_CALL(cudaMemcpy(
            &h_numOccupied,
            d_numOccupied,
            sizeof(size_t),
            cudaMemcpyDeviceToHost
        ));

        return h_numOccupied;
    }

    size_t insertManySorted(const T* keys, const size_t n) {
        T* d_keys;
        PackedTagType* d_packedTags;

        CUDA_CALL(cudaMalloc(&d_keys, n * sizeof(T)));
        CUDA_CALL(cudaMalloc(&d_packedTags, n * sizeof(PackedTagType)));

        CUDA_CALL(
            cudaMemcpy(d_keys, keys, n * sizeof(T), cudaMemcpyHostToDevice)
        );

        size_t numBlocks = SDIV(n, blockSize);

        computePackedTagsKernel<Config>
            <<<numBlocks, blockSize>>>(d_keys, d_packedTags, n, numBuckets);

        void* d_tempStorage = nullptr;
        size_t tempStorageBytes = 0;

        cub::DeviceRadixSort::SortKeys(
            d_tempStorage, tempStorageBytes, d_packedTags, d_packedTags, n
        );

        CUDA_CALL(cudaMalloc(&d_tempStorage, tempStorageBytes));

        cub::DeviceRadixSort::SortKeys(
            d_tempStorage, tempStorageBytes, d_packedTags, d_packedTags, n
        );

        CUDA_CALL(cudaFree(d_tempStorage));

        insertKernelSorted<Config><<<numBlocks, blockSize>>>(
            d_keys, d_packedTags, n, getDeviceView()
        );

        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaFree(d_keys));
        CUDA_CALL(cudaFree(d_packedTags));

        CUDA_CALL(cudaMemcpy(
            &h_numOccupied,
            d_numOccupied,
            sizeof(size_t),
            cudaMemcpyDeviceToHost
        ));

        return h_numOccupied;
    }

    void containsMany(const T* d_keys, const size_t n, bool* d_output) {
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
                size_t numBlocks = SDIV(currentChunkSize, blockSize);
                containsKernel<Config><<<numBlocks, blockSize, 0, streams[i]>>>(
                    d_keys + offset,
                    d_output + offset,
                    currentChunkSize,
                    getDeviceView()
                );
            }
        }

        CUDA_CALL(cudaDeviceSynchronize());

        for (auto& stream : streams) {
            CUDA_CALL(cudaStreamDestroy(stream));
        }
    }

    size_t
    deleteMany(const T* d_keys, const size_t n, bool* d_output = nullptr) {
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
                size_t numBlocks = SDIV(currentChunkSize, blockSize);
                bool* outputPtr =
                    d_output != nullptr ? d_output + offset : nullptr;
                deleteKernel<Config><<<numBlocks, blockSize, 0, streams[i]>>>(
                    d_keys + offset,
                    outputPtr,
                    currentChunkSize,
                    getDeviceView()
                );
            }
        }

        CUDA_CALL(cudaDeviceSynchronize());

        for (auto& stream : streams) {
            CUDA_CALL(cudaStreamDestroy(stream));
        }

        CUDA_CALL(cudaMemcpy(
            &h_numOccupied,
            d_numOccupied,
            sizeof(size_t),
            cudaMemcpyDeviceToHost
        ));

        return h_numOccupied;
    }

#ifdef CUCKOO_FILTER_HAS_THRUST
    size_t insertMany(const thrust::device_vector<T>& d_keys) {
        return insertMany(
            thrust::raw_pointer_cast(d_keys.data()), d_keys.size()
        );
    }

    void containsMany(
        const thrust::device_vector<T>& d_keys,
        thrust::device_vector<bool>& d_output
    ) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        containsMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            thrust::raw_pointer_cast(d_output.data())
        );
    }

    void containsMany(
        const thrust::device_vector<T>& d_keys,
        thrust::device_vector<uint8_t>& d_output
    ) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        containsMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))
        );
    }

    size_t deleteMany(
        const thrust::device_vector<T>& d_keys,
        thrust::device_vector<bool>& d_output
    ) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        return deleteMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            thrust::raw_pointer_cast(d_output.data())
        );
    }

    size_t deleteMany(
        const thrust::device_vector<T>& d_keys,
        thrust::device_vector<uint8_t>& d_output
    ) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        return deleteMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))
        );
    }

    size_t deleteMany(const thrust::device_vector<T>& d_keys) {
        return deleteMany(
            thrust::raw_pointer_cast(d_keys.data()), d_keys.size(), nullptr
        );
    }
#endif  // CUCKOO_FILTER_HAS_THRUST

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

    size_t capacity() {
        return numBuckets * bucketSize;
    }

    [[nodiscard]] size_t getNumBuckets() const {
        return numBuckets;
    }

    struct DeviceView {
        Bucket* d_buckets;
        cuda::std::atomic<size_t>* d_numOccupied;
        size_t numBuckets;

        __device__ bool tryRemoveAtBucket(size_t bucketIdx, TagType tag) {
            uint32_t idx = tag & (bucketSize - 1);
            for (size_t i = 0; i < bucketSize; ++i) {
                if (d_buckets[bucketIdx].tryRemoveAt(idx, tag)) {
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
                        currentFp, cuda::memory_order_relaxed
                    );

                currentFp = evictedFp;
                currentBucket = CuckooFilter::getAlternateBucket(
                    currentBucket, evictedFp, numBuckets
                );
            }
            return false;
        }

        __device__ bool insert(const T& key) {
            auto [i1, i2, fp] = getCandidateBuckets(key, numBuckets);

            if (tryInsertAtBucket(i1, fp) || tryInsertAtBucket(i2, fp)) {
                return true;
            }

            auto startBucket = (fp & 1) == 0 ? i1 : i2;

            return insertWithEviction(fp, startBucket);
        }

        __device__ bool contains(const T& key) const {
            auto [i1, i2, fp] = getCandidateBuckets(key, numBuckets);
            return d_buckets[i1].contains(fp) || d_buckets[i2].contains(fp);
        }

        __device__ bool remove(const T& key) {
            auto [i1, i2, fp] = getCandidateBuckets(key, numBuckets);

            return tryRemoveAtBucket(i1, fp) || tryRemoveAtBucket(i2, fp);
        }
    };

    DeviceView getDeviceView() {
        return DeviceView{
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
    typename CuckooFilter<Config>::DeviceView view
) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    auto idx = globalThreadId();

    if (idx >= n) {
        return;
    }

    int success = view.insert(keys[idx]);

    int tile_sum = cg::reduce(tile, success, cg::plus<int>());

    if (tile.thread_rank() == 0 && tile_sum > 0) {
        view.d_numOccupied->fetch_add(tile_sum, cuda::memory_order_relaxed);
    }
}

template <typename Config>
__global__ void containsKernel(
    const typename Config::KeyType* keys,
    bool* output,
    size_t n,
    typename CuckooFilter<Config>::DeviceView view
) {
    auto idx = globalThreadId();

    if (idx < n) {
        output[idx] = view.contains(keys[idx]);
    }
}

template <typename Config>
__global__ void deleteKernel(
    const typename Config::KeyType* keys,
    bool* output,
    size_t n,
    typename CuckooFilter<Config>::DeviceView view
) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    auto idx = globalThreadId();

    if (idx >= n) {
        return;
    }

    int success = view.remove(keys[idx]);

    if (output != nullptr) {
        output[idx] = success;
    }

    int tileSum = cg::reduce(tile, success, cg::plus<int>());

    if (tile.thread_rank() == 0 && tileSum > 0) {
        view.d_numOccupied->fetch_sub(tileSum, cuda::memory_order_relaxed);
    }
}

template <typename Config>
__global__ void computePackedTagsKernel(
    const typename Config::KeyType* keys,
    typename CuckooFilter<Config>::PackedTagType* packedTags,
    size_t n,
    size_t numBuckets
) {
    size_t idx = globalThreadId();

    if (idx >= n) {
        return;
    }

    using Filter = CuckooFilter<Config>;
    using PackedTagType = typename Filter::PackedTagType;
    constexpr size_t bitsPerTag = Config::bitsPerTag;

    typename Config::KeyType key = keys[idx];
    auto [i1, i2, fp] = Filter::getCandidateBuckets(key, numBuckets);

    packedTags[idx] = (static_cast<PackedTagType>(i1) << bitsPerTag) |
                      static_cast<PackedTagType>(fp);
}

template <typename Config>
__global__ void insertKernelSorted(
    const typename Config::KeyType* keys,
    const typename CuckooFilter<Config>::PackedTagType* packedTags,
    size_t n,
    typename CuckooFilter<Config>::DeviceView view
) {
    auto block = cg::this_thread_block();
    auto tile = cg::tiled_partition<32>(block);

    size_t idx = globalThreadId();

    if (idx >= n)
        return;

    using Filter = CuckooFilter<Config>;
    using TagType = typename Filter::TagType;
    using PackedTagType = typename Filter::PackedTagType;

    constexpr size_t bitsPerTag = Config::bitsPerTag;
    constexpr TagType fpMask = (1ULL << bitsPerTag) - 1;

    PackedTagType packedTag = packedTags[idx];
    size_t primaryBucket = packedTag >> bitsPerTag;
    auto fp = static_cast<TagType>(packedTag & fpMask);

    size_t secondaryBucket =
        Filter::getAlternateBucket(primaryBucket, fp, view.numBuckets);

    bool success = false;
    if (view.tryInsertAtBucket(primaryBucket, fp) ||
        view.tryInsertAtBucket(secondaryBucket, fp)) {
        success = true;
    } else {
        auto startBucket = (fp & 1) == 0 ? primaryBucket : secondaryBucket;
        success = view.insertWithEviction(fp, startBucket);
    }

    int tileSum = cg::reduce(tile, static_cast<int>(success), cg::plus<int>());

    if (tile.thread_rank() == 0 && tileSum > 0) {
        view.d_numOccupied->fetch_add(tileSum, cuda::memory_order_relaxed);
    }
}
