#pragma once

#include <cmath>
#include <cstdint>
#include <ctime>
#include <cub/cub.cuh>
#include <cuda/std/atomic>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <iostream>
#include <vector>
#include "hashutil.cuh"
#include "helpers.cuh"

#if __has_include(<thrust/device_vector.h>)
    #include <thrust/device_vector.h>
    #define CUCKOO_FILTER_HAS_THRUST 1
#endif

template <
    typename T,
    size_t bitsPerTag_,
    size_t maxEvictions_,
    size_t blockSize_,
    size_t bucketSize_>
struct CuckooConfig {
    using KeyType = T;
    static constexpr size_t bitsPerTag = bitsPerTag_;
    static constexpr size_t maxEvictions = maxEvictions_;
    static constexpr size_t blockSize = blockSize_;
    static constexpr size_t bucketSize = bucketSize_;
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
        typename std::conditional<bitsPerTag <= 16, uint16_t, uint32_t>::type>::type;

    static constexpr size_t tagEntryBytes = sizeof(TagType);
    static constexpr size_t bucketSize = Config::bucketSize;

    static constexpr size_t maxEvictions = Config::maxEvictions;
    static constexpr size_t blockSize = Config::blockSize;
    static_assert(bitsPerTag <= 32, "The tag cannot be larger than 32 bits");
    static_assert(bitsPerTag >= 1, "The tag must be at least 1 bit");
    static_assert(
        bitsPerTag <= 8 * sizeof(T),
        "The tag cannot be larger than the size of the type"
    );

    static_assert(bucketSize > 0, "Bucket size must be greater than 0");
    static_assert(powerOfTwo(bucketSize), "Bucket size must be a power of 2");

    using PackedTagType = uint64_t;

    struct PackedTag {
        PackedTagType value;

        // Lower bits = fingerprint
        // Upper bits = bucket index
        static constexpr size_t fpBits = bitsPerTag;
        static constexpr size_t totalBits = sizeof(PackedTagType) * 8;
        static constexpr size_t bucketIdxBits = totalBits - fpBits;

        static_assert(fpBits < totalBits, "fpBits must leave at least some bits for bucketIdx");

        static constexpr PackedTagType fpMask = PackedTagType((1ULL << fpBits) - 1ULL);

        static constexpr PackedTagType bucketIdxMask =
            PackedTagType(((1ULL << bucketIdxBits) - 1ULL) << fpBits);

        __host__ __device__ PackedTag() : value(0) {
        }

        __host__ __device__ PackedTag(TagType fp, uint64_t bucketIdx) : value(0) {
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
            value = (value & ~fpMask) | (static_cast<PackedTagType>(fp) & fpMask);
        }

        __host__ __device__ void setBucketIdx(size_t bucketIdx) {
            PackedTagType v = static_cast<PackedTagType>(bucketIdx) << fpBits;
            value = (value & ~bucketIdxMask) | v;
        }
    };

    static constexpr TagType EMPTY = 0;
    static constexpr size_t fpMask = (1ULL << bitsPerTag) - 1;

    struct Bucket {
        static_assert(powerOfTwo(bitsPerTag), "bitsPerTag must be a power of 2");
        static constexpr size_t tagsPerAtomic = sizeof(uint64_t) / sizeof(TagType);
        static_assert(
            bucketSize % tagsPerAtomic == 0,
            "bucketSize must be divisible by tagsPerAtomic"
        );
        static_assert(powerOfTwo(tagsPerAtomic), "tagsPerAtomic must be a power of 2");

        static constexpr size_t atomicCount = bucketSize / tagsPerAtomic;
        static_assert(powerOfTwo(atomicCount), "atomicCount must be a power of 2");

        cuda::std::atomic<uint64_t> packedTags[atomicCount];

        __host__ __device__ TagType extractTag(uint64_t packed, size_t tagIdx) const {
            return static_cast<TagType>((packed >> (tagIdx * bitsPerTag)) & fpMask);
        }

        __host__ __device__ uint64_t
        replaceTag(uint64_t packed, size_t tagIdx, TagType newTag) const {
            size_t shift = tagIdx * bitsPerTag;
            uint64_t cleared = packed & ~(static_cast<uint64_t>(fpMask) << shift);
            return cleared | (static_cast<uint64_t>(newTag) << shift);
        }

        __forceinline__ __device__ int32_t findSlot(TagType tag) {
            const uint32_t startSlot = tag & (bucketSize - 1);
            const size_t startAtomicIdx = startSlot / tagsPerAtomic;

            if constexpr (atomicCount >= 2) {
                // round down to the nearest even number
                const size_t startPairIdx = startAtomicIdx & ~1;

                for (size_t i = 0; i < atomicCount / 2; i++) {
                    const size_t pairIdx = (startPairIdx + i * 2) & (atomicCount - 1);

                    const auto vec =
                        __ldg(reinterpret_cast<const ulonglong2*>(&packedTags[pairIdx]));
                    const uint64_t loaded[2] = {vec.x, vec.y};

                    _Pragma("unroll")
                    for (size_t k = 0; k < 2; ++k) {
                        const size_t currentAtomicIdx = pairIdx + k;
                        const auto packed = loaded[k];

                        for (size_t j = 0; j < tagsPerAtomic; ++j) {
                            if (extractTag(packed, j) == tag) {
                                return static_cast<int32_t>(currentAtomicIdx * tagsPerAtomic + j);
                            }
                        }
                    }
                }
            } else {
                // just check the single atomic
                const auto packed = packedTags[0].load(cuda::memory_order_relaxed);
                for (size_t j = 0; j < tagsPerAtomic; ++j) {
                    if (extractTag(packed, j) == tag) {
                        return static_cast<int32_t>(j);
                    }
                }
            }

            return -1;
        }

        __device__ bool contains(TagType tag) {
            return findSlot(tag) != -1;
        }
    };

   private:
    size_t numBuckets;
    Bucket* d_buckets;
    cuda::std::atomic<size_t>* d_numOccupied{};
    size_t h_numOccupied = 0;

   public:
    template <typename H>
    static __host__ __device__ uint64_t hash64(const H& key) {
        return xxhash::xxhash64(key);
    }

    static __host__ __device__ cuda::std::tuple<size_t, size_t, TagType>
    getCandidateBuckets(const T& key, size_t numBuckets) {
        const uint64_t h = hash64(key);

        // Upper 32 bits for the fingerprint
        const uint32_t h_fp = h >> 32;
        const TagType fp = static_cast<TagType>(h_fp & fpMask) + 1;

        // Lower 32 bits for the bucket indices
        const uint32_t h_bucket = h & 0xFFFFFFFF;
        const size_t i1 = h_bucket & (numBuckets - 1);
        const size_t i2 = getAlternateBucket(i1, fp, numBuckets);

        return {i1, i2, fp};
    }

    static __host__ __device__ size_t
    getAlternateBucket(size_t bucket, TagType fp, size_t numBuckets) {
        return bucket ^ (hash64(fp) & (numBuckets - 1));
    }

    static size_t calculateNumBuckets(size_t capacity) {
        auto requiredBuckets = std::ceil(static_cast<double>(capacity) / bucketSize);

        return nextPowerOfTwo(requiredBuckets);
    }

   public:
    CuckooFilter(const CuckooFilter&) = delete;
    CuckooFilter& operator=(const CuckooFilter&) = delete;

    explicit CuckooFilter(size_t capacity) : numBuckets(calculateNumBuckets(capacity)) {
        CUDA_CALL(cudaMalloc(&d_buckets, numBuckets * sizeof(Bucket)));
        CUDA_CALL(cudaMalloc(&d_numOccupied, sizeof(cuda::std::atomic<size_t>)));

        assert(powerOfTwo(numBuckets) && "Number of buckets must be a power of 2");

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
        const size_t numStreams = std::clamp(n / blockSize, size_t(1), size_t(12));

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

        CUDA_CALL(
            cudaMemcpy(&h_numOccupied, d_numOccupied, sizeof(size_t), cudaMemcpyDeviceToHost)
        );

        return h_numOccupied;
    }

    size_t insertManySorted(const T* keys, const size_t n) {
        T* d_keys;
        PackedTagType* d_packedTags;

        CUDA_CALL(cudaMalloc(&d_keys, n * sizeof(T)));
        CUDA_CALL(cudaMalloc(&d_packedTags, n * sizeof(PackedTagType)));

        CUDA_CALL(cudaMemcpy(d_keys, keys, n * sizeof(T), cudaMemcpyHostToDevice));

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

        insertKernelSorted<Config>
            <<<numBlocks, blockSize>>>(d_keys, d_packedTags, n, getDeviceView());

        CUDA_CALL(cudaDeviceSynchronize());

        CUDA_CALL(cudaFree(d_keys));
        CUDA_CALL(cudaFree(d_packedTags));

        CUDA_CALL(
            cudaMemcpy(&h_numOccupied, d_numOccupied, sizeof(size_t), cudaMemcpyDeviceToHost)
        );

        return h_numOccupied;
    }

    void containsMany(const T* d_keys, const size_t n, bool* d_output) {
        const size_t numStreams = std::clamp(n / blockSize, size_t(1), size_t(12));
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
                    d_keys + offset, d_output + offset, currentChunkSize, getDeviceView()
                );
            }
        }

        CUDA_CALL(cudaDeviceSynchronize());

        for (auto& stream : streams) {
            CUDA_CALL(cudaStreamDestroy(stream));
        }
    }

    size_t deleteMany(const T* d_keys, const size_t n, bool* d_output = nullptr) {
        const size_t numStreams = std::clamp(n / blockSize, size_t(1), size_t(12));
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
                bool* outputPtr = d_output != nullptr ? d_output + offset : nullptr;
                deleteKernel<Config><<<numBlocks, blockSize, 0, streams[i]>>>(
                    d_keys + offset, outputPtr, currentChunkSize, getDeviceView()
                );
            }
        }

        CUDA_CALL(cudaDeviceSynchronize());

        for (auto& stream : streams) {
            CUDA_CALL(cudaStreamDestroy(stream));
        }

        CUDA_CALL(
            cudaMemcpy(&h_numOccupied, d_numOccupied, sizeof(size_t), cudaMemcpyDeviceToHost)
        );

        return h_numOccupied;
    }

#ifdef CUCKOO_FILTER_HAS_THRUST
    size_t insertMany(const thrust::device_vector<T>& d_keys) {
        return insertMany(thrust::raw_pointer_cast(d_keys.data()), d_keys.size());
    }

    size_t insertManySorted(const thrust::device_vector<T>& d_keys) {
        return insertManySorted(thrust::raw_pointer_cast(d_keys.data()), d_keys.size());
    }

    void
    containsMany(const thrust::device_vector<T>& d_keys, thrust::device_vector<bool>& d_output) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        containsMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            thrust::raw_pointer_cast(d_output.data())
        );
    }

    void
    containsMany(const thrust::device_vector<T>& d_keys, thrust::device_vector<uint8_t>& d_output) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        containsMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))
        );
    }

    size_t
    deleteMany(const thrust::device_vector<T>& d_keys, thrust::device_vector<bool>& d_output) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        return deleteMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            thrust::raw_pointer_cast(d_output.data())
        );
    }

    size_t
    deleteMany(const thrust::device_vector<T>& d_keys, thrust::device_vector<uint8_t>& d_output) {
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
        return deleteMany(thrust::raw_pointer_cast(d_keys.data()), d_keys.size(), nullptr);
    }
#endif  // CUCKOO_FILTER_HAS_THRUST

    void clear() {
        CUDA_CALL(cudaMemset(d_buckets, 0, numBuckets * sizeof(Bucket)));
        CUDA_CALL(cudaMemset(d_numOccupied, 0, sizeof(cuda::std::atomic<size_t>)));
        h_numOccupied = 0;
    }

    float loadFactor() {
        CUDA_CALL(
            cudaMemcpy(&h_numOccupied, d_numOccupied, sizeof(size_t), cudaMemcpyDeviceToHost)
        );
        return static_cast<float>(h_numOccupied) / (numBuckets * bucketSize);
    }

    size_t capacity() {
        return numBuckets * bucketSize;
    }

    [[nodiscard]] size_t getNumBuckets() const {
        return numBuckets;
    }

    [[nodiscard]] size_t sizeInBytes() const {
        return numBuckets * sizeof(Bucket);
    }

    size_t countOccupiedSlots() {
        std::vector<Bucket> h_buckets(numBuckets);

        CUDA_CALL(cudaMemcpy(
            h_buckets.data(), d_buckets, numBuckets * sizeof(Bucket), cudaMemcpyDeviceToHost
        ));

        size_t occupiedCount = 0;

        for (size_t bucketIdx = 0; bucketIdx < numBuckets; ++bucketIdx) {
            const Bucket& bucket = h_buckets[bucketIdx];

            for (size_t atomicIdx = 0; atomicIdx < Bucket::atomicCount; ++atomicIdx) {
                uint64_t packed = reinterpret_cast<const uint64_t&>(bucket.packedTags[atomicIdx]);

                for (size_t tagIdx = 0; tagIdx < Bucket::tagsPerAtomic; ++tagIdx) {
                    auto tag = bucket.extractTag(packed, tagIdx);

                    if (tag != EMPTY) {
                        occupiedCount++;
                    }
                }
            }
        }

        return occupiedCount;
    }

    struct DeviceView {
        Bucket* d_buckets;
        cuda::std::atomic<size_t>* d_numOccupied;
        size_t numBuckets;

        __device__ bool tryRemoveAtBucket(size_t bucketIdx, TagType tag) {
            Bucket& bucket = d_buckets[bucketIdx];

            int32_t slot = bucket.findSlot(tag);
            if (slot == -1) {
                return false;
            }

            size_t atomicIdx = slot / Bucket::tagsPerAtomic;
            size_t tagIdx = slot & (Bucket::tagsPerAtomic - 1);

            auto expected = bucket.packedTags[atomicIdx].load(cuda::memory_order_relaxed);

            do {
                // Verify the tag is still there
                if (bucket.extractTag(expected, tagIdx) != tag) {
                    return false;
                }

                auto desired = bucket.replaceTag(expected, tagIdx, EMPTY);

                if (bucket.packedTags[atomicIdx].compare_exchange_weak(
                        expected, desired, cuda::memory_order_relaxed, cuda::memory_order_relaxed
                    )) {
                    return true;
                }
            } while (bucket.extractTag(expected, tagIdx) == tag);

            return false;  // Tag was removed by another thread during our CAS attempts
        }

        __device__ bool tryInsertAtBucket(size_t bucketIdx, TagType tag) {
            Bucket& bucket = d_buckets[bucketIdx];
            const uint32_t startIdx = tag & (bucketSize - 1);
            const size_t startAtomicIdx = startIdx / Bucket::tagsPerAtomic;

            for (size_t i = 0; i < Bucket::atomicCount; ++i) {
                const size_t atomicIdx = (startAtomicIdx + i) & (Bucket::atomicCount - 1);
                auto expected = bucket.packedTags[atomicIdx].load(cuda::memory_order_relaxed);

                bool retryWord;
                do {
                    retryWord = false;

                    for (size_t j = 0; j < Bucket::tagsPerAtomic; ++j) {
                        if (bucket.extractTag(expected, j) == EMPTY) {
                            auto desired = bucket.replaceTag(expected, j, tag);

                            if (bucket.packedTags[atomicIdx].compare_exchange_weak(
                                    expected,
                                    desired,
                                    cuda::memory_order_relaxed,
                                    cuda::memory_order_relaxed
                                )) {
                                return true;
                            } else {
                                retryWord = true;
                                break;
                            }
                        }
                    }
                } while (retryWord);
            }
            return false;
        }

        __device__ bool insertWithEviction(TagType fp, size_t startBucket) {
            TagType currentFp = fp;
            size_t currentBucket = startBucket;

            for (size_t evictions = 0; evictions < maxEvictions; ++evictions) {
                auto evictSlot = (currentFp + evictions * 7) & (bucketSize - 1);

                size_t atomicIdx = evictSlot / Bucket::tagsPerAtomic;
                size_t tagIdx = evictSlot & (Bucket::tagsPerAtomic - 1);

                Bucket& bucket = d_buckets[currentBucket];
                auto expected = bucket.packedTags[atomicIdx].load(cuda::memory_order_relaxed);
                uint64_t desired;
                TagType evictedFp;

                do {
                    evictedFp = bucket.extractTag(expected, tagIdx);
                    desired = bucket.replaceTag(expected, tagIdx, currentFp);
                } while (!bucket.packedTags[atomicIdx].compare_exchange_weak(
                    expected, desired, cuda::memory_order_relaxed, cuda::memory_order_relaxed
                ));

                currentFp = evictedFp;
                currentBucket = getAlternateBucket(currentBucket, evictedFp, numBuckets);

                if (tryInsertAtBucket(currentBucket, currentFp)) {
                    return true;
                }
            }
            return false;
        }

        __device__ bool insertWithEvictionBFS(TagType fp, size_t startBucket) {
            constexpr size_t numCandidates = std::min(8UL, bucketSize / 2);

            Bucket& bucket = d_buckets[startBucket];

            for (size_t i = 0; i < numCandidates; ++i) {
                size_t slot = (fp + i * 7) & (bucketSize - 1);
                size_t atomicIdx = slot / Bucket::tagsPerAtomic;
                size_t tagIdx = slot & (Bucket::tagsPerAtomic - 1);

                auto packed = bucket.packedTags[atomicIdx].load(cuda::memory_order_relaxed);
                TagType candidateFp = bucket.extractTag(packed, tagIdx);

                if (candidateFp == EMPTY) {
                    if (tryInsertAtBucket(startBucket, fp)) {
                        return true;
                    }
                    continue;
                }

                size_t altBucket =
                    CuckooFilter::getAlternateBucket(startBucket, candidateFp, numBuckets);
                if (tryInsertAtBucket(altBucket, candidateFp)) {
                    // Successfully inserted the evicted tag at its alternate location
                    // Now atomically swap in our tag at the original location
                    auto expected = bucket.packedTags[atomicIdx].load(cuda::memory_order_relaxed);

                    // Verify the tag is still there and try to replace it
                    if (bucket.extractTag(expected, tagIdx) == candidateFp) {
                        auto desired = bucket.replaceTag(expected, tagIdx, fp);

                        if (bucket.packedTags[atomicIdx].compare_exchange_weak(
                                expected,
                                desired,
                                cuda::memory_order_relaxed,
                                cuda::memory_order_relaxed
                            )) {
                            return true;
                        }
                    }

                    // Failed to swap, clean up the tag we inserted to avoid duplicates
                    tryRemoveAtBucket(altBucket, candidateFp);
                }
            }

            // fall back to greedy DFS
            return insertWithEviction(fp, startBucket);
        }

        __device__ bool insert(const T& key) {
            auto [i1, i2, fp] = getCandidateBuckets(key, numBuckets);

            if (tryInsertAtBucket(i1, fp) || tryInsertAtBucket(i2, fp)) {
                return true;
            }

            auto startBucket = (fp & 1) == 0 ? i1 : i2;

            return insertWithEviction(fp, startBucket);
        }

        // FIXME: Somehow this isn't guaranteed to find all existing keys?
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
    using BlockReduce = cub::BlockReduce<int32_t, Config::blockSize>;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    auto idx = globalThreadId();

    int32_t success = 0;

    if (idx < n) {
        success = view.insert(keys[idx]);
    }

    int32_t blockSuccessSum = BlockReduce(tempStorage).Sum(success);
    __syncthreads();

    if (threadIdx.x == 0) {
        if (blockSuccessSum > 0) {
            view.d_numOccupied->fetch_add(blockSuccessSum, cuda::memory_order_relaxed);
        }
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
    using BlockReduce = cub::BlockReduce<int32_t, Config::blockSize>;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    auto idx = globalThreadId();

    int32_t success = 0;
    if (idx < n) {
        success = view.remove(keys[idx]);

        if (output != nullptr) {
            output[idx] = success;
        }
    }

    int32_t blockSum = BlockReduce(tempStorage).Sum(success);

    if (threadIdx.x == 0 && blockSum > 0) {
        view.d_numOccupied->fetch_sub(blockSum, cuda::memory_order_relaxed);
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

    packedTags[idx] =
        (static_cast<PackedTagType>(i1) << bitsPerTag) | static_cast<PackedTagType>(fp);
}

template <typename Config>
__global__ void insertKernelSorted(
    const typename Config::KeyType* keys,
    const typename CuckooFilter<Config>::PackedTagType* packedTags,
    size_t n,
    typename CuckooFilter<Config>::DeviceView view
) {
    using BlockReduce = cub::BlockReduce<int, Config::blockSize>;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    size_t idx = globalThreadId();

    using Filter = CuckooFilter<Config>;
    using TagType = typename Filter::TagType;
    using PackedTagType = typename Filter::PackedTagType;

    constexpr size_t bitsPerTag = Config::bitsPerTag;
    constexpr TagType fpMask = (1ULL << bitsPerTag) - 1;

    int32_t success = 0;
    if (idx < n) {
        PackedTagType packedTag = packedTags[idx];
        size_t primaryBucket = packedTag >> bitsPerTag;
        auto fp = static_cast<TagType>(packedTag & fpMask);

        if (view.tryInsertAtBucket(primaryBucket, fp)) {
            success = 1;
        } else {
            size_t secondaryBucket = Filter::getAlternateBucket(primaryBucket, fp, view.numBuckets);

            if (view.tryInsertAtBucket(secondaryBucket, fp)) {
                success = 1;
            } else {
                auto startBucket = (fp & 1) == 0 ? primaryBucket : secondaryBucket;
                success = view.insertWithEviction(fp, startBucket);
            }
        }
    }

    int32_t blockSum = BlockReduce(tempStorage).Sum(success);

    if (threadIdx.x == 0 && blockSum > 0) {
        view.d_numOccupied->fetch_add(blockSum, cuda::memory_order_relaxed);
    }
}
