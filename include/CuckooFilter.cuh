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
#include "hash_strategies.cuh"
#include "hashutil.cuh"
#include "helpers.cuh"

#if __has_include(<thrust/device_vector.h>)
    #include <thrust/device_vector.h>
    #define CUCKOO_FILTER_HAS_THRUST 1
#endif

template <
    typename T,
    size_t bitsPerTag_,
    size_t maxEvictions_ = 500,
    size_t blockSize_ = 256,
    size_t bucketSize_ = 16,
    template <typename, typename, size_t, size_t> class AltBucketPolicy_ = XorAltBucketPolicy>
struct CuckooConfig {
    using KeyType = T;
    static constexpr size_t bitsPerTag = bitsPerTag_;
    static constexpr size_t maxEvictions = maxEvictions_;
    static constexpr size_t blockSize = blockSize_;
    static constexpr size_t bucketSize = bucketSize_;

    using TagType = typename std::conditional<
        bitsPerTag <= 8,
        uint8_t,
        typename std::conditional<bitsPerTag <= 16, uint16_t, uint32_t>::type>::type;

    using AltBucketPolicy = AltBucketPolicy_<KeyType, TagType, bitsPerTag, bucketSize_>;
};

template <typename Config>
class CuckooFilter;

template <typename Config>
__global__ void
insertKernel(const typename Config::KeyType* keys, size_t n, CuckooFilter<Config>* filter);

template <typename Config>
__global__ void insertKernelSorted(
    const typename Config::KeyType* keys,
    const typename CuckooFilter<Config>::PackedTagType* packedTags,
    size_t n,
    CuckooFilter<Config>* filter
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
    CuckooFilter<Config>* filter
);

template <typename Config>
__global__ void deleteKernel(
    const typename Config::KeyType* keys,
    bool* output,
    size_t n,
    CuckooFilter<Config>* filter
);

template <typename Config>
struct CuckooFilter {
    using T = typename Config::KeyType;
    static constexpr size_t bitsPerTag = Config::bitsPerTag;

    using TagType = typename Config::TagType;
    using AltBucketPolicy = typename Config::AltBucketPolicy;

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

    using PackedTagType = typename std::conditional<bitsPerTag <= 8, uint32_t, uint64_t>::type;

    /**
     * @brief This is used by the sorted insert kernel to store the fingerprint and primary bucket
     * index in a compact format that allows you to sort them directly since the bucket index lives
     * in the upper bits.
     *
     */
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

    /**
     * @brief Bucket structure that holds the fingerprint and tags for a given bucket.
     * The bucket is divided into words of 64 bits each, where each word is made up of at least two
     * fingerprints. How many depends on the fingerprint size specified by the user of the filter.
     *
     * This optimisation allows us to avoid having to perform atomic operations on every fingerprint
     * in the bucket, the extra computational overhead is negligible.
     *
     * For efficiency reasons, the number of fingerprints per word is enforced to be a power of 2,
     * same goes for the total number of fingerprints in the bucket.
     *
     */
    struct Bucket {
        static_assert(powerOfTwo(bitsPerTag), "bitsPerTag must be a power of 2");
        static constexpr size_t tagsPerWord = sizeof(uint64_t) / sizeof(TagType);
        static_assert(
            bucketSize % tagsPerWord == 0,
            "bucketSize must be divisible by tagsPerAtomic"
        );
        static_assert(powerOfTwo(tagsPerWord), "tagsPerAtomic must be a power of 2");

        static constexpr size_t wordCount = bucketSize / tagsPerWord;
        static_assert(powerOfTwo(wordCount), "atomicCount must be a power of 2");

        cuda::std::atomic<uint64_t> packedTags[wordCount];

        __host__ __device__ TagType extractTag(uint64_t packed, size_t tagIdx) const {
            return static_cast<TagType>((packed >> (tagIdx * bitsPerTag)) & fpMask);
        }

        __host__ __device__ uint64_t
        replaceTag(uint64_t packed, size_t tagIdx, TagType newTag) const {
            size_t shift = tagIdx * bitsPerTag;
            uint64_t cleared = packed & ~(static_cast<uint64_t>(fpMask) << shift);
            return cleared | (static_cast<uint64_t>(newTag) << shift);
        }

        /**
         * @brief Finds the index of a tag in the bucket. Because we can guarantee that no threads
         * will try to insert into the bucket while doing so, we can make use of non-atomic
         * vectorised loads to speed up the search.
         *
         * @param tag Tag to search for
         * @return Index of the tag in the bucket, or -1 if not found
         */
        __forceinline__ __device__ int32_t findSlot(TagType tag) {
            const uint32_t startSlot = tag & (bucketSize - 1);
            const size_t startAtomicIdx = startSlot / tagsPerWord;

            if constexpr (wordCount >= 2) {
                // round down to the nearest even number
                const size_t startPairIdx = startAtomicIdx & ~1;

                for (size_t i = 0; i < wordCount / 2; i++) {
                    const size_t pairIdx = (startPairIdx + i * 2) & (wordCount - 1);

                    const auto vec =
                        __ldg(reinterpret_cast<const ulonglong2*>(&packedTags[pairIdx]));
                    const uint64_t loaded[2] = {vec.x, vec.y};

                    _Pragma("unroll")
                    for (size_t k = 0; k < 2; ++k) {
                        const size_t currentAtomicIdx = pairIdx + k;
                        const auto packed = loaded[k];

                        for (size_t j = 0; j < tagsPerWord; ++j) {
                            if (extractTag(packed, j) == tag) {
                                return static_cast<int32_t>(currentAtomicIdx * tagsPerWord + j);
                            }
                        }
                    }
                }
            } else {
                // just check the single atomic
                const auto packed = reinterpret_cast<const uint64_t&>(packedTags[0]);
                for (size_t j = 0; j < tagsPerWord; ++j) {
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

    size_t numBuckets;
    Bucket* d_buckets;
    cuda::std::atomic<size_t>* d_numOccupied{};

    size_t h_numOccupied = 0;

    template <typename H>
    static __host__ __device__ uint64_t hash64(const H& key) {
        return AltBucketPolicy::hash64(key);
    }

    static __host__ __device__ cuda::std::tuple<size_t, size_t, TagType>
    getCandidateBuckets(const T& key, size_t numBuckets) {
        return AltBucketPolicy::getCandidateBuckets(key, numBuckets);
    }

    static __host__ __device__ size_t
    getAlternateBucket(size_t bucket, TagType fp, size_t numBuckets) {
        return AltBucketPolicy::getAlternateBucket(bucket, fp, numBuckets);
    }

    /**
     * @brief The number of buckets is enforced to be a power of two in order to allow for efficient
     * modulo on the bucket indices
     */
    static size_t calculateNumBuckets(size_t capacity) {
        return AltBucketPolicy::calculateNumBuckets(capacity);
    }

    CuckooFilter(const CuckooFilter&) = delete;
    CuckooFilter& operator=(const CuckooFilter&) = delete;

    explicit CuckooFilter(size_t capacity) : numBuckets(calculateNumBuckets(capacity)) {
        CUDA_CALL(cudaMalloc(&d_buckets, numBuckets * sizeof(Bucket)));
        CUDA_CALL(cudaMalloc(&d_numOccupied, sizeof(cuda::std::atomic<size_t>)));

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

    size_t insertMany(const T* d_keys, const size_t n, cudaStream_t stream = {}) {
        size_t numBlocks = SDIV(n, blockSize);
        insertKernel<Config><<<numBlocks, blockSize, 0, stream>>>(d_keys, n, this);

        CUDA_CALL(cudaStreamSynchronize(stream));

        return occupiedSlots();
    }

    /**
     * @brief This pre-sorts the input keys based on the primary bucket index to allow for coalesced
     * memory access when you later insert them into the filter.
     *
     * @param d_keys Pointer to device memory array of keys to insert
     * @param n Number of keys to insert
     * @return size_t Updated number of occupied slots in the filter
     */
    size_t insertManySorted(const T* d_keys, const size_t n, cudaStream_t stream = {}) {
        PackedTagType* d_packedTags;

        CUDA_CALL(cudaMalloc(&d_packedTags, n * sizeof(PackedTagType)));

        size_t numBlocks = SDIV(n, blockSize);

        computePackedTagsKernel<Config>
            <<<numBlocks, blockSize, 0, stream>>>(d_keys, d_packedTags, n, numBuckets);

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
            <<<numBlocks, blockSize, 0, stream>>>(d_keys, d_packedTags, n, this);

        CUDA_CALL(cudaStreamSynchronize(stream));

        CUDA_CALL(cudaFree(d_packedTags));

        return occupiedSlots();
    }

    void containsMany(const T* d_keys, const size_t n, bool* d_output, cudaStream_t stream = {}) {
        size_t numBlocks = SDIV(n, blockSize);
        containsKernel<Config><<<numBlocks, blockSize, 0, stream>>>(d_keys, d_output, n, this);

        CUDA_CALL(cudaStreamSynchronize(stream));
    }

    /**
     * @brief Tries to remove a set of keys from the filter. If the key is not present, it is
     * ignored.
     *
     * @param d_keys Pointer to the array of keys to remove
     * @param n Number of keys to remove
     * @param d_output Optional pointer to an output array indicating the success of each key
     * removal
     * @param stream CUDA stream to use for the operation
     * @return size_t Updated number of occupied slots in the filter
     */
    size_t deleteMany(
        const T* d_keys,
        const size_t n,
        bool* d_output = nullptr,
        cudaStream_t stream = {}
    ) {
        size_t numBlocks = SDIV(n, blockSize);
        deleteKernel<Config><<<numBlocks, blockSize, 0, stream>>>(d_keys, d_output, n, this);

        CUDA_CALL(cudaStreamSynchronize(stream));

        return occupiedSlots();
    }

#ifdef CUCKOO_FILTER_HAS_THRUST
    size_t insertMany(const thrust::device_vector<T>& d_keys, cudaStream_t stream = {}) {
        return insertMany(thrust::raw_pointer_cast(d_keys.data()), d_keys.size(), stream);
    }

    size_t insertManySorted(const thrust::device_vector<T>& d_keys, cudaStream_t stream = {}) {
        return insertManySorted(thrust::raw_pointer_cast(d_keys.data()), d_keys.size(), stream);
    }

    void containsMany(
        const thrust::device_vector<T>& d_keys,
        thrust::device_vector<bool>& d_output,
        cudaStream_t stream = {}
    ) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        containsMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            thrust::raw_pointer_cast(d_output.data()),
            stream
        );
    }

    void containsMany(
        const thrust::device_vector<T>& d_keys,
        thrust::device_vector<uint8_t>& d_output,
        cudaStream_t stream = {}
    ) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        containsMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data())),
            stream
        );
    }

    size_t deleteMany(
        const thrust::device_vector<T>& d_keys,
        thrust::device_vector<bool>& d_output,
        cudaStream_t stream = {}
    ) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        return deleteMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            thrust::raw_pointer_cast(d_output.data()),
            stream
        );
    }

    size_t deleteMany(
        const thrust::device_vector<T>& d_keys,
        thrust::device_vector<uint8_t>& d_output,
        cudaStream_t stream = {}
    ) {
        if (d_output.size() != d_keys.size()) {
            d_output.resize(d_keys.size());
        }
        return deleteMany(
            thrust::raw_pointer_cast(d_keys.data()),
            d_keys.size(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data())),
            stream
        );
    }

    size_t deleteMany(const thrust::device_vector<T>& d_keys, cudaStream_t stream = {}) {
        return deleteMany(thrust::raw_pointer_cast(d_keys.data()), d_keys.size(), nullptr, stream);
    }
#endif  // CUCKOO_FILTER_HAS_THRUST

    void clear() {
        CUDA_CALL(cudaMemset(d_buckets, 0, numBuckets * sizeof(Bucket)));
        CUDA_CALL(cudaMemset(d_numOccupied, 0, sizeof(cuda::std::atomic<size_t>)));
        h_numOccupied = 0;
    }

    [[nodiscard]] float loadFactor() {
        return static_cast<float>(occupiedSlots()) / (numBuckets * bucketSize);
    }

    size_t occupiedSlots() {
        CUDA_CALL(
            cudaMemcpy(&h_numOccupied, d_numOccupied, sizeof(size_t), cudaMemcpyDeviceToHost)
        );
        return h_numOccupied;
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

            for (size_t atomicIdx = 0; atomicIdx < Bucket::wordCount; ++atomicIdx) {
                uint64_t packed = reinterpret_cast<const uint64_t&>(bucket.packedTags[atomicIdx]);

                for (size_t tagIdx = 0; tagIdx < Bucket::tagsPerWord; ++tagIdx) {
                    auto tag = bucket.extractTag(packed, tagIdx);

                    if (tag != EMPTY) {
                        occupiedCount++;
                    }
                }
            }
        }

        return occupiedCount;
    }

    /**
     * @brief Attempt to remove a single instance of a fingerprint from a bucket
     *
     * Scans the atomic words that make up the bucket and attempts a CAS on each matching
     * tag position until one removal succeeds. This allows multiple concurrent
     * deleters to remove distinct copies when a bucket contains several identical
     * fingerprints instead of all trying to clear the same slot
     *
     * The function is lock-free and uses per-word compare-and-swap operations.
     * It does NOT update any global occupancy counter, the caller is responsible
     * for decrementing d_numOccupied if appropriate
     *
     * @param bucketIdx Index of the bucket to search.
     * @param tag       Fingerprint value to remove (must not be EMPTY).
     * @return true if a single instance of `tag` was removed from the bucket;
     *         false if no matching tag remained (or another thread removed the
     *         last matching instance before this call could succeed).
     */
    __device__ bool tryRemoveAtBucket(size_t bucketIdx, TagType tag) {
        Bucket& bucket = d_buckets[bucketIdx];

        const uint32_t startSlot = tag & (bucketSize - 1);
        const size_t startWord = startSlot / Bucket::tagsPerWord;

        for (size_t i = 0; i < Bucket::wordCount; ++i) {
            const size_t currIdx = (startWord + i) & (Bucket::wordCount - 1);

            while (true) {
                uint64_t expected = bucket.packedTags[currIdx].load(cuda::memory_order_relaxed);

                bool anyMatch = false;
                for (size_t tagIdx = 0; tagIdx < Bucket::tagsPerWord; ++tagIdx) {
                    if (bucket.extractTag(expected, tagIdx) == tag) {
                        anyMatch = true;

                        uint64_t desired = bucket.replaceTag(expected, tagIdx, EMPTY);

                        if (bucket.packedTags[currIdx].compare_exchange_weak(
                                expected,
                                desired,
                                cuda::memory_order_relaxed,
                                cuda::memory_order_relaxed
                            )) {
                            return true;
                        }
                        break;
                    }
                }

                if (!anyMatch) {
                    break;
                }
            }
        }

        return false;
    }

    __device__ bool tryInsertAtBucket(size_t bucketIdx, TagType tag) {
        Bucket& bucket = d_buckets[bucketIdx];
        const uint32_t startIdx = tag & (bucketSize - 1);
        const size_t startWord = startIdx / Bucket::tagsPerWord;

        for (size_t i = 0; i < Bucket::wordCount; ++i) {
            const size_t currWord = (startWord + i) & (Bucket::wordCount - 1);
            auto expected = bucket.packedTags[currWord].load(cuda::memory_order_relaxed);

            bool retryWord;
            do {
                retryWord = false;

                for (size_t j = 0; j < Bucket::tagsPerWord; ++j) {
                    if (bucket.extractTag(expected, j) == EMPTY) {
                        auto desired = bucket.replaceTag(expected, j, tag);

                        if (bucket.packedTags[currWord].compare_exchange_strong(
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

    /**
     * @brief Inserts a fingerprint into the filter by evicting existing fingerprints. The
     * thread first picks a pseudo-random target to replace with the new fingerprint. Then it
     * tries to insert the evicted fingerprint into its alternate bucket. This process is
     * repeated until either a fingerprint is inserted into an empty slot or the maximum number
     * of evictions is reached.
     *
     * @param fp Fingerprint to insert
     * @param startBucket Index of the bucket to start the search from
     * @return true if the insertion was successful, false otherwise
     */
    __device__ bool insertWithEviction(TagType fp, size_t startBucket) {
        TagType currentFp = fp;
        size_t currentBucket = startBucket;

        for (size_t evictions = 0; evictions < maxEvictions; ++evictions) {
            auto evictSlot = (currentFp + evictions * 7) & (bucketSize - 1);

            size_t evictWord = evictSlot / Bucket::tagsPerWord;
            size_t evictTagIdx = evictSlot & (Bucket::tagsPerWord - 1);

            Bucket& bucket = d_buckets[currentBucket];
            auto expected = bucket.packedTags[evictWord].load(cuda::memory_order_relaxed);
            uint64_t desired;
            TagType evictedFp;

            do {
                evictedFp = bucket.extractTag(expected, evictTagIdx);
                desired = bucket.replaceTag(expected, evictTagIdx, currentFp);
            } while (!bucket.packedTags[evictWord].compare_exchange_strong(
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

    /**
     * @brief Inserts a fingerprint into the filter by evicting existing fingerprints using a
     * very shallow breadth-first search. It tries multiple random random eviction targets and
     * checks if there is an empty slot in their alternate buckets. If there is, it inserts the
     * existing fingerprint into its alternate bucket and places the new fingerprint in the
     * original location.
     *
     * @param fp Fingerprint to insert
     * @param startBucket Index of the bucket to start the search from
     * @return true if the insertion was successful, false otherwise
     */
    __device__ bool insertWithEvictionBFS(TagType fp, size_t startBucket) {
        constexpr size_t numCandidates = std::min(8UL, bucketSize / 2);

        Bucket& bucket = d_buckets[startBucket];

        for (size_t i = 0; i < numCandidates; ++i) {
            size_t slot = (fp + i * 7) & (bucketSize - 1);
            size_t atomicIdx = slot / Bucket::tagsPerWord;
            size_t tagIdx = slot & (Bucket::tagsPerWord - 1);

            auto packed = bucket.packedTags[atomicIdx].load(cuda::memory_order_relaxed);
            TagType candidateFp = bucket.extractTag(packed, tagIdx);

            if (candidateFp == EMPTY) {
                if (tryInsertAtBucket(startBucket, fp)) {
                    return true;
                }
                continue;
            }

            size_t altBucket = getAlternateBucket(startBucket, candidateFp, numBuckets);
            if (tryInsertAtBucket(altBucket, candidateFp)) {
                // Successfully inserted the evicted tag at its alternate location
                // Now atomically swap in our tag at the original location
                auto expected = bucket.packedTags[atomicIdx].load(cuda::memory_order_relaxed);

                // Verify the tag is still there and try to replace it
                if (bucket.extractTag(expected, tagIdx) == candidateFp) {
                    auto desired = bucket.replaceTag(expected, tagIdx, fp);

                    if (bucket.packedTags[atomicIdx].compare_exchange_strong(
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

        return insertWithEvictionBFS(fp, startBucket);
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

template <typename Config>
__global__ void
insertKernel(const typename Config::KeyType* keys, size_t n, CuckooFilter<Config>* filter) {
    using BlockReduce = cub::BlockReduce<int32_t, Config::blockSize>;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    auto idx = globalThreadId();

    int32_t success = 0;

    if (idx < n) {
        success = filter->insert(keys[idx]);
    }

    int32_t blockSuccessSum = BlockReduce(tempStorage).Sum(success);
    __syncthreads();

    if (threadIdx.x == 0) {
        if (blockSuccessSum > 0) {
            filter->d_numOccupied->fetch_add(blockSuccessSum, cuda::memory_order_relaxed);
        }
    }
}

template <typename Config>
__global__ void containsKernel(
    const typename Config::KeyType* keys,
    bool* output,
    size_t n,
    CuckooFilter<Config>* filter
) {
    auto idx = globalThreadId();

    if (idx < n) {
        output[idx] = filter->contains(keys[idx]);
    }
}

template <typename Config>
__global__ void deleteKernel(
    const typename Config::KeyType* keys,
    bool* output,
    size_t n,
    CuckooFilter<Config>* filter
) {
    using BlockReduce = cub::BlockReduce<int32_t, Config::blockSize>;
    __shared__ typename BlockReduce::TempStorage tempStorage;

    auto idx = globalThreadId();

    int32_t success = 0;
    if (idx < n) {
        success = filter->remove(keys[idx]);

        if (output != nullptr) {
            output[idx] = success;
        }
    }

    int32_t blockSum = BlockReduce(tempStorage).Sum(success);

    if (threadIdx.x == 0 && blockSum > 0) {
        filter->d_numOccupied->fetch_sub(blockSum, cuda::memory_order_relaxed);
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
    CuckooFilter<Config>* filter
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

        if (filter->tryInsertAtBucket(primaryBucket, fp)) {
            success = 1;
        } else {
            size_t secondaryBucket =
                Filter::getAlternateBucket(primaryBucket, fp, filter->numBuckets);

            if (filter->tryInsertAtBucket(secondaryBucket, fp)) {
                success = 1;
            } else {
                auto startBucket = (fp & 1) == 0 ? primaryBucket : secondaryBucket;
                success = filter->insertWithEviction(fp, startBucket);
            }
        }
    }

    int32_t blockSum = BlockReduce(tempStorage).Sum(success);

    if (threadIdx.x == 0 && blockSum > 0) {
        filter->d_numOccupied->fetch_add(blockSum, cuda::memory_order_relaxed);
    }
}
