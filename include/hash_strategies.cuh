#pragma once

#include <cmath>
#include <cstdint>
#include <cuda/std/tuple>
#include "hashutil.cuh"
#include "helpers.cuh"

/**
 * @brief Default XOR-based hashing strategy for cuckoo filters.
 * This uses the traditional partial-key cuckoo hashing with XOR operation
 * to compute alternate bucket indices.
 */
template <typename KeyType, typename TagType, size_t bitsPerTag, size_t bucketSize>
struct XorHashStrategy {
    static constexpr size_t fpMask = (1ULL << bitsPerTag) - 1;

    static constexpr char name[] = "XorHashStrategy";

    template <typename H>
    static __host__ __device__ uint64_t hash64(const H& key) {
        return xxhash::xxhash64(key);
    }

    /**
     * @brief Performs partial-key cuckoo hashing to find the fingerprint and both
     * bucket indices for a given key.
     *
     * We derive the fingerprint and bucket indices from different parts of the key's
     * hash to greatly reduce the number of collisions. The fingerprint value 0 is reserved to
     * indicate an empty slot, so we use 1 if the computed fingerprint is 0.
     *
     * @param key Key to hash
     * @param numBuckets Number of buckets in the filter
     * @return A tuple of (bucket1, bucket2, fingerprint)
     */
    static __host__ __device__ cuda::std::tuple<size_t, size_t, TagType>
    getCandidateBuckets(const KeyType& key, size_t numBuckets) {
        const uint64_t h = hash64(key);

        // Upper 32 bits for the fingerprint
        const uint32_t h_fp = h >> 32;
        const uint32_t masked = h_fp & fpMask;

        const auto fp = TagType(masked == 0 ? 1 : masked);

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

    /**
     * @brief The number of buckets is enforced to be a power of two in order to allow for efficient
     * modulo on the bucket indices with XOR-based hashing.
     */
    static size_t calculateNumBuckets(size_t capacity) {
        auto requiredBuckets = std::ceil(static_cast<double>(capacity) / bucketSize);
        return nextPowerOfTwo(requiredBuckets);
    }
};

/**
 * @brief Addition/Subtraction-based hashing strategy for cuckoo filters (ASCF).
 * This implements the scheme from "Additive and Subtractive Cuckoo Filters" paper
 * which uses ADD/SUB operations instead of XOR, allowing for non-power-of-two
 * bucket counts and better space efficiency.
 *
 * The filter is conceptually divided into two blocks of equal size.
 * - Block 0: buckets [0, numBuckets/2)
 * - Block 1: buckets [numBuckets/2, numBuckets)
 */
template <typename KeyType, typename TagType, size_t bitsPerTag, size_t bucketSize>
struct AddSubHashStrategy {
    static constexpr size_t fpMask = (1ULL << bitsPerTag) - 1;

    static constexpr char name[] = "AddSubHashStrategy";

    template <typename H>
    static __host__ __device__ uint64_t hash64(const H& key) {
        return xxhash::xxhash64(key);
    }

    /**
     * @brief Computes fingerprint and candidate buckets using ADD/SUB operations.
     *
     * The first bucket is in block 0, computed from the hash.
     * The second bucket is in block 1, computed by adding the fingerprint hash
     * to the first bucket index (with wraparound within block 1).
     *
     * @param key Key to hash
     * @param numBuckets Total number of buckets (should be even for proper block division)
     * @return A tuple of (bucket1, bucket2, fingerprint)
     */
    static __host__ __device__ cuda::std::tuple<size_t, size_t, TagType>
    getCandidateBuckets(const KeyType& key, size_t numBuckets) {
        const uint64_t h = hash64(key);

        // Upper 32 bits for the fingerprint
        const uint32_t h_fp = h >> 32;
        const uint32_t masked = h_fp & fpMask;
        const auto fp = TagType(masked == 0 ? 1 : masked);

        // Lower 32 bits for bucket index in block 0
        const uint32_t h_bucket = h & 0xFFFFFFFF;
        const size_t bucketsPerBlock = numBuckets / 2;

        const size_t i1 = h_bucket % bucketsPerBlock;
        const size_t i2 = getAlternateBucket(i1, fp, numBuckets);

        return {i1, i2, fp};
    }

    /**
     * @brief Computes alternate bucket using ADD/SUB operations.
     *
     * If current bucket is in block 0, alternate is in block 1 (use ADD).
     * If current bucket is in block 1, alternate is in block 0 (use SUB).
     *
     * @param bucket Current bucket index
     * @param fp Fingerprint value
     * @param numBuckets Total number of buckets
     * @return Alternate bucket index
     */
    static __host__ __device__ size_t
    getAlternateBucket(size_t bucket, TagType fp, size_t numBuckets) {
        const size_t bucketsPerBlock = numBuckets / 2;
        const uint64_t fpHash = hash64(fp) % bucketsPerBlock;

        if (bucket < bucketsPerBlock) {
            return ((bucket + fpHash) % bucketsPerBlock) + bucketsPerBlock;
        } else {
            return (bucket - fpHash) % bucketsPerBlock;
        }
    }

    /**
     * @brief Calculates number of buckets without requiring power of two.
     * The total number is rounded to ensure it's even (for equal block sizes).
     */
    static size_t calculateNumBuckets(size_t capacity) {
        auto requiredBuckets = std::ceil(static_cast<double>(capacity) / bucketSize);

        // Round up to next even number to ensure equal block sizes
        if (static_cast<size_t>(requiredBuckets) % 2 != 0) {
            requiredBuckets += 1;
        }

        return static_cast<size_t>(requiredBuckets);
    }
};