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
struct XorAltBucketPolicy {
    static constexpr size_t fpMask = (1ULL << bitsPerTag) - 1;

    static constexpr bool usesChoiceBit = false;

    static constexpr char name[] = "XorAltBucketPolicy";

    /**
     * @brief Computes a 64-bit hash of the key.
     * @tparam H Type of the key.
     * @param key The key to hash.
     * @return uint64_t The 64-bit hash value.
     */
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

    /**
     * @brief Computes the alternate bucket index using XOR.
     *
     * @param bucket Current bucket index.
     * @param fp Fingerprint.
     * @param numBuckets Total number of buckets (must be power of 2).
     * @return size_t Alternate bucket index.
     */
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
struct AddSubAltBucketPolicy {
    static constexpr size_t fpMask = (1ULL << bitsPerTag) - 1;

    static constexpr bool usesChoiceBit = false;

    static constexpr char name[] = "AddSubAltBucketPolicy";

    /**
     * @brief Computes a 64-bit hash of the key.
     * @tparam H Type of the key.
     * @param key The key to hash.
     * @return uint64_t The 64-bit hash value.
     */
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

/**
 * @brief Offset-based alternate bucket policy from "Smaller and More Flexible Cuckoo Filters"
 * by Schmitz, Zentgraf, and Rahmann.
 *
 * Uses an asymmetric offset with a choice bit:
 * - b' = (b + offset(fp)) mod numBuckets (when choice=0, primary bucket)
 * - b  = (b' - offset(fp)) mod numBuckets (when choice=1, alternate bucket)
 *
 * The choice bit is stored in the MSB of the fingerprint:
 * - Lower (bitsPerTag-1) bits: actual fingerprint
 * - MSB: choice bit (0 = primary, 1 = alternate)
 *
 * This allows non-power-of-two bucket counts with proper symmetry.
 */
template <typename KeyType, typename TagType, size_t bitsPerTag, size_t bucketSize>
struct OffsetAltBucketPolicy {
    static_assert(bitsPerTag >= 2, "bitsPerTag must be at least 2 for choice bit");

    // Choice bit is MSB, fingerprint uses remaining bits
    static constexpr size_t choiceBitPos = bitsPerTag - 1;
    static constexpr TagType choiceBitMask = TagType(1) << choiceBitPos;
    static constexpr TagType pureFpMask = (TagType(1) << choiceBitPos) - 1;
    static constexpr size_t fpMask = (1ULL << bitsPerTag) - 1;

    static constexpr bool usesChoiceBit = true;

    static constexpr char name[] = "OffsetAltBucketPolicy";

    /**
     * @brief Computes a 64-bit hash of the key.
     */
    template <typename H>
    static __host__ __device__ uint64_t hash64(const H& key) {
        return xxhash::xxhash64(key);
    }

    /**
     * @brief Extracts the pure fingerprint (without choice bit).
     */
    static __host__ __device__ TagType getPureFingerprint(TagType fp) {
        return fp & pureFpMask;
    }

    /**
     * @brief Extracts the choice bit (0 or 1).
     */
    static __host__ __device__ TagType getChoiceBit(TagType fp) {
        return (fp >> choiceBitPos) & 1;
    }

    /**
     * @brief Sets the choice bit in a fingerprint.
     */
    static __host__ __device__ TagType setChoiceBit(TagType fp, TagType choice) {
        return (fp & pureFpMask) | (choice << choiceBitPos);
    }

    /**
     * @brief Computes a non-zero offset from a fingerprint (uses pure fp).
     *
     * @param pureFp Pure fingerprint (without choice bit)
     * @param numBuckets Total number of buckets
     * @return Non-zero offset value in [1, numBuckets-1]
     */
    static __host__ __device__ size_t computeOffset(TagType pureFp, size_t numBuckets) {
        const uint64_t fpHash = hash64(pureFp);
        const uint64_t offset = fpHash % numBuckets;
        return offset == 0 ? 1 : offset;
    }

    /**
     * @brief Computes the primary and alternate bucket indices for a given key.
     *
     * Returns fingerprint with choice=0 (indicating primary bucket).
     *
     * @param key Key to hash
     * @param numBuckets Number of buckets in the filter
     * @return A tuple of (bucket1, bucket2, fingerprint_with_choice_0)
     */
    static __host__ __device__ cuda::std::tuple<size_t, size_t, TagType>
    getCandidateBuckets(const KeyType& key, size_t numBuckets) {
        const uint64_t h = hash64(key);

        // Upper 32 bits for the fingerprint (use bitsPerTag-1 bits)
        const uint32_t h_fp = h >> 32;
        const uint32_t masked = h_fp & pureFpMask;
        // Ensure non-zero (0 is reserved for empty)
        const TagType pureFp = TagType(masked == 0 ? 1 : masked);

        // Combine with choice=0 (primary bucket)
        const TagType fp = setChoiceBit(pureFp, 0);

        // Lower 32 bits for the bucket index
        const uint32_t h_bucket = h & 0xFFFFFFFF;
        const size_t i1 = h_bucket % numBuckets;

        // Compute i2 using the fingerprint (this will flip the choice bit)
        const size_t offset = computeOffset(pureFp, numBuckets);
        const size_t i2 = (i1 + offset) % numBuckets;

        return {i1, i2, fp};
    }

    /**
     * @brief Computes the alternate bucket index using asymmetric offset.
     *
     * The choice bit determines direction:
     * - choice=0: we're at primary bucket, add offset to get alternate
     * - choice=1: we're at alternate bucket, subtract offset to get primary
     *
     * IMPORTANT: This also flips the choice bit in the fingerprint!
     * The caller must update the stored fingerprint with the returned value.
     *
     * @param bucket Current bucket index.
     * @param fp Fingerprint (with choice bit) - will be modified to flip choice.
     * @param numBuckets Total number of buckets.
     * @return size_t Alternate bucket index.
     */
    static __host__ __device__ size_t
    getAlternateBucket(size_t bucket, TagType& fp, size_t numBuckets) {
        const TagType pureFp = getPureFingerprint(fp);
        const TagType choice = getChoiceBit(fp);
        const size_t offset = computeOffset(pureFp, numBuckets);

        size_t alt;
        if (choice == 0) {
            // At primary bucket, add offset to get alternate
            alt = (bucket + offset) % numBuckets;
        } else {
            // At alternate bucket, subtract offset to get primary (avoid underflow)
            alt = (bucket + numBuckets - offset) % numBuckets;
        }

        // Flip the choice bit
        fp = setChoiceBit(pureFp, 1 - choice);

        return alt;
    }

    /**
     * @brief Const version that doesn't modify the fingerprint.
     * Returns what the alternate bucket would be AND what the new fp would be.
     */
    static __host__ __device__ cuda::std::tuple<size_t, TagType>
    getAlternateBucketWithNewFp(size_t bucket, TagType fp, size_t numBuckets) {
        const TagType pureFp = getPureFingerprint(fp);
        const TagType choice = getChoiceBit(fp);
        const size_t offset = computeOffset(pureFp, numBuckets);

        size_t alt;
        if (choice == 0) {
            alt = (bucket + offset) % numBuckets;
        } else {
            alt = (bucket + numBuckets - offset) % numBuckets;
        }

        TagType newFp = setChoiceBit(pureFp, 1 - choice);
        return {alt, newFp};
    }

    /**
     * @brief Calculates number of buckets without requiring power of two.
     */
    static size_t calculateNumBuckets(size_t capacity) {
        auto requiredBuckets = std::ceil(static_cast<double>(capacity) / bucketSize);
        return static_cast<size_t>(requiredBuckets);
    }
};