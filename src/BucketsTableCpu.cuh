#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cuco/hash_functions.cuh>
#include <iostream>
#include <random>
#include "common.cuh"

template <
    typename T,
    size_t bitsPerTag,
    size_t bucketSize = 4,
    size_t maxProbes = 500>
class BucketsTableCpu {
    static_assert(bitsPerTag <= 32, "The tag cannot be larger than 32 bits");
    static_assert(bitsPerTag >= 1, "The tag must be at least 1 bit");
    static_assert(
        bitsPerTag <= 8 * sizeof(T),
        "The tag cannot be larger than the size of the type"
    );
    static_assert(bucketSize > 0, "Bucket size must be greater than 0");

    using TagType = typename std::conditional<
        bitsPerTag <= 8,
        uint8_t,
        typename std::conditional<bitsPerTag <= 16, uint16_t, uint32_t>::type>::
        type;

    static constexpr TagType EMPTY = 0;
    static constexpr size_t tagMask = (1ULL << bitsPerTag) - 1;

    struct __align__(alignof(TagType)) Bucket {
        TagType tags[bucketSize];

        [[nodiscard]] int findEmptySlot() const {
            for (size_t i = 0; i < bucketSize; ++i) {
                if (tags[i] == EMPTY) {
                    return static_cast<int>(i);
                }
            }
            return -1;
        }

        bool contains(TagType tag) const {
            for (size_t i = 0; i < bucketSize; ++i) {
                if (tags[i] == tag) {
                    return true;
                }
            }
            return false;
        }

        void insertAt(size_t slot, TagType tag) {
            tags[slot] = tag;
        }

        bool remove(TagType tag) {
            for (size_t i = 0; i < bucketSize; ++i) {
                if (tags[i] == tag) {
                    tags[i] = EMPTY;
                    return true;
                }
            }
            return false;
        }

        TagType getTagAt(size_t slot) const {
            return tags[slot];
        }
    };

    std::vector<Bucket> buckets;
    std::atomic<size_t> numOccupied = 0;
    std::mt19937 rng;
    size_t numBuckets;

    template <typename H>
    static __host__ __device__ uint32_t hash(const H& key) {
        auto bytes = reinterpret_cast<const cuda::std::byte*>(&key);
        cuco::xxhash_32<H> hasher;
        return hasher.compute_hash(bytes, sizeof(H));
    }

    static TagType fingerprint(const T& key) {
        uint32_t hash_val = hash(key);
        auto fp = static_cast<TagType>(hash_val & tagMask);
        // 0 is reserved for empty slots
        return fp == 0 ? 1 : fp;
    }

    std::tuple<size_t, size_t, size_t> getCandidateBuckets(const T& key) {
        TagType fp = fingerprint(key);
        size_t h1 = hash(key) & (numBuckets - 1);
        size_t h2 = h1 ^ (hash(fp) & (numBuckets - 1));
        return {h1, h2, fp};
    }

    size_t getAlternateBucket(size_t bucket, TagType fp) const {
        return bucket ^ (hash(fp) & (numBuckets - 1));
    }

    bool tryInsertAtBucket(size_t bucketIdx, TagType tag) {
        int slot = buckets[bucketIdx].findEmptySlot();
        if (slot >= 0) {
            buckets[bucketIdx].insertAt(slot, tag);
            numOccupied++;
            return true;
        }
        return false;
    }

    bool insertWithEviction(TagType fp, size_t startBucket) {
        TagType currentFp = fp;
        size_t currentBucket = startBucket;

        std::uniform_int_distribution<size_t> slotDist(0, bucketSize - 1);

        for (size_t evictions = 0; evictions < maxProbes; ++evictions) {
            if (tryInsertAtBucket(currentBucket, currentFp)) {
                return true;
            }

            size_t evictSlot = slotDist(rng);
            TagType evictedFp = buckets[currentBucket].getTagAt(evictSlot);
            buckets[currentBucket].insertAt(evictSlot, currentFp);

            currentFp = evictedFp;
            currentBucket = getAlternateBucket(currentBucket, evictedFp);
        }

        return false;
    }

   public:
    explicit BucketsTableCpu(size_t numBuckets)
        : buckets(numBuckets),
          rng(std::random_device{}()),
          numBuckets(numBuckets) {
        assert(
            powerOfTwo(numBuckets) && "Number of buckets must be a power of 2"
        );
        std::memset(buckets.data(), 0, sizeof(Bucket) * numBuckets);
    }

    bool insert(const T& key) {
        auto [h1, h2, fp] = getCandidateBuckets(key);

        if (tryInsertAtBucket(h1, fp) || tryInsertAtBucket(h2, fp)) {
            return true;
        }

        return insertWithEviction(fp, h1);
    }

    bool contains(const T& key) const {
        auto [h1, h2, fp] = getCandidateBuckets(key);

        return buckets[h1].contains(fp) || buckets[h2].contains(fp);
    }

    const bool* containsMany(const T* keys, const size_t n) {
        bool* output = static_cast<bool*>(std::malloc(n * sizeof(bool)));
        for (size_t i = 0; i < n; ++i) {
            output[i] = contains(keys[i]);
        }
        return output;
    }

    bool remove(const T& key) {
        auto [h1, h2, fp] = getCandidateBuckets(key);

        if (buckets[h1].contains(fp)) {
            if (buckets[h1].remove(fp)) {
                numOccupied--;
                return true;
            }
        }

        if (buckets[h2].contains(fp)) {
            if (buckets[h2].remove(fp)) {
                numOccupied--;
                return true;
            }
        }

        return false;
    }

    [[nodiscard]] float loadFactor() const {
        return static_cast<float>(numOccupied.load()) /
               (numBuckets * bucketSize);
    }

    [[nodiscard]] double expectedFalsePositiveRate() const {
        return (2.0 * bucketSize) / (1ULL << bitsPerTag);
    }

    void clear() {
        std::memset(buckets.data(), 0, sizeof(Bucket) * numBuckets);
        numOccupied = 0;
    }
};
