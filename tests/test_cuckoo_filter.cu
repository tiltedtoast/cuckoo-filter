#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <BucketsTableGpu.cuh>
#include <helpers.cuh>
#include <random>
#include <unordered_set>
#include <vector>
#include "subprojects/googletest-1.17.0/googletest/include/gtest/gtest.h"

class CuckooFilterTest : public ::testing::Test {
   protected:
    using Config = CuckooConfig<uint32_t, 16, 1000, 256, 128>;

    void SetUp() override {
        rng.seed(42);
    }

    template <typename T>
    std::vector<T> generateRandomKeys(size_t n) {
        std::vector<T> keys(n);
        std::uniform_int_distribution<T> dist(1, std::numeric_limits<T>::max());
        std::generate(keys.begin(), keys.end(), [&]() { return dist(rng); });

        return keys;
    }

    std::mt19937 rng;
};

TEST_F(CuckooFilterTest, BasicInsertAndQuery) {
    const size_t numBuckets = 1024;
    BucketsTableGpu<Config> filter(numBuckets);

    std::vector<uint32_t> keys = {1, 2, 3, 4, 5, 100, 200, 300};

    size_t inserted = filter.insertMany(keys.data(), keys.size());
    EXPECT_EQ(inserted, keys.size())
        << "All keys should be inserted successfully";

    std::vector<uint8_t> output(keys.size());
    bool* outputPtr = reinterpret_cast<bool*>(output.data());
    filter.containsMany(keys.data(), keys.size(), outputPtr);

    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_TRUE(output[i]) << "Key " << keys[i] << " should be found";
    }
}

TEST_F(CuckooFilterTest, EmptyFilter) {
    const size_t numBuckets = 1024;
    BucketsTableGpu<Config> filter(numBuckets);

    std::vector<uint32_t> queryKeys = {1, 2, 3, 4, 5};
    std::vector<uint8_t> output(queryKeys.size());
    bool* outputPtr = reinterpret_cast<bool*>(output.data());

    filter.containsMany(queryKeys.data(), queryKeys.size(), outputPtr);

    size_t found = std::count(output.begin(), output.end(), true);

    EXPECT_EQ(found, 0) << "Empty filter should not find any keys";
}

TEST_F(CuckooFilterTest, ClearOperation) {
    const size_t numBuckets = 1024;
    BucketsTableGpu<Config> filter(numBuckets);

    std::vector<uint32_t> keys = {1, 2, 3, 4, 5};

    filter.insertMany(keys.data(), keys.size());

    std::vector<uint8_t> output(keys.size());
    bool* outputPtr = reinterpret_cast<bool*>(output.data());
    filter.containsMany(keys.data(), keys.size(), outputPtr);

    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_TRUE(outputPtr[i]) << "Key should be found before clear";
    }

    filter.clear();
    filter.containsMany(keys.data(), keys.size(), outputPtr);

    size_t found = std::count(output.begin(), output.end(), true);

    EXPECT_EQ(found, 0) << "After clear, no keys should be found";
    EXPECT_FLOAT_EQ(filter.loadFactor(), 0.0f)
        << "Load factor should be 0 after clear";
}

TEST_F(CuckooFilterTest, LoadFactor) {
    const size_t numBuckets = 1024;
    BucketsTableGpu<Config> filter(numBuckets);

    EXPECT_FLOAT_EQ(filter.loadFactor(), 0.0f);

    const size_t numKeys = 5000;

    auto keys = generateRandomKeys<uint32_t>(numKeys);

    size_t inserted = filter.insertMany(keys.data(), keys.size());

    float loadFactor = filter.loadFactor();
    EXPECT_GT(loadFactor, 0.0f)
        << "Load factor should be positive after insertions";
    EXPECT_LE(loadFactor, 1.0f) << "Load factor should not exceed 1.0";

    const auto totalCapacity =
        static_cast<float>(numBuckets * BucketsTableGpu<Config>::bucketSize);

    EXPECT_FLOAT_EQ(loadFactor, inserted / totalCapacity)
        << "Load factor should be approximately inserted / totalCapacity";

    EXPECT_GT(inserted, 0) << "Some keys should be inserted";
}

TEST_F(CuckooFilterTest, NearCapacityInsertion) {
    const size_t numKeys = 1 << 20;
    const size_t numBuckets = numKeys / BucketsTableGpu<Config>::bucketSize;
    BucketsTableGpu<Config> filter(numBuckets);

    auto keys = generateRandomKeys<uint32_t>(numKeys);

    size_t inserted = filter.insertMany(keys.data(), keys.size());
    EXPECT_GT(inserted, numKeys * 0.99) << "Should insert at least 99% of keys";

    std::vector<uint8_t> output(numKeys);
    bool* outputPtr = reinterpret_cast<bool*>(output.data());
    filter.containsMany(keys.data(), keys.size(), outputPtr);

    size_t found = std::count(output.begin(), output.end(), true);

    EXPECT_GT(found, inserted * 0.99)
        << "Should find at least 99% of inserted keys";
}

TEST_F(CuckooFilterTest, DuplicateInsertions) {
    const size_t numBuckets = 1024;
    BucketsTableGpu<Config> filter(numBuckets);

    std::vector<uint32_t> keys = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5};

    size_t inserted = filter.insertMany(keys.data(), keys.size());

    EXPECT_EQ(inserted, keys.size());

    std::vector<uint32_t> uniqueKeys = {1, 2, 3, 4, 5};
    std::vector<uint8_t> output(uniqueKeys.size());
    bool* outputPtr = reinterpret_cast<bool*>(output.data());
    filter.containsMany(uniqueKeys.data(), uniqueKeys.size(), outputPtr);

    for (size_t i = 0; i < uniqueKeys.size(); ++i) {
        EXPECT_TRUE(outputPtr[i])
            << "Unique key " << uniqueKeys[i] << " should be found";
    }
}
