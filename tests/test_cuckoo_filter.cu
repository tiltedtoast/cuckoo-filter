#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
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
    const size_t capacity = 10000;
    BucketsTableGpu<Config> filter(capacity);

    std::vector<uint32_t> keys = {1, 2, 3, 4, 5, 100, 200, 300};
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> d_output(keys.size());

    size_t inserted = filter.insertMany(d_keys);
    EXPECT_EQ(inserted, keys.size())
        << "All keys should be inserted successfully";

    filter.containsMany(d_keys, d_output);

    std::vector<uint8_t> output(keys.size());
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_TRUE(output[i]) << "Key " << keys[i] << " should be found";
    }
}

TEST_F(CuckooFilterTest, EmptyFilter) {
    const size_t capacity = 10000;
    BucketsTableGpu<Config> filter(capacity);

    std::vector<uint32_t> queryKeys = {1, 2, 3, 4, 5};
    thrust::device_vector<uint32_t> d_keys(queryKeys.begin(), queryKeys.end());
    thrust::device_vector<uint8_t> d_output(queryKeys.size());

    filter.containsMany(d_keys, d_output);

    std::vector<uint8_t> output(queryKeys.size());
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    size_t found = std::count(output.begin(), output.end(), true);

    EXPECT_EQ(found, 0) << "Empty filter should not find any keys";
}

TEST_F(CuckooFilterTest, ClearOperation) {
    const size_t capacity = 10000;
    BucketsTableGpu<Config> filter(capacity);

    std::vector<uint32_t> keys = {1, 2, 3, 4, 5};
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> d_output(keys.size());

    filter.insertMany(d_keys);

    filter.containsMany(d_keys, d_output);

    std::vector<uint8_t> output(keys.size());
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_TRUE(output[i]) << "Key should be found before clear";
    }

    filter.clear();
    filter.containsMany(d_keys, d_output);

    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    size_t found = std::count(output.begin(), output.end(), true);

    EXPECT_EQ(found, 0) << "After clear, no keys should be found";
    EXPECT_FLOAT_EQ(filter.loadFactor(), 0.0f)
        << "Load factor should be 0 after clear";
}

TEST_F(CuckooFilterTest, LoadFactor) {
    const size_t capacity = 10000;
    BucketsTableGpu<Config> filter(capacity);

    EXPECT_FLOAT_EQ(filter.loadFactor(), 0.0f);

    const size_t numKeys = 5000;

    auto keys = generateRandomKeys<uint32_t>(numKeys);
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());

    size_t inserted = filter.insertMany(d_keys);

    float loadFactor = filter.loadFactor();
    EXPECT_GT(loadFactor, 0.0f)
        << "Load factor should be positive after insertions";
    EXPECT_LE(loadFactor, 1.0f) << "Load factor should not exceed 1.0";

    const auto totalCapacity = static_cast<float>(filter.capacity());

    EXPECT_FLOAT_EQ(loadFactor, inserted / totalCapacity)
        << "Load factor should be approximately inserted / totalCapacity";

    EXPECT_GT(inserted, 0) << "Some keys should be inserted";
}

TEST_F(CuckooFilterTest, NearCapacityInsertion) {
    const size_t numKeys = 1 << 20;
    BucketsTableGpu<Config> filter(numKeys);

    auto keys = generateRandomKeys<uint32_t>(numKeys);
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> d_output(numKeys);

    size_t inserted = filter.insertMany(d_keys);
    EXPECT_GT(inserted, numKeys * 0.99) << "Should insert at least 99% of keys";

    filter.containsMany(d_keys, d_output);

    std::vector<uint8_t> output(numKeys);
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    size_t found = std::count(output.begin(), output.end(), true);

    EXPECT_GT(found, inserted * 0.99)
        << "Should find at least 99% of inserted keys";
}

TEST_F(CuckooFilterTest, DuplicateInsertions) {
    const size_t capacity = 10000;
    BucketsTableGpu<Config> filter(capacity);

    std::vector<uint32_t> keys = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());

    size_t inserted = filter.insertMany(d_keys);

    EXPECT_EQ(inserted, keys.size());

    std::vector<uint32_t> uniqueKeys = {1, 2, 3, 4, 5};
    thrust::device_vector<uint32_t> d_unique_keys(
        uniqueKeys.begin(), uniqueKeys.end()
    );
    thrust::device_vector<uint8_t> d_output(uniqueKeys.size());

    filter.containsMany(d_unique_keys, d_output);

    std::vector<uint8_t> output(uniqueKeys.size());
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    for (size_t i = 0; i < uniqueKeys.size(); ++i) {
        EXPECT_TRUE(output[i])
            << "Unique key " << uniqueKeys[i] << " should be found";
    }
}
