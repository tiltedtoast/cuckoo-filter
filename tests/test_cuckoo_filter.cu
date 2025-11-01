#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <CuckooFilter.cuh>
#include <helpers.cuh>
#include <random>
#include <unordered_set>
#include <vector>

class CuckooFilterTest : public ::testing::Test {
   protected:
    using Config = CuckooConfig<uint32_t, 16, 500, 256, 16>;

    const double TARGET_LOAD_FACTOR = 0.95;

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
    CuckooFilter<Config> filter(capacity);

    std::vector<uint32_t> keys = {1, 2, 3, 4, 5, 100, 200, 300};
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> d_output(keys.size());

    size_t inserted = filter.insertMany(d_keys);
    EXPECT_EQ(inserted, keys.size()) << "All keys should be inserted successfully";

    filter.containsMany(d_keys, d_output);

    std::vector<uint8_t> output(keys.size());
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_TRUE(output[i]) << "Key " << keys[i] << " should be found";
    }
}

TEST_F(CuckooFilterTest, EmptyFilter) {
    const size_t capacity = 10000;
    CuckooFilter<Config> filter(capacity);

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
    CuckooFilter<Config> filter(capacity);

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
    EXPECT_FLOAT_EQ(filter.loadFactor(), 0.0f) << "Load factor should be 0 after clear";
}

TEST_F(CuckooFilterTest, LoadFactor) {
    const size_t capacity = 10000;
    CuckooFilter<Config> filter(capacity);

    EXPECT_FLOAT_EQ(filter.loadFactor(), 0.0f);

    const size_t numKeys = 5000;

    auto keys = generateRandomKeys<uint32_t>(numKeys);
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());

    size_t inserted = filter.insertMany(d_keys);

    float loadFactor = filter.loadFactor();
    EXPECT_GT(loadFactor, 0.0f) << "Load factor should be positive after insertions";
    EXPECT_LE(loadFactor, 1.0f) << "Load factor should not exceed 1.0";

    const auto totalCapacity = static_cast<float>(filter.capacity());

    EXPECT_FLOAT_EQ(loadFactor, inserted / totalCapacity)
        << "Load factor should be approximately inserted / totalCapacity";

    EXPECT_GT(inserted, 0) << "Some keys should be inserted";
}

TEST_F(CuckooFilterTest, NearCapacityInsertion) {
    const size_t numKeys = (1 << 20) * TARGET_LOAD_FACTOR;
    CuckooFilter<Config> filter(numKeys);

    auto keys = generateRandomKeys<uint32_t>(numKeys);
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> d_output(numKeys);

    size_t inserted = filter.insertMany(d_keys);
    EXPECT_GT(inserted, numKeys * 0.99) << "Should insert at least 99% of keys";

    filter.containsMany(d_keys, d_output);

    std::vector<uint8_t> output(numKeys);
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    size_t found = std::count(output.begin(), output.end(), true);

    EXPECT_GT(found, inserted * 0.99) << "Should find at least 99% of inserted keys";
}

TEST_F(CuckooFilterTest, DuplicateInsertions) {
    const size_t capacity = 10000;
    CuckooFilter<Config> filter(capacity);

    std::vector<uint32_t> keys = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5};
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());

    size_t inserted = filter.insertMany(d_keys);

    EXPECT_EQ(inserted, keys.size());

    std::vector<uint32_t> uniqueKeys = {1, 2, 3, 4, 5};
    thrust::device_vector<uint32_t> d_unique_keys(uniqueKeys.begin(), uniqueKeys.end());
    thrust::device_vector<uint8_t> d_output(uniqueKeys.size());

    filter.containsMany(d_unique_keys, d_output);

    std::vector<uint8_t> output(uniqueKeys.size());
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    for (size_t i = 0; i < uniqueKeys.size(); ++i) {
        EXPECT_TRUE(output[i]) << "Unique key " << uniqueKeys[i] << " should be found";
    }
}

TEST_F(CuckooFilterTest, BasicDeletion) {
    const size_t capacity = 10000;
    CuckooFilter<Config> filter(capacity);

    std::vector<uint32_t> keys = {1, 2, 3, 4, 5, 100, 200, 300};
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> d_output(keys.size());

    size_t inserted = filter.insertMany(d_keys);
    EXPECT_EQ(inserted, keys.size()) << "All keys should be inserted";

    filter.containsMany(d_keys, d_output);
    std::vector<uint8_t> output(keys.size());
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_TRUE(output[i]) << "Key " << keys[i] << " should be found before deletion";
    }

    thrust::device_vector<uint8_t> d_deleteOutput(keys.size());
    size_t remainingAfterDelete = filter.deleteMany(d_keys, d_deleteOutput);

    std::vector<uint8_t> deleteOutput(keys.size());
    thrust::copy(d_deleteOutput.begin(), d_deleteOutput.end(), deleteOutput.begin());

    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_TRUE(deleteOutput[i]) << "Key " << keys[i] << " should be successfully deleted";
    }

    EXPECT_EQ(remainingAfterDelete, 0) << "No keys should remain after deletion";

    filter.containsMany(d_keys, d_output);
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    size_t found = std::count(output.begin(), output.end(), true);
    EXPECT_EQ(found, 0) << "No keys should be found after deletion";
}

TEST_F(CuckooFilterTest, DeleteNonExistentKeys) {
    const size_t capacity = 10000;
    CuckooFilter<Config> filter(capacity);

    std::vector<uint32_t> keys = {1, 2, 3, 4, 5};
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> d_deleteOutput(keys.size());

    filter.deleteMany(d_keys, d_deleteOutput);

    std::vector<uint8_t> deleteOutput(keys.size());
    thrust::copy(d_deleteOutput.begin(), d_deleteOutput.end(), deleteOutput.begin());

    size_t deleted = std::count(deleteOutput.begin(), deleteOutput.end(), true);
    EXPECT_EQ(deleted, 0) << "Should not delete any non-existent keys";
}

TEST_F(CuckooFilterTest, PartialDeletion) {
    const size_t capacity = 10000;
    CuckooFilter<Config> filter(capacity);

    std::vector<uint32_t> insertKeys = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    thrust::device_vector<uint32_t> d_insertKeys(insertKeys.begin(), insertKeys.end());

    size_t inserted = filter.insertMany(d_insertKeys);
    EXPECT_EQ(inserted, insertKeys.size()) << "All keys should be inserted";

    std::vector<uint32_t> deleteKeys = {2, 4, 6, 8, 10};
    thrust::device_vector<uint32_t> d_delete_keys(deleteKeys.begin(), deleteKeys.end());
    thrust::device_vector<uint8_t> d_deleteOutput(deleteKeys.size());

    size_t remainingAfterDelete = filter.deleteMany(d_delete_keys, d_deleteOutput);

    std::vector<uint8_t> deleteOutput(deleteKeys.size());
    thrust::copy(d_deleteOutput.begin(), d_deleteOutput.end(), deleteOutput.begin());

    for (size_t i = 0; i < deleteKeys.size(); ++i) {
        EXPECT_TRUE(deleteOutput[i]) << "Key " << deleteKeys[i] << " should be deleted";
    }

    EXPECT_EQ(remainingAfterDelete, 5) << "5 keys should remain after deleting 5";

    thrust::device_vector<uint8_t> d_output(deleteKeys.size());
    filter.containsMany(d_delete_keys, d_output);

    std::vector<uint8_t> output(deleteKeys.size());
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    size_t found = std::count(output.begin(), output.end(), true);
    EXPECT_EQ(found, 0) << "Deleted keys should not be found";

    std::vector<uint32_t> remainingKeys = {1, 3, 5, 7, 9};
    thrust::device_vector<uint32_t> d_remainingKeys(remainingKeys.begin(), remainingKeys.end());
    thrust::device_vector<uint8_t> d_remainingOutput(remainingKeys.size());

    filter.containsMany(d_remainingKeys, d_remainingOutput);

    std::vector<uint8_t> remainingOutput(remainingKeys.size());
    thrust::copy(d_remainingOutput.begin(), d_remainingOutput.end(), remainingOutput.begin());

    for (size_t i = 0; i < remainingKeys.size(); ++i) {
        EXPECT_TRUE(remainingOutput[i]) << "Key " << remainingKeys[i] << " should still be found";
    }
}

TEST_F(CuckooFilterTest, DeleteAndReinsert) {
    const size_t capacity = 10000;
    CuckooFilter<Config> filter(capacity);

    std::vector<uint32_t> keys = {1, 2, 3, 4, 5};
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());
    thrust::device_vector<uint8_t> d_output(keys.size());

    filter.insertMany(d_keys);
    filter.deleteMany(d_keys);

    filter.containsMany(d_keys, d_output);
    std::vector<uint8_t> output(keys.size());
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    size_t found = std::count(output.begin(), output.end(), true);
    EXPECT_EQ(found, 0) << "Keys should not be found after deletion";

    size_t reinserted = filter.insertMany(d_keys);
    EXPECT_EQ(reinserted, keys.size()) << "Should be able to reinsert deleted keys";

    filter.containsMany(d_keys, d_output);
    thrust::copy(d_output.begin(), d_output.end(), output.begin());

    for (size_t i = 0; i < keys.size(); ++i) {
        EXPECT_TRUE(output[i]) << "Key " << keys[i] << " should be found after reinsertion";
    }
}

TEST_F(CuckooFilterTest, LoadFactorAfterDeletion) {
    const size_t capacity = 10000;
    CuckooFilter<Config> filter(capacity);

    const size_t numKeys = 1000;
    auto keys = generateRandomKeys<uint32_t>(numKeys);
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());

    size_t inserted = filter.insertMany(d_keys);
    float loadFactorBefore = filter.loadFactor();

    EXPECT_GT(loadFactorBefore, 0.0f) << "Load factor should be positive after insertion";

    size_t remaining = filter.deleteMany(d_keys);
    float loadFactorAfter = filter.loadFactor();

    EXPECT_LT(loadFactorAfter, loadFactorBefore) << "Load factor should decrease after deletion";
    EXPECT_FLOAT_EQ(loadFactorAfter, static_cast<float>(remaining) / filter.capacity());
}

TEST_F(CuckooFilterTest, DeleteDuplicates) {
    const size_t capacity = 10000;
    CuckooFilter<Config> filter(capacity);

    std::vector<uint32_t> keys = {1, 1, 1, 2, 2, 3};
    thrust::device_vector<uint32_t> d_keys(keys.begin(), keys.end());

    filter.insertMany(d_keys);

    std::vector<uint32_t> deleteKeys = {1, 1, 1};
    thrust::device_vector<uint32_t> d_deleteKeys(deleteKeys.begin(), deleteKeys.end());
    thrust::device_vector<uint8_t> d_deleteOutput(deleteKeys.size());

    filter.deleteMany(d_deleteKeys, d_deleteOutput);

    std::vector<uint8_t> deleteOutput(deleteKeys.size());
    thrust::copy(d_deleteOutput.begin(), d_deleteOutput.end(), deleteOutput.begin());

    size_t successfulDeletions = std::count(deleteOutput.begin(), deleteOutput.end(), true);
    EXPECT_GE(successfulDeletions, 1) << "At least one deletion should succeed";
}
