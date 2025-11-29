#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <helpers.cuh>
#include "benchmark_common.cuh"

namespace bm = benchmark;

template <size_t bucketSize>
class BucketSizeFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    using Config = CuckooConfig<uint64_t, 16, 500, 128, bucketSize>;
    static constexpr double TARGET_LOAD_FACTOR = 0.95;

    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(state.range(0), TARGET_LOAD_FACTOR);
        capacity = cap;
        n = num;

        d_keys.resize(n);
        d_output.resize(n);
        generateKeysGPU(d_keys);

        filter = std::make_unique<CuckooFilter<Config>>(capacity);
        filterMemory = filter->sizeInBytes();
    }

    void TearDown(const benchmark::State&) override {
        filter.reset();
        d_keys.clear();
        d_keys.shrink_to_fit();
        d_output.clear();
        d_output.shrink_to_fit();
    }

    void setCounters(benchmark::State& state) {
        setCommonCounters(state, filterMemory, n);
        state.counters["bucket_size"] = bm::Counter(bucketSize);
    }

    size_t capacity;
    size_t n;
    size_t filterMemory;
    thrust::device_vector<uint64_t> d_keys;
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<CuckooFilter<Config>> filter;
    Timer timer;
};

#define REGISTER_BUCKET_BENCHMARK(BSize, BenchName)   \
    BENCHMARK_REGISTER_F(BSFixture##BSize, BenchName) \
    BENCHMARK_CONFIG

#define REGISTER_ALL_FOR_BUCKET_SIZE(BSize)   \
    REGISTER_BUCKET_BENCHMARK(BSize, Insert); \
    REGISTER_BUCKET_BENCHMARK(BSize, Query);  \
    REGISTER_BUCKET_BENCHMARK(BSize, Delete);

#define DEFINE_BUCKET_SIZE_BENCHMARKS(BSize)           \
    using BSFixture##BSize = BucketSizeFixture<BSize>; \
    DEFINE_CORE_BENCHMARKS(BSFixture##BSize)

#define DEFINE_AND_REGISTER_BUCKET_SIZE_BENCHMARKS(BSize) \
    DEFINE_BUCKET_SIZE_BENCHMARKS(BSize)                  \
    REGISTER_ALL_FOR_BUCKET_SIZE(BSize)

DEFINE_AND_REGISTER_BUCKET_SIZE_BENCHMARKS(4)
DEFINE_AND_REGISTER_BUCKET_SIZE_BENCHMARKS(8)
DEFINE_AND_REGISTER_BUCKET_SIZE_BENCHMARKS(16)
DEFINE_AND_REGISTER_BUCKET_SIZE_BENCHMARKS(32)
DEFINE_AND_REGISTER_BUCKET_SIZE_BENCHMARKS(64)
DEFINE_AND_REGISTER_BUCKET_SIZE_BENCHMARKS(128)

BENCHMARK_MAIN();