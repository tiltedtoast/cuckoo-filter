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
        d_output.clear();
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

#define DEFINE_BUCKET_SIZE_BENCHMARKS(BSize)                                     \
    using BSFixture##BSize = BucketSizeFixture<BSize>;                           \
                                                                                 \
    BENCHMARK_DEFINE_F(BSFixture##BSize, Insert)(bm::State & state) {            \
        for (auto _ : state) {                                                   \
            filter->clear();                                                     \
            cudaDeviceSynchronize();                                             \
                                                                                 \
            timer.start();                                                       \
            size_t inserted = adaptiveInsert(*filter, d_keys);                   \
            double elapsed = timer.stop();                                       \
                                                                                 \
            state.SetIterationTime(elapsed);                                     \
            bm::DoNotOptimize(inserted);                                         \
        }                                                                        \
        setCounters(state);                                                      \
    }                                                                            \
                                                                                 \
    BENCHMARK_DEFINE_F(BSFixture##BSize, Query)(bm::State & state) {             \
        adaptiveInsert(*filter, d_keys);                                         \
                                                                                 \
        for (auto _ : state) {                                                   \
            timer.start();                                                       \
            filter->containsMany(d_keys, d_output);                              \
            double elapsed = timer.stop();                                       \
                                                                                 \
            state.SetIterationTime(elapsed);                                     \
            bm::DoNotOptimize(d_output.data().get());                            \
        }                                                                        \
        setCounters(state);                                                      \
    }                                                                            \
                                                                                 \
    BENCHMARK_DEFINE_F(BSFixture##BSize, Delete)(bm::State & state) {            \
        for (auto _ : state) {                                                   \
            filter->clear();                                                     \
            adaptiveInsert(*filter, d_keys);                                     \
            cudaDeviceSynchronize();                                             \
                                                                                 \
            timer.start();                                                       \
            size_t remaining = filter->deleteMany(d_keys, d_output);             \
            double elapsed = timer.stop();                                       \
                                                                                 \
            state.SetIterationTime(elapsed);                                     \
            bm::DoNotOptimize(remaining);                                        \
            bm::DoNotOptimize(d_output.data().get());                            \
        }                                                                        \
        setCounters(state);                                                      \
    }                                                                            \
                                                                                 \
    BENCHMARK_DEFINE_F(BSFixture##BSize, InsertAndQuery)(bm::State & state) {    \
        for (auto _ : state) {                                                   \
            filter->clear();                                                     \
            cudaDeviceSynchronize();                                             \
                                                                                 \
            timer.start();                                                       \
            size_t inserted = adaptiveInsert(*filter, d_keys);                   \
            filter->containsMany(d_keys, d_output);                              \
            double elapsed = timer.stop();                                       \
                                                                                 \
            state.SetIterationTime(elapsed);                                     \
            bm::DoNotOptimize(inserted);                                         \
            bm::DoNotOptimize(d_output.data().get());                            \
        }                                                                        \
        setCounters(state);                                                      \
    }                                                                            \
                                                                                 \
    BENCHMARK_DEFINE_F(BSFixture##BSize, InsertQueryDelete)(bm::State & state) { \
        for (auto _ : state) {                                                   \
            filter->clear();                                                     \
            cudaDeviceSynchronize();                                             \
                                                                                 \
            timer.start();                                                       \
            size_t inserted = adaptiveInsert(*filter, d_keys);                   \
            filter->containsMany(d_keys, d_output);                              \
            size_t remaining = filter->deleteMany(d_keys, d_output);             \
            double elapsed = timer.stop();                                       \
                                                                                 \
            state.SetIterationTime(elapsed);                                     \
            bm::DoNotOptimize(inserted);                                         \
            bm::DoNotOptimize(remaining);                                        \
            bm::DoNotOptimize(d_output.data().get());                            \
        }                                                                        \
        setCounters(state);                                                      \
    }

DEFINE_BUCKET_SIZE_BENCHMARKS(4)
DEFINE_BUCKET_SIZE_BENCHMARKS(8)
DEFINE_BUCKET_SIZE_BENCHMARKS(16)
DEFINE_BUCKET_SIZE_BENCHMARKS(32)
DEFINE_BUCKET_SIZE_BENCHMARKS(64)
DEFINE_BUCKET_SIZE_BENCHMARKS(128)

#define REGISTER_BUCKET_BENCHMARK(BSize, BenchName)   \
    BENCHMARK_REGISTER_F(BSFixture##BSize, BenchName) \
    BENCHMARK_CONFIG

#define REGISTER_ALL_FOR_BUCKET_SIZE(BSize)           \
    REGISTER_BUCKET_BENCHMARK(BSize, Insert);         \
    REGISTER_BUCKET_BENCHMARK(BSize, Query);          \
    REGISTER_BUCKET_BENCHMARK(BSize, Delete);         \
    REGISTER_BUCKET_BENCHMARK(BSize, InsertAndQuery); \
    REGISTER_BUCKET_BENCHMARK(BSize, InsertQueryDelete);

REGISTER_ALL_FOR_BUCKET_SIZE(4);
REGISTER_ALL_FOR_BUCKET_SIZE(8);
REGISTER_ALL_FOR_BUCKET_SIZE(16);
REGISTER_ALL_FOR_BUCKET_SIZE(32);
REGISTER_ALL_FOR_BUCKET_SIZE(64);
REGISTER_ALL_FOR_BUCKET_SIZE(128);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }

    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();

    fflush(stdout);
    std::cout << std::flush;

    std::_Exit(0);
}