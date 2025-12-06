// Enable eviction counting for this benchmark
#define CUCKOO_FILTER_COUNT_EVICTIONS

#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <helpers.cuh>
#include "benchmark_common.cuh"

namespace bm = benchmark;

using BFSConfig = CuckooConfig<uint64_t, 16, 500, 256, 16, XorAltBucketPolicy, EvictionPolicy::BFS>;
using DFSConfig = CuckooConfig<uint64_t, 16, 500, 256, 16, XorAltBucketPolicy, EvictionPolicy::DFS>;

static constexpr double LOAD_FACTORS[] = {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
                                          0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                                          0.85, 0.90, 0.95, 0.98, 0.99, 0.995};

static constexpr size_t NUM_LOAD_FACTORS = sizeof(LOAD_FACTORS) / sizeof(LOAD_FACTORS[0]);

template <typename ConfigType>
class EvictionBenchmarkFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    void SetUp(const benchmark::State& state) override {
        capacity = state.range(0);
        loadFactor = LOAD_FACTORS[state.range(1)];
        n = static_cast<size_t>(capacity * loadFactor);

        d_keys.resize(n);
        generateKeysGPU(d_keys);

        filter = std::make_unique<CuckooFilter<ConfigType>>(capacity);
        filterMemory = filter->sizeInBytes();
    }

    void TearDown(const benchmark::State&) override {
        filter.reset();
        d_keys.clear();
        d_keys.shrink_to_fit();
    }

    void setCounters(benchmark::State& state, size_t evictions, size_t inserted) {
        state.counters["load_factor"] = bm::Counter(loadFactor * 100);
        state.counters["evictions"] = bm::Counter(static_cast<double>(evictions));
        state.counters["inserted"] = bm::Counter(static_cast<double>(inserted));
        state.counters["evictions_per_insert"] = bm::Counter(
            inserted > 0 ? static_cast<double>(evictions) / static_cast<double>(inserted) : 0
        );
        state.counters["memory_bytes"] = bm::Counter(
            static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
        );
    }

    size_t capacity;
    double loadFactor;
    size_t n;
    size_t filterMemory;
    thrust::device_vector<uint64_t> d_keys;
    std::unique_ptr<CuckooFilter<ConfigType>> filter;
    GPUTimer timer;
};

using BFSFixture = EvictionBenchmarkFixture<BFSConfig>;
using DFSFixture = EvictionBenchmarkFixture<DFSConfig>;

BENCHMARK_DEFINE_F(BFSFixture, Evictions)(bm::State& state) {
    size_t totalEvictions = 0;
    size_t totalInserted = 0;

    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = filter->insertMany(d_keys);
        double elapsed = timer.elapsed();

        size_t evictions = filter->evictionCount();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);

        totalEvictions = evictions;
        totalInserted = inserted;
    }

    setCounters(state, totalEvictions, totalInserted);
}

BENCHMARK_DEFINE_F(DFSFixture, Evictions)(bm::State& state) {
    size_t totalEvictions = 0;
    size_t totalInserted = 0;

    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = filter->insertMany(d_keys);
        double elapsed = timer.elapsed();

        size_t evictions = filter->evictionCount();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);

        totalEvictions = evictions;
        totalInserted = inserted;
    }

    setCounters(state, totalEvictions, totalInserted);
}

#define EVICTION_BENCHMARK_CONFIG                                                       \
    ->ArgsProduct({{1 << 24}, benchmark::CreateDenseRange(0, NUM_LOAD_FACTORS - 1, 1)}) \
        ->Unit(benchmark::kMillisecond)                                                 \
        ->UseManualTime()                                                               \
        ->Iterations(10)                                                                \
        ->Repetitions(5)                                                                \
        ->ReportAggregatesOnly(true)

BENCHMARK_REGISTER_F(BFSFixture, Evictions)
EVICTION_BENCHMARK_CONFIG;

BENCHMARK_REGISTER_F(DFSFixture, Evictions)
EVICTION_BENCHMARK_CONFIG;

STANDARD_BENCHMARK_MAIN();
