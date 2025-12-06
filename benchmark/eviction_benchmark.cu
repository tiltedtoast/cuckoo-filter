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

static constexpr double PREFILL_LOAD_FACTOR = 0.75;

static constexpr double LOAD_FACTORS[] = {
    0.76,
    0.78,
    0.80,
    0.82,
    0.84,
    0.86,
    0.88,
    0.90,
    0.92,
    0.94,
    0.95,
    0.96,
    0.97,
    0.98,
    0.99,
    0.995,
    0.999
};

static constexpr size_t NUM_LOAD_FACTORS = sizeof(LOAD_FACTORS) / sizeof(LOAD_FACTORS[0]);

template <typename ConfigType>
class EvictionBenchmarkFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    void SetUp(const benchmark::State& state) override {
        capacity = state.range(0);
        loadFactor = LOAD_FACTORS[state.range(1)];

        auto totalKeys = static_cast<size_t>(capacity * loadFactor);
        nPrefill = static_cast<size_t>(capacity * PREFILL_LOAD_FACTOR);
        nMeasured = totalKeys > nPrefill ? totalKeys - nPrefill : 0;

        d_keysPrefill.resize(nPrefill);
        d_keysMeasured.resize(nMeasured);
        generateKeysGPU(d_keysPrefill);
        generateKeysGPURange(
            d_keysMeasured, nMeasured, static_cast<uint64_t>(UINT32_MAX) + 1, UINT64_MAX
        );

        filter = std::make_unique<CuckooFilter<ConfigType>>(capacity);
        filterMemory = filter->sizeInBytes();
    }

    void TearDown(const benchmark::State&) override {
        filter.reset();
        d_keysPrefill.clear();
        d_keysPrefill.shrink_to_fit();
        d_keysMeasured.clear();
        d_keysMeasured.shrink_to_fit();
    }

    void setCounters(benchmark::State& state, size_t evictions, size_t inserted) {
        state.counters["load_factor"] = bm::Counter(loadFactor * 100);
        state.counters["prefill_load_factor"] = bm::Counter(PREFILL_LOAD_FACTOR * 100);
        state.counters["evictions"] = bm::Counter(static_cast<double>(evictions));
        state.counters["inserted"] = bm::Counter(static_cast<double>(inserted));
        state.counters["evictions_per_insert"] = bm::Counter(
            inserted > 0 ? static_cast<double>(evictions) / static_cast<double>(inserted) : 0
        );
        state.SetItemsProcessed(
            static_cast<int64_t>(state.iterations()) * static_cast<int64_t>(nMeasured)
        );
        state.counters["memory_bytes"] = bm::Counter(
            static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
        );
    }

    size_t capacity;
    double loadFactor;
    size_t nPrefill;
    size_t nMeasured;
    size_t filterMemory;
    thrust::device_vector<uint64_t> d_keysPrefill;
    thrust::device_vector<uint64_t> d_keysMeasured;
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
        filter->insertMany(d_keysPrefill);
        filter->resetEvictionCount();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = filter->insertMany(d_keysMeasured);
        double elapsed = timer.elapsed();

        size_t evictions = filter->evictionCount();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);

        totalEvictions = evictions;
        totalInserted = nMeasured;
    }

    setCounters(state, totalEvictions, totalInserted);
}

BENCHMARK_DEFINE_F(DFSFixture, Evictions)(bm::State& state) {
    size_t totalEvictions = 0;
    size_t totalInserted = 0;

    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();
        filter->insertMany(d_keysPrefill);
        filter->resetEvictionCount();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = filter->insertMany(d_keysMeasured);
        double elapsed = timer.elapsed();

        size_t evictions = filter->evictionCount();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);

        totalEvictions = evictions;
        totalInserted = nMeasured;
    }

    setCounters(state, totalEvictions, totalInserted);
}

#define EVICTION_BENCHMARK_CONFIG                                                       \
    ->ArgsProduct({{1 << 24}, benchmark::CreateDenseRange(0, NUM_LOAD_FACTORS - 1, 1)}) \
        ->Unit(benchmark::kMillisecond)                                                 \
        ->UseManualTime()                                                               \
        ->Iterations(20)                                                                \
        ->Repetitions(5)                                                                \
        ->ReportAggregatesOnly(true)

BENCHMARK_REGISTER_F(BFSFixture, Evictions)
EVICTION_BENCHMARK_CONFIG;

BENCHMARK_REGISTER_F(DFSFixture, Evictions)
EVICTION_BENCHMARK_CONFIG;

STANDARD_BENCHMARK_MAIN();
