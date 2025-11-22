#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <CuckooFilterMultiGPU.cuh>
#include <helpers.cuh>
#include "benchmark_common.cuh"

namespace bm = benchmark;

using Config = CuckooConfig<uint64_t, 16, 500, 128, 16>;

template <typename ConfigType, double loadFactor = 0.95>
class MultiGPUFixture_ : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    using KeyType = typename ConfigType::KeyType;

    void SetUp(const benchmark::State& state) override {
        int deviceCount;
        CUDA_CALL(cudaGetDeviceCount(&deviceCount));
        numGPUs = static_cast<size_t>(deviceCount);

        auto [cap, num] = calculateCapacityAndSize(state.range(0), loadFactor);
        capacity = cap;
        n = num;

        h_keys.resize(n);
        thrust::device_vector<KeyType> d_temp(n);
        generateKeysGPU(d_temp);
        thrust::copy(d_temp.begin(), d_temp.end(), h_keys.begin());

        h_output.resize(n);

        filter = std::make_unique<CuckooFilterMultiGPU<ConfigType>>(numGPUs, capacity);
        filterMemory = filter->sizeInBytes();
    }

    void TearDown(const benchmark::State&) override {
        filter.reset();
        h_keys.clear();
        h_output.clear();
        h_keys.shrink_to_fit();
        h_output.shrink_to_fit();
    }

    void setCounters(benchmark::State& state) {
        setCommonCounters(state, filterMemory, n);
        state.counters["gpus"] = static_cast<double>(numGPUs);
    }

    size_t numGPUs;
    size_t capacity;
    size_t n;
    size_t filterMemory;
    thrust::host_vector<KeyType> h_keys;
    thrust::host_vector<bool> h_output;
    std::unique_ptr<CuckooFilterMultiGPU<ConfigType>> filter;
    Timer timer;
};

using SingleGPUFixture = CuckooFilterFixture<Config, 0.95>;
using MultiGPUFixture = MultiGPUFixture_<Config>;

BENCHMARK_DEFINE_F(SingleGPUFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = adaptiveInsert(*filter, d_keys);
        cudaDeviceSynchronize();
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(SingleGPUFixture, Query)(bm::State& state) {
    adaptiveInsert(*filter, d_keys);
    cudaDeviceSynchronize();

    for (auto _ : state) {
        timer.start();
        filter->containsMany(d_keys, d_output);
        cudaDeviceSynchronize();
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(SingleGPUFixture, Delete)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        adaptiveInsert(*filter, d_keys);
        cudaDeviceSynchronize();

        timer.start();
        size_t remaining = filter->deleteMany(d_keys, d_output);
        cudaDeviceSynchronize();
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(MultiGPUFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        filter->synchronizeAllGPUs();

        timer.start();
        size_t inserted = filter->insertMany(h_keys);
        filter->synchronizeAllGPUs();
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(MultiGPUFixture, Query)(bm::State& state) {
    filter->insertMany(h_keys);
    filter->synchronizeAllGPUs();

    for (auto _ : state) {
        timer.start();
        filter->containsMany(h_keys, h_output);
        filter->synchronizeAllGPUs();
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(h_output.data());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(MultiGPUFixture, Delete)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        filter->insertMany(h_keys);
        filter->synchronizeAllGPUs();

        timer.start();
        size_t remaining = filter->deleteMany(h_keys, h_output);
        filter->synchronizeAllGPUs();
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(h_output.data());
    }
    setCounters(state);
}

REGISTER_BENCHMARK(SingleGPUFixture, Insert);
REGISTER_BENCHMARK(SingleGPUFixture, Query);
REGISTER_BENCHMARK(SingleGPUFixture, Delete);

REGISTER_BENCHMARK(MultiGPUFixture, Insert);
REGISTER_BENCHMARK(MultiGPUFixture, Query);
REGISTER_BENCHMARK(MultiGPUFixture, Delete);

BENCHMARK_MAIN();
