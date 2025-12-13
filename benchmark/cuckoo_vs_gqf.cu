#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <cstddef>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <cuda/std/cstdint>

#include <bucket_policies.cuh>
#include <helpers.cuh>
#include <random>
#include "benchmark_common.cuh"

#include <gqf.cuh>
#include <gqf_int.cuh>

namespace bm = benchmark;

using Config = CuckooConfig<uint64_t, 16, 500, 128, 16, XorAltBucketPolicy>;

size_t getQFSizeHost(QF* d_qf) {
    QF h_qf;
    cudaMemcpy(&h_qf, d_qf, sizeof(QF), cudaMemcpyDeviceToHost);

    qfmetadata h_metadata;
    cudaMemcpy(&h_metadata, h_qf.metadata, sizeof(qfmetadata), cudaMemcpyDeviceToHost);

    return h_metadata.total_size_in_bytes;
}

using CFFixture = CuckooFilterFixture<Config>;

void convertGQFResults(thrust::device_vector<uint64_t>& d_results) {
    thrust::device_ptr<uint64_t> d_resultsPtr(d_results.data().get());
    thrust::transform(
        d_resultsPtr, d_resultsPtr + d_results.size(), d_resultsPtr, [] __device__(uint64_t val) {
            return val > 0;
        }
    );
}

class GQFFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    static constexpr double TARGET_LOAD_FACTOR = 0.95;

    void SetUp(const benchmark::State& state) override {
        q = static_cast<uint32_t>(std::log2(state.range(0)));
        capacity = 1ULL << q;
        n = capacity * TARGET_LOAD_FACTOR;

        d_keys.resize(n);
        d_results.resize(n);
        generateKeysGPU(d_keys);

        qf_malloc_device(&qf, q, true);
        filterMemory = getQFSizeHost(qf);
    }

    void TearDown(const benchmark::State&) override {
        if (qf != nullptr) {
            qf_destroy_device(qf);
            qf = nullptr;
        }
        d_keys.clear();
        d_keys.shrink_to_fit();
        d_results.clear();
        d_results.shrink_to_fit();
    }

    void setCounters(benchmark::State& state) const {
        setCommonCounters(state, filterMemory, n);
    }

    uint32_t q;
    size_t capacity;
    size_t n;
    size_t filterMemory;
    QF* qf;
    thrust::device_vector<uint64_t> d_keys;
    thrust::device_vector<uint64_t> d_results;
    GPUTimer timer;
};

static void CF_FPR(bm::State& state) {
    GPUTimer timer;
    auto [capacity, n] = calculateCapacityAndSize(state.range(0), 0.95);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);

    using FPRConfig = CuckooConfig<uint64_t, 16, 500, 128, 16, XorAltBucketPolicy>;

    auto filter = std::make_unique<CuckooFilter<FPRConfig>>(capacity);
    size_t filterMemory = filter->sizeInBytes();
    adaptiveInsert(*filter, d_keys);

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    thrust::device_vector<uint64_t> d_neverInserted(fprTestSize);
    thrust::device_vector<uint8_t> d_output(fprTestSize);

    generateKeysGPURange(
        d_neverInserted,
        fprTestSize,
        static_cast<uint64_t>(UINT16_MAX) + 1,
        static_cast<uint64_t>(UINT32_MAX)
    );

    for (auto _ : state) {
        timer.start();
        filter->containsMany(d_neverInserted, d_output);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }

    size_t falsePositives =
        thrust::reduce(d_output.begin(), d_output.end(), 0ULL, cuda::std::plus<size_t>());
    double fpr = static_cast<double>(falsePositives) / static_cast<double>(fprTestSize);

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * fprTestSize));
    state.counters["fpr_percentage"] = bm::Counter(fpr * 100);
    state.counters["false_positives"] = bm::Counter(static_cast<double>(falsePositives));
    state.counters["bits_per_item"] = bm::Counter(
        static_cast<double>(filterMemory * 8) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
}

BENCHMARK_DEFINE_F(GQFFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        qf_destroy_device(qf);
        cudaFree(qf);  // Free the QF device pointer itself
        qf_malloc_device(&qf, q, true);
        cudaDeviceSynchronize();

        timer.start();
        bulk_insert(qf, n, thrust::raw_pointer_cast(d_keys.data()), 0);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(GQFFixture, Query)(bm::State& state) {
    bulk_insert(qf, n, thrust::raw_pointer_cast(d_keys.data()), 0);
    cudaDeviceSynchronize();

    for (auto _ : state) {
        timer.start();
        bulk_get(
            qf,
            n,
            thrust::raw_pointer_cast(d_keys.data()),
            thrust::raw_pointer_cast(d_results.data())
        );
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_results.data().get());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(GQFFixture, Delete)(bm::State& state) {
    for (auto _ : state) {
        qf_destroy_device(qf);
        qf_malloc_device(&qf, q, true);
        bulk_insert(qf, n, thrust::raw_pointer_cast(d_keys.data()), 0);
        cudaDeviceSynchronize();

        timer.start();
        bulk_delete(qf, n, thrust::raw_pointer_cast(d_keys.data()), 0);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
    }
    setCounters(state);
}

static void GQF_FPR(bm::State& state) {
    GPUTimer timer;
    auto q = static_cast<uint32_t>(std::log2(state.range(0)));
    size_t capacity = 1ULL << q;
    size_t n = capacity * 0.95;

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU<uint64_t>(d_keys, UINT16_MAX);

    QF* qf;
    qf_malloc_device(&qf, q, true);
    bulk_insert(qf, n, thrust::raw_pointer_cast(d_keys.data()), 0);
    cudaDeviceSynchronize();

    size_t filterMemory = getQFSizeHost(qf);

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    thrust::device_vector<uint64_t> d_neverInserted(fprTestSize);

    generateKeysGPURange(
        d_neverInserted,
        fprTestSize,
        static_cast<uint64_t>(UINT16_MAX) + 1,
        static_cast<uint64_t>(UINT32_MAX)
    );

    thrust::device_vector<uint64_t> d_results(fprTestSize);

    for (auto _ : state) {
        timer.start();
        bulk_get(
            qf,
            fprTestSize,
            thrust::raw_pointer_cast(d_neverInserted.data()),
            thrust::raw_pointer_cast(d_results.data())
        );
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_results.data().get());
    }

    convertGQFResults(d_results);

    size_t falsePositives =
        thrust::reduce(d_results.begin(), d_results.end(), 0ULL, thrust::plus<size_t>());
    double fpr = static_cast<double>(falsePositives) / static_cast<double>(fprTestSize);

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * fprTestSize));
    state.counters["fpr_percentage"] = bm::Counter(fpr * 100);
    state.counters["false_positives"] = bm::Counter(static_cast<double>(falsePositives));
    state.counters["bits_per_item"] = bm::Counter(
        static_cast<double>(filterMemory * 8) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );

    qf_destroy_device(qf);
}

DEFINE_AND_REGISTER_CORE_BENCHMARKS(CFFixture)

REGISTER_BENCHMARK(GQFFixture, Insert);
REGISTER_BENCHMARK(GQFFixture, Query);
REGISTER_BENCHMARK(GQFFixture, Delete);

REGISTER_FUNCTION_BENCHMARK(CF_FPR);
REGISTER_FUNCTION_BENCHMARK(GQF_FPR);

STANDARD_BENCHMARK_MAIN();
