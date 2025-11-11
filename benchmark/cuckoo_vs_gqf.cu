#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <cstddef>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <cuda/std/cstdint>
#include <hash_strategies.cuh>
#include <helpers.cuh>
#include <quotientFilter.cuh>
#include <random>
#include "benchmark_common.cuh"

namespace bm = benchmark;

constexpr double TARGET_LOAD_FACTOR = 0.95;
constexpr unsigned int QF_RBITS = 13;
using Config = CuckooConfig<uint32_t, 16, 500, 128, 16, XorAltBucketPolicy>;

size_t calcQuotientFilterMemory(unsigned int q, unsigned int r) {
    size_t tableBits = (1ULL << q) * (r + 3);
    size_t tableSlots = tableBits / 8;
    return static_cast<size_t>(tableSlots * 1.1);  // 10% overflow allowance
}

void convertKeysToUint(
    const thrust::device_vector<uint64_t>& d_keys_64,
    thrust::device_vector<unsigned int>& d_keys_32
) {
    thrust::transform(
        d_keys_64.begin(), d_keys_64.end(), d_keys_32.begin(), [] __device__(uint64_t key) {
            return static_cast<unsigned int>(key & 0xFFFFFFFF);
        }
    );
}

static void CuckooFilter_Insert(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    CuckooFilter<Config> filter(capacity);

    size_t filterMemory = filter.sizeInBytes();

    for (auto _ : state) {
        state.PauseTiming();
        filter.clear();
        state.ResumeTiming();

        size_t inserted = filter.insertMany(d_keys);
        cudaDeviceSynchronize();
        bm::DoNotOptimize(inserted);
    }

    setCommonCounters(state, filterMemory, n);
}

static void CuckooFilter_Query(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    CuckooFilter<Config> filter(capacity);
    thrust::device_vector<uint8_t> d_output(n);

    filter.insertMany(d_keys);

    size_t filterMemory = filter.sizeInBytes();

    for (auto _ : state) {
        filter.containsMany(d_keys, d_output);
        cudaDeviceSynchronize();
        bm::DoNotOptimize(d_output.data().get());
    }

    setCommonCounters(state, filterMemory, n);
}

static void CuckooFilter_Delete(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    CuckooFilter<Config> filter(capacity);
    thrust::device_vector<uint8_t> d_output(n);

    size_t filterMemory = filter.sizeInBytes();

    for (auto _ : state) {
        state.PauseTiming();
        filter.clear();
        filter.insertMany(d_keys);
        state.ResumeTiming();

        size_t remaining = filter.deleteMany(d_keys, d_output);
        cudaDeviceSynchronize();
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(d_output.data().get());
    }

    setCommonCounters(state, filterMemory, n);
}

static void CuckooFilter_InsertAndQuery(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    thrust::device_vector<uint8_t> d_output(n);
    CuckooFilter<Config> filter(capacity);

    size_t filterMemory = filter.sizeInBytes();

    for (auto _ : state) {
        state.PauseTiming();
        filter.clear();
        state.ResumeTiming();

        size_t inserted = filter.insertMany(d_keys);
        filter.containsMany(d_keys, d_output);

        cudaDeviceSynchronize();

        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(d_output.data().get());
    }

    setCommonCounters(state, filterMemory, n);
}

static void CuckooFilter_InsertQueryDelete(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    thrust::device_vector<uint8_t> d_output(n);
    CuckooFilter<Config> filter(capacity);

    size_t filterMemory = filter.sizeInBytes();

    for (auto _ : state) {
        state.PauseTiming();
        filter.clear();
        state.ResumeTiming();

        size_t inserted = filter.insertMany(d_keys);
        filter.containsMany(d_keys, d_output);
        size_t remaining = filter.deleteMany(d_keys, d_output);

        cudaDeviceSynchronize();

        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(d_output.data().get());
    }

    setCommonCounters(state, filterMemory, n);
}

static void CuckooFilter_FalsePositiveRate(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU<uint64_t>(d_keys, UINT32_MAX);

    CuckooFilter<Config> filter(capacity);
    filter.insertMany(d_keys);

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    thrust::device_vector<uint64_t> d_neverInserted(fprTestSize);
    thrust::device_vector<uint8_t> d_output(fprTestSize);

    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(fprTestSize),
        d_neverInserted.begin(),
        [] __device__(size_t idx) {
            thrust::default_random_engine rng(99999);
            thrust::uniform_int_distribution<uint64_t> dist(
                static_cast<uint64_t>(UINT32_MAX) + 1, UINT64_MAX
            );
            rng.discard(idx);
            return dist(rng);
        }
    );

    size_t filterMemory = filter.sizeInBytes();

    for (auto _ : state) {
        filter.containsMany(d_neverInserted, d_output);
        cudaDeviceSynchronize();
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

static void QuotientFilter_BulkBuild(bm::State& state) {
    auto q = static_cast<unsigned int>(std::log2(state.range(0)));
    size_t capacity = 1ULL << q;
    size_t n = capacity * TARGET_LOAD_FACTOR;

    thrust::device_vector<unsigned int> d_keys(n);
    generateKeysGPU(d_keys);

    struct quotient_filter qf;
    initFilterGPU(&qf, q, QF_RBITS);

    size_t filterMemory = calcQuotientFilterMemory(q, QF_RBITS);

    for (auto _ : state) {
        state.PauseTiming();
        cudaMemset(qf.table, 0, filterMemory);
        state.ResumeTiming();

        float time = bulkBuildSegmentedLayouts(
            qf, static_cast<int>(n), thrust::raw_pointer_cast(d_keys.data()), false
        );
        cudaDeviceSynchronize();
        bm::DoNotOptimize(time);
    }

    setCommonCounters(state, filterMemory, n);

    if (qf.table != nullptr) {
        cudaFree(qf.table);
    }
}

static void QuotientFilter_Insert(bm::State& state) {
    auto q = static_cast<uint32_t>(std::log2(state.range(0)));
    size_t capacity = 1ULL << q;
    size_t n = capacity * TARGET_LOAD_FACTOR;

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);

    struct quotient_filter qf;
    initFilterGPU(&qf, q, QF_RBITS);

    size_t filterMemory = calcQuotientFilterMemory(q, QF_RBITS);

    for (auto _ : state) {
        state.PauseTiming();
        cudaMemset(qf.table, 0, filterMemory);
        state.ResumeTiming();

        float time = insert(qf, static_cast<int>(n), thrust::raw_pointer_cast(d_keys.data()));
        cudaDeviceSynchronize();
        bm::DoNotOptimize(time);
    }

    setCommonCounters(state, filterMemory, n);

    if (qf.table != nullptr) {
        cudaFree(qf.table);
    }
}

static void QuotientFilter_Query_Sorted(bm::State& state) {
    auto q = static_cast<uint32_t>(std::log2(state.range(0)));
    size_t capacity = 1ULL << q;
    size_t n = capacity * TARGET_LOAD_FACTOR;

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);

    struct quotient_filter qf;
    initFilterGPU(&qf, q, QF_RBITS);

    bulkBuildSegmentedLayouts(
        qf, static_cast<int>(n), thrust::raw_pointer_cast(d_keys.data()), false
    );

    thrust::device_vector<unsigned int> d_results(n);

    size_t filterMemory = calcQuotientFilterMemory(q, QF_RBITS);

    for (auto _ : state) {
        float time = launchSortedLookups(
            qf,
            static_cast<int>(n),
            thrust::raw_pointer_cast(d_keys.data()),
            thrust::raw_pointer_cast(d_results.data())
        );
        cudaDeviceSynchronize();
        bm::DoNotOptimize(time);
        bm::DoNotOptimize(d_results.data().get());
    }

    setCommonCounters(state, filterMemory, n);

    if (qf.table != nullptr) {
        cudaFree(qf.table);
    }
}

static void QuotientFilter_Query_Unsorted(bm::State& state) {
    auto q = static_cast<uint32_t>(std::log2(state.range(0)));
    size_t capacity = 1ULL << q;
    size_t n = capacity * TARGET_LOAD_FACTOR;

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);

    struct quotient_filter qf;
    initFilterGPU(&qf, q, QF_RBITS);

    bulkBuildSegmentedLayouts(
        qf, static_cast<int>(n), thrust::raw_pointer_cast(d_keys.data()), false
    );

    thrust::device_vector<unsigned int> d_results(n);

    size_t filterMemory = calcQuotientFilterMemory(q, QF_RBITS);

    for (auto _ : state) {
        float time = launchUnsortedLookups(
            qf,
            static_cast<int>(n),
            thrust::raw_pointer_cast(d_keys.data()),
            thrust::raw_pointer_cast(d_results.data())
        );
        cudaDeviceSynchronize();
        bm::DoNotOptimize(time);
        bm::DoNotOptimize(d_results.data().get());
    }

    setCommonCounters(state, filterMemory, n);

    if (qf.table != nullptr) {
        cudaFree(qf.table);
    }
}

static void QuotientFilter_Delete(bm::State& state) {
    auto q = static_cast<uint32_t>(std::log2(state.range(0)));
    size_t capacity = 1ULL << q;
    size_t n = capacity * TARGET_LOAD_FACTOR;

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);

    struct quotient_filter qf;
    initFilterGPU(&qf, q, QF_RBITS);

    size_t filterMemory = calcQuotientFilterMemory(q, QF_RBITS);

    for (auto _ : state) {
        state.PauseTiming();
        cudaMemset(qf.table, 0, filterMemory);
        bulkBuildSegmentedLayouts(
            qf, static_cast<int>(n), thrust::raw_pointer_cast(d_keys.data()), false
        );
        state.ResumeTiming();

        float time =
            superclusterDeletes(qf, static_cast<int>(n), thrust::raw_pointer_cast(d_keys.data()));
        cudaDeviceSynchronize();
        bm::DoNotOptimize(time);
    }

    setCommonCounters(state, filterMemory, n);

    if (qf.table != nullptr) {
        cudaFree(qf.table);
    }
}

static void QuotientFilter_BuildAndQuery(bm::State& state) {
    auto q = static_cast<uint32_t>(std::log2(state.range(0)));
    size_t capacity = 1ULL << q;
    size_t n = capacity * TARGET_LOAD_FACTOR;

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);

    struct quotient_filter qf;
    initFilterGPU(&qf, q, QF_RBITS);

    thrust::device_vector<unsigned int> d_results(n);

    size_t filterMemory = calcQuotientFilterMemory(q, QF_RBITS);

    for (auto _ : state) {
        state.PauseTiming();
        cudaMemset(qf.table, 0, filterMemory);
        state.ResumeTiming();

        float buildTime = bulkBuildSegmentedLayouts(
            qf, static_cast<int>(n), thrust::raw_pointer_cast(d_keys.data()), false
        );
        float queryTime = launchSortedLookups(
            qf,
            static_cast<int>(n),
            thrust::raw_pointer_cast(d_keys.data()),
            thrust::raw_pointer_cast(d_results.data())
        );

        cudaDeviceSynchronize();
        bm::DoNotOptimize(buildTime);
        bm::DoNotOptimize(queryTime);
        bm::DoNotOptimize(d_results.data().get());
    }

    setCommonCounters(state, filterMemory, n);

    if (qf.table != nullptr) {
        cudaFree(qf.table);
    }
}

static void QuotientFilter_BuildQueryDelete(bm::State& state) {
    auto q = static_cast<uint32_t>(std::log2(state.range(0)));
    size_t capacity = 1ULL << q;
    size_t n = capacity * TARGET_LOAD_FACTOR;

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);

    struct quotient_filter qf;
    initFilterGPU(&qf, q, QF_RBITS);

    thrust::device_vector<unsigned int> d_results(n);

    size_t filterMemory = calcQuotientFilterMemory(q, QF_RBITS);

    for (auto _ : state) {
        state.PauseTiming();
        cudaMemset(qf.table, 0, filterMemory);
        state.ResumeTiming();

        float buildTime = bulkBuildSegmentedLayouts(
            qf, static_cast<int>(n), thrust::raw_pointer_cast(d_keys.data()), false
        );
        float queryTime = launchSortedLookups(
            qf,
            static_cast<int>(n),
            thrust::raw_pointer_cast(d_keys.data()),
            thrust::raw_pointer_cast(d_results.data())
        );
        float deleteTime =
            superclusterDeletes(qf, static_cast<int>(n), thrust::raw_pointer_cast(d_keys.data()));

        cudaDeviceSynchronize();
        bm::DoNotOptimize(buildTime);
        bm::DoNotOptimize(queryTime);
        bm::DoNotOptimize(deleteTime);
        bm::DoNotOptimize(d_results.data().get());
    }

    setCommonCounters(state, filterMemory, n);

    if (qf.table != nullptr) {
        cudaFree(qf.table);
    }
}

// FIXME: This segfaults on the GPU for some reason
// It also just kind of sucks that we have to use uint32_t for the keys because there aren't that
// many of them
static void QuotientFilter_FalsePositiveRate(bm::State& state) {
    auto q = static_cast<uint32_t>(std::log2(state.range(0)));
    size_t capacity = 1ULL << q;
    size_t n = capacity * TARGET_LOAD_FACTOR;

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU<uint32_t>(d_keys, UINT16_MAX);

    struct quotient_filter qf;
    initFilterGPU(&qf, q, QF_RBITS);

    bulkBuildSegmentedLayouts(
        qf, static_cast<int>(n), thrust::raw_pointer_cast(d_keys.data()), false
    );

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    thrust::device_vector<uint32_t> d_neverInserted(fprTestSize);

    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(fprTestSize),
        d_neverInserted.begin(),
        [] __device__(size_t idx) {
            thrust::default_random_engine rng(99999);
            thrust::uniform_int_distribution<uint32_t> dist(
                static_cast<uint32_t>(UINT16_MAX) + 1, UINT32_MAX
            );
            rng.discard(idx);
            return dist(rng);
        }
    );

    thrust::device_vector<unsigned int> d_results(fprTestSize);

    size_t filterMemory = calcQuotientFilterMemory(q, QF_RBITS);

    for (auto _ : state) {
        float time = launchSortedLookups(
            qf,
            static_cast<int>(fprTestSize),
            thrust::raw_pointer_cast(d_neverInserted.data()),
            thrust::raw_pointer_cast(d_results.data())
        );
        cudaDeviceSynchronize();
        bm::DoNotOptimize(time);
        bm::DoNotOptimize(d_results.data().get());
    }

    thrust::device_vector<unsigned int> d_found(fprTestSize);
    thrust::transform(
        d_results.begin(), d_results.end(), d_found.begin(), [] __device__(unsigned int val) {
            return (val != UINT_MAX) ? 1u : 0u;
        }
    );

    size_t falsePositives =
        thrust::reduce(d_found.begin(), d_found.end(), 0ULL, thrust::plus<size_t>());
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

    if (qf.table != nullptr) {
        cudaFree(qf.table);
    }
}

BENCHMARK(CuckooFilter_Insert)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1ULL << 18)
    ->Unit(bm::kMillisecond);

BENCHMARK(CuckooFilter_Query)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1ULL << 18)
    ->Unit(bm::kMillisecond);

BENCHMARK(CuckooFilter_Delete)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1ULL << 18)
    ->Unit(bm::kMillisecond);

BENCHMARK(CuckooFilter_InsertAndQuery)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1ULL << 18)
    ->Unit(bm::kMillisecond);

BENCHMARK(CuckooFilter_InsertQueryDelete)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1ULL << 18)
    ->Unit(bm::kMillisecond);

BENCHMARK(CuckooFilter_FalsePositiveRate)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1ULL << 18)
    ->Unit(bm::kMillisecond);

BENCHMARK(QuotientFilter_BulkBuild)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1ULL << 18)
    ->Unit(bm::kMillisecond);

BENCHMARK(QuotientFilter_Insert)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1ULL << 18)
    ->Unit(bm::kMillisecond);

BENCHMARK(QuotientFilter_Query_Sorted)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1ULL << 18)
    ->Unit(bm::kMillisecond);

BENCHMARK(QuotientFilter_Query_Unsorted)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1ULL << 18)
    ->Unit(bm::kMillisecond);

BENCHMARK(QuotientFilter_Delete)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1ULL << 18)
    ->Unit(bm::kMillisecond);

BENCHMARK(QuotientFilter_BuildAndQuery)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1ULL << 18)
    ->Unit(bm::kMillisecond);

BENCHMARK(QuotientFilter_BuildQueryDelete)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1ULL << 18)
    ->Unit(bm::kMillisecond);

BENCHMARK(QuotientFilter_FalsePositiveRate)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1ULL << 18)
    ->Unit(bm::kMillisecond);

BENCHMARK_MAIN();
