#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <bulk_tcf_host.cuh>
#include <CuckooFilter.cuh>
#include <cuco/bloom_filter.cuh>
#include <cuda/std/cstdint>
#include <gqf.cuh>
#include <gqf_int.cuh>
#include <random>
#include <thread>
#include "benchmark_common.cuh"

namespace bm = benchmark;

constexpr size_t FIXED_CAPACITY = 1ULL << 28;
constexpr size_t FPR_TEST_SIZE = 10'000'000;

const size_t L2_CACHE_SIZE = getL2CacheSize();

template <size_t bitsPerTag>
using GPUCuckooConfig = CuckooConfig<uint64_t, bitsPerTag, 500, 128, 16, XorAltBucketPolicy>;

size_t getQFSizeHost(QF* d_qf) {
    QF h_qf;
    cudaMemcpy(&h_qf, d_qf, sizeof(QF), cudaMemcpyDeviceToHost);

    qfmetadata h_metadata;
    cudaMemcpy(&h_metadata, h_qf.metadata, sizeof(qfmetadata), cudaMemcpyDeviceToHost);

    return h_metadata.total_size_in_bytes;
}

void convertGQFResults(thrust::device_vector<uint64_t>& d_results) {
    thrust::device_ptr<uint64_t> d_resultsPtr(d_results.data().get());
    thrust::transform(
        d_resultsPtr, d_resultsPtr + d_results.size(), d_resultsPtr, [] __device__(uint64_t val) {
            return val > 0;
        }
    );
}

void setFPRSweepCounters(
    benchmark::State& state,
    size_t memory,
    size_t n,
    double fpr,
    size_t falsePositives,
    size_t testSize,
    size_t bitsPerTag,
    double loadFactor
) {
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * testSize));
    state.counters["memory_bytes"] = benchmark::Counter(
        static_cast<double>(memory), benchmark::Counter::kDefaults, benchmark::Counter::kIs1024
    );
    state.counters["bits_per_item"] =
        benchmark::Counter(static_cast<double>(memory * 8) / static_cast<double>(n));
    state.counters["fpr_percentage"] = benchmark::Counter(fpr * 100);
    state.counters["fpr_log2"] = benchmark::Counter(fpr > 0 ? std::log2(fpr) : -30);
    state.counters["false_positives"] = benchmark::Counter(static_cast<double>(falsePositives));
    state.counters["num_items"] = benchmark::Counter(static_cast<double>(n));
    state.counters["fingerprint_bits"] = benchmark::Counter(static_cast<double>(bitsPerTag));
    state.counters["load_factor"] = benchmark::Counter(loadFactor * 100);
}

struct FPRTestData {
    thrust::device_vector<uint64_t> d_negative_keys;
    std::vector<uint64_t> h_negative_keys;

    FPRTestData() : d_negative_keys(FPR_TEST_SIZE) {
        generateKeysGPURange(d_negative_keys, FPR_TEST_SIZE, uint64_t(UINT32_MAX) + 1, UINT64_MAX);
        h_negative_keys =
            generateKeysCPU<uint64_t>(FPR_TEST_SIZE, 99999, uint64_t(UINT32_MAX) + 1, UINT64_MAX);
    }
};

// Shared test data to avoid regenerating for each benchmark
static FPRTestData& getFPRTestData() {
    static FPRTestData data;
    return data;
}

template <size_t bitsPerTag, int loadFactorPercent>
static void GPUCF_FPR_Sweep(bm::State& state) {
    using Config = GPUCuckooConfig<bitsPerTag>;
    constexpr double loadFactor = loadFactorPercent / 100.0;

    GPUTimer timer;
    auto n = static_cast<size_t>(FIXED_CAPACITY * loadFactor);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));

    auto filter = std::make_unique<CuckooFilter<Config>>(FIXED_CAPACITY);
    size_t filterMemory = filter->sizeInBytes();
    adaptiveInsert(*filter, d_keys);

    auto& testData = getFPRTestData();
    thrust::device_vector<uint8_t> d_output(FPR_TEST_SIZE);

    size_t totalFalsePositives = 0;
    size_t iterations = 0;

    for (auto _ : state) {
        timer.start();
        filter->containsMany(testData.d_negative_keys, d_output);
        double elapsed = timer.elapsed();

        size_t falsePositives =
            thrust::reduce(d_output.begin(), d_output.end(), 0ULL, cuda::std::plus<size_t>());
        totalFalsePositives += falsePositives;
        ++iterations;

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }

    double avgFalsePositives =
        static_cast<double>(totalFalsePositives) / static_cast<double>(iterations);
    double fpr = avgFalsePositives / static_cast<double>(FPR_TEST_SIZE);

    setFPRSweepCounters(
        state,
        filterMemory,
        n,
        fpr,
        static_cast<size_t>(avgFalsePositives),
        FPR_TEST_SIZE,
        bitsPerTag,
        loadFactor
    );
}

template <size_t bitsPerTag, int loadFactorPercent>
static void GPUCF_Insert_Sweep(bm::State& state) {
    using Config = GPUCuckooConfig<bitsPerTag>;
    constexpr double loadFactor = loadFactorPercent / 100.0;

    GPUTimer timer;
    auto n = static_cast<size_t>(FIXED_CAPACITY * loadFactor);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));

    size_t filterMemory = 0;

    for (auto _ : state) {
        auto filter = std::make_unique<CuckooFilter<Config>>(FIXED_CAPACITY);
        filterMemory = filter->sizeInBytes();

        timer.start();
        adaptiveInsert(*filter, d_keys);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(filter.get());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = benchmark::Counter(
        static_cast<double>(filterMemory),
        benchmark::Counter::kDefaults,
        benchmark::Counter::kIs1024
    );
    state.counters["bits_per_item"] =
        benchmark::Counter(static_cast<double>(filterMemory * 8) / static_cast<double>(n));
    state.counters["num_items"] = benchmark::Counter(static_cast<double>(n));
    state.counters["fingerprint_bits"] = benchmark::Counter(static_cast<double>(bitsPerTag));
    state.counters["load_factor"] = benchmark::Counter(loadFactor * 100);
}

template <size_t bitsPerTag, int loadFactorPercent>
static void GPUCF_PositiveQuery_Sweep(bm::State& state) {
    using Config = GPUCuckooConfig<bitsPerTag>;
    constexpr double loadFactor = loadFactorPercent / 100.0;

    GPUTimer timer;
    auto n = static_cast<size_t>(FIXED_CAPACITY * loadFactor);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));

    auto filter = std::make_unique<CuckooFilter<Config>>(FIXED_CAPACITY);
    size_t filterMemory = filter->sizeInBytes();
    adaptiveInsert(*filter, d_keys);

    // Use the inserted keys for positive lookups
    thrust::device_vector<uint8_t> d_output(n);

    for (auto _ : state) {
        timer.start();
        filter->containsMany(d_keys, d_output);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = benchmark::Counter(
        static_cast<double>(filterMemory),
        benchmark::Counter::kDefaults,
        benchmark::Counter::kIs1024
    );
    state.counters["bits_per_item"] =
        benchmark::Counter(static_cast<double>(filterMemory * 8) / static_cast<double>(n));
    state.counters["num_items"] = benchmark::Counter(static_cast<double>(n));
    state.counters["fingerprint_bits"] = benchmark::Counter(static_cast<double>(bitsPerTag));
    state.counters["load_factor"] = benchmark::Counter(loadFactor * 100);
}

using BloomFilter = cuco::bloom_filter<uint64_t>;

template <int bitsPerElement, int loadFactorPercent>
static void Bloom_FPR_Sweep(bm::State& state) {
    constexpr double loadFactor = loadFactorPercent / 100.0;
    constexpr size_t bitsPerBlock =
        BloomFilter::words_per_block * sizeof(typename BloomFilter::word_type) * 8;

    GPUTimer timer;
    auto n = static_cast<size_t>(FIXED_CAPACITY * loadFactor);
    size_t numBlocks = std::max(1UL, SDIV(n * bitsPerElement, bitsPerBlock));

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));

    auto filter = std::make_unique<BloomFilter>(numBlocks);
    size_t filterMemory = filter->block_extent() * BloomFilter::words_per_block *
                          sizeof(typename BloomFilter::word_type);
    filter->add(d_keys.begin(), d_keys.end());

    auto& testData = getFPRTestData();
    thrust::device_vector<uint8_t> d_output(FPR_TEST_SIZE);

    size_t totalFalsePositives = 0;
    size_t iterations = 0;

    for (auto _ : state) {
        timer.start();
        filter->contains(
            testData.d_negative_keys.begin(),
            testData.d_negative_keys.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))
        );
        double elapsed = timer.elapsed();

        size_t falsePositives =
            thrust::reduce(d_output.begin(), d_output.end(), 0ULL, cuda::std::plus<size_t>());
        totalFalsePositives += falsePositives;
        ++iterations;

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }

    double avgFalsePositives =
        static_cast<double>(totalFalsePositives) / static_cast<double>(iterations);
    double fpr = avgFalsePositives / static_cast<double>(FPR_TEST_SIZE);

    setFPRSweepCounters(
        state,
        filterMemory,
        n,
        fpr,
        static_cast<size_t>(avgFalsePositives),
        FPR_TEST_SIZE,
        bitsPerElement,
        loadFactor
    );
}

template <typename FingerprintType, int loadFactorPercent>
static void TCF_FPR_Sweep(bm::State& state) {
    using TCFType = host_bulk_tcf<uint64_t, FingerprintType>;
    constexpr double loadFactor = loadFactorPercent / 100.0;
    constexpr size_t fingerprintBits = sizeof(FingerprintType) * 8;

    // TCF can only hold 0.85 * capacity items
    constexpr double TCF_CAPACITY_FACTOR = 0.85;

    GPUTimer timer;
    auto n = static_cast<size_t>(FIXED_CAPACITY * loadFactor);
    auto requiredUsableCapacity = static_cast<size_t>(n / loadFactor);
    auto capacity = static_cast<size_t>(requiredUsableCapacity / TCF_CAPACITY_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));

    TCFType* filter = TCFType::host_build_tcf(capacity);
    size_t filterMemory = capacity * sizeof(FingerprintType);

    uint64_t* d_misses;
    cudaMalloc(&d_misses, sizeof(uint64_t));
    cudaMemset(d_misses, 0, sizeof(uint64_t));
    filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);

    auto& testData = getFPRTestData();
    size_t totalFalsePositives = 0;
    size_t iterations = 0;

    for (auto _ : state) {
        timer.start();
        bool* d_output = filter->bulk_query(
            thrust::raw_pointer_cast(testData.d_negative_keys.data()), FPR_TEST_SIZE
        );
        double elapsed = timer.elapsed();

        thrust::device_ptr<bool> d_outputPtr(d_output);
        size_t falsePositives = thrust::reduce(
            d_outputPtr, d_outputPtr + FPR_TEST_SIZE, 0ULL, cuda::std::plus<size_t>()
        );
        totalFalsePositives += falsePositives;
        ++iterations;

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output);
        cudaFree(d_output);
    }

    double avgFalsePositives =
        static_cast<double>(totalFalsePositives) / static_cast<double>(iterations);
    double fpr = avgFalsePositives / static_cast<double>(FPR_TEST_SIZE);

    setFPRSweepCounters(
        state,
        filterMemory,
        n,
        fpr,
        static_cast<size_t>(avgFalsePositives),
        FPR_TEST_SIZE,
        fingerprintBits,
        loadFactor
    );

    cudaFree(d_misses);
    TCFType::host_free_tcf(filter);
}

// GQF FPR Sweep
// Note: GQF bit width (QF_BITS_PER_SLOT) is set at compile time.
// We build separate executables for each configuration and link against
// the appropriate GQF library variant.

#ifndef GQF_BITS
    #define GQF_BITS 16  // Default to 16-bit if not specified
#endif

template <int loadFactorPercent>
static void GQF_FPR_Sweep(bm::State& state) {
    constexpr double loadFactor = loadFactorPercent / 100.0;

    GPUTimer timer;
    auto q = static_cast<uint32_t>(std::log2(FIXED_CAPACITY));
    size_t capacity = 1ULL << q;
    auto n = static_cast<size_t>(capacity * loadFactor);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));

    QF* qf;
    qf_malloc_device(&qf, q, true);
    bulk_insert(qf, n, thrust::raw_pointer_cast(d_keys.data()), 0);
    cudaDeviceSynchronize();

    size_t filterMemory = getQFSizeHost(qf);

    auto& testData = getFPRTestData();
    thrust::device_vector<uint64_t> d_results(FPR_TEST_SIZE);

    size_t totalFalsePositives = 0;
    size_t iterations = 0;

    for (auto _ : state) {
        timer.start();
        bulk_get(
            qf,
            FPR_TEST_SIZE,
            thrust::raw_pointer_cast(testData.d_negative_keys.data()),
            thrust::raw_pointer_cast(d_results.data())
        );
        double elapsed = timer.elapsed();

        convertGQFResults(d_results);
        size_t falsePositives =
            thrust::reduce(d_results.begin(), d_results.end(), 0ULL, thrust::plus<size_t>());
        totalFalsePositives += falsePositives;
        ++iterations;

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_results.data().get());
    }

    double avgFalsePositives =
        static_cast<double>(totalFalsePositives) / static_cast<double>(iterations);
    double fpr = avgFalsePositives / static_cast<double>(FPR_TEST_SIZE);

    setFPRSweepCounters(
        state,
        filterMemory,
        n,
        fpr,
        static_cast<size_t>(avgFalsePositives),
        FPR_TEST_SIZE,
        GQF_BITS,
        loadFactor
    );

    qf_destroy_device(qf);
}

template <int bitsPerElement, int loadFactorPercent>
static void Bloom_Insert_Sweep(bm::State& state) {
    constexpr double loadFactor = loadFactorPercent / 100.0;
    constexpr size_t bitsPerBlock =
        BloomFilter::words_per_block * sizeof(typename BloomFilter::word_type) * 8;

    GPUTimer timer;
    auto n = static_cast<size_t>(FIXED_CAPACITY * loadFactor);
    size_t numBlocks = std::max(1UL, SDIV(n * bitsPerElement, bitsPerBlock));

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));

    size_t filterMemory = 0;

    for (auto _ : state) {
        auto filter = std::make_unique<BloomFilter>(numBlocks);
        filterMemory = filter->block_extent() * BloomFilter::words_per_block *
                       sizeof(typename BloomFilter::word_type);

        timer.start();
        filter->add(d_keys.begin(), d_keys.end());
        cudaDeviceSynchronize();
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(filter.get());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = benchmark::Counter(
        static_cast<double>(filterMemory),
        benchmark::Counter::kDefaults,
        benchmark::Counter::kIs1024
    );
    state.counters["bits_per_item"] =
        benchmark::Counter(static_cast<double>(filterMemory * 8) / static_cast<double>(n));
    state.counters["num_items"] = benchmark::Counter(static_cast<double>(n));
    state.counters["fingerprint_bits"] = benchmark::Counter(static_cast<double>(bitsPerElement));
    state.counters["load_factor"] = benchmark::Counter(loadFactor * 100);
}

template <typename FingerprintType, int loadFactorPercent>
static void TCF_Insert_Sweep(bm::State& state) {
    using TCFType = host_bulk_tcf<uint64_t, FingerprintType>;
    constexpr double loadFactor = loadFactorPercent / 100.0;
    constexpr size_t fingerprintBits = sizeof(FingerprintType) * 8;
    constexpr double TCF_CAPACITY_FACTOR = 0.85;

    GPUTimer timer;
    auto n = static_cast<size_t>(FIXED_CAPACITY * loadFactor);
    auto requiredUsableCapacity = static_cast<size_t>(n / loadFactor);
    auto capacity = static_cast<size_t>(requiredUsableCapacity / TCF_CAPACITY_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));

    size_t filterMemory = capacity * sizeof(FingerprintType);

    for (auto _ : state) {
        TCFType* filter = TCFType::host_build_tcf(capacity);

        uint64_t* d_misses;
        cudaMalloc(&d_misses, sizeof(uint64_t));
        cudaMemset(d_misses, 0, sizeof(uint64_t));

        timer.start();
        filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);
        cudaDeviceSynchronize();
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(filter);

        cudaFree(d_misses);
        TCFType::host_free_tcf(filter);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = benchmark::Counter(
        static_cast<double>(filterMemory),
        benchmark::Counter::kDefaults,
        benchmark::Counter::kIs1024
    );
    state.counters["bits_per_item"] =
        benchmark::Counter(static_cast<double>(filterMemory * 8) / static_cast<double>(n));
    state.counters["num_items"] = benchmark::Counter(static_cast<double>(n));
    state.counters["fingerprint_bits"] = benchmark::Counter(static_cast<double>(fingerprintBits));
    state.counters["load_factor"] = benchmark::Counter(loadFactor * 100);
}

template <int loadFactorPercent>
static void GQF_Insert_Sweep(bm::State& state) {
    constexpr double loadFactor = loadFactorPercent / 100.0;

    GPUTimer timer;
    auto q = static_cast<uint32_t>(std::log2(FIXED_CAPACITY));
    size_t capacity = 1ULL << q;
    auto n = static_cast<size_t>(capacity * loadFactor);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));

    size_t filterMemory = 0;

    for (auto _ : state) {
        QF* qf;
        qf_malloc_device(&qf, q, true);
        filterMemory = getQFSizeHost(qf);

        timer.start();
        bulk_insert(qf, n, thrust::raw_pointer_cast(d_keys.data()), 0);
        cudaDeviceSynchronize();
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(qf);

        qf_destroy_device(qf);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = benchmark::Counter(
        static_cast<double>(filterMemory),
        benchmark::Counter::kDefaults,
        benchmark::Counter::kIs1024
    );
    state.counters["bits_per_item"] =
        benchmark::Counter(static_cast<double>(filterMemory * 8) / static_cast<double>(n));
    state.counters["num_items"] = benchmark::Counter(static_cast<double>(n));
    state.counters["fingerprint_bits"] = benchmark::Counter(static_cast<double>(GQF_BITS));
    state.counters["load_factor"] = benchmark::Counter(loadFactor * 100);
}

template <int bitsPerElement, int loadFactorPercent>
static void Bloom_PositiveQuery_Sweep(bm::State& state) {
    constexpr double loadFactor = loadFactorPercent / 100.0;
    constexpr size_t bitsPerBlock =
        BloomFilter::words_per_block * sizeof(typename BloomFilter::word_type) * 8;

    GPUTimer timer;
    auto n = static_cast<size_t>(FIXED_CAPACITY * loadFactor);
    size_t numBlocks = std::max(1UL, SDIV(n * bitsPerElement, bitsPerBlock));

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));

    auto filter = std::make_unique<BloomFilter>(numBlocks);
    size_t filterMemory = filter->block_extent() * BloomFilter::words_per_block *
                          sizeof(typename BloomFilter::word_type);
    filter->add(d_keys.begin(), d_keys.end());

    thrust::device_vector<uint8_t> d_output(n);

    for (auto _ : state) {
        timer.start();
        filter->contains(
            d_keys.begin(),
            d_keys.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))
        );
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = benchmark::Counter(
        static_cast<double>(filterMemory),
        benchmark::Counter::kDefaults,
        benchmark::Counter::kIs1024
    );
    state.counters["bits_per_item"] =
        benchmark::Counter(static_cast<double>(filterMemory * 8) / static_cast<double>(n));
    state.counters["num_items"] = benchmark::Counter(static_cast<double>(n));
    state.counters["fingerprint_bits"] = benchmark::Counter(static_cast<double>(bitsPerElement));
    state.counters["load_factor"] = benchmark::Counter(loadFactor * 100);
}

template <typename FingerprintType, int loadFactorPercent>
static void TCF_PositiveQuery_Sweep(bm::State& state) {
    using TCFType = host_bulk_tcf<uint64_t, FingerprintType>;
    constexpr double loadFactor = loadFactorPercent / 100.0;
    constexpr size_t fingerprintBits = sizeof(FingerprintType) * 8;
    constexpr double TCF_CAPACITY_FACTOR = 0.85;

    GPUTimer timer;
    auto n = static_cast<size_t>(FIXED_CAPACITY * loadFactor);
    auto requiredUsableCapacity = static_cast<size_t>(n / loadFactor);
    auto capacity = static_cast<size_t>(requiredUsableCapacity / TCF_CAPACITY_FACTOR);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));

    TCFType* filter = TCFType::host_build_tcf(capacity);
    size_t filterMemory = capacity * sizeof(FingerprintType);

    uint64_t* d_misses;
    cudaMalloc(&d_misses, sizeof(uint64_t));
    cudaMemset(d_misses, 0, sizeof(uint64_t));
    filter->bulk_insert(thrust::raw_pointer_cast(d_keys.data()), n, d_misses);

    for (auto _ : state) {
        timer.start();
        bool* d_output = filter->bulk_query(thrust::raw_pointer_cast(d_keys.data()), n);
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output);
        cudaFree(d_output);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = benchmark::Counter(
        static_cast<double>(filterMemory),
        benchmark::Counter::kDefaults,
        benchmark::Counter::kIs1024
    );
    state.counters["bits_per_item"] =
        benchmark::Counter(static_cast<double>(filterMemory * 8) / static_cast<double>(n));
    state.counters["num_items"] = benchmark::Counter(static_cast<double>(n));
    state.counters["fingerprint_bits"] = benchmark::Counter(static_cast<double>(fingerprintBits));
    state.counters["load_factor"] = benchmark::Counter(loadFactor * 100);

    cudaFree(d_misses);
    TCFType::host_free_tcf(filter);
}

template <int loadFactorPercent>
static void GQF_PositiveQuery_Sweep(bm::State& state) {
    constexpr double loadFactor = loadFactorPercent / 100.0;

    GPUTimer timer;
    auto q = static_cast<uint32_t>(std::log2(FIXED_CAPACITY));
    size_t capacity = 1ULL << q;
    auto n = static_cast<size_t>(capacity * loadFactor);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPURange(d_keys, n, uint64_t(0), uint64_t(UINT32_MAX));

    QF* qf;
    qf_malloc_device(&qf, q, true);
    bulk_insert(qf, n, thrust::raw_pointer_cast(d_keys.data()), 0);
    cudaDeviceSynchronize();

    size_t filterMemory = getQFSizeHost(qf);
    thrust::device_vector<uint64_t> d_results(n);

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

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = benchmark::Counter(
        static_cast<double>(filterMemory),
        benchmark::Counter::kDefaults,
        benchmark::Counter::kIs1024
    );
    state.counters["bits_per_item"] =
        benchmark::Counter(static_cast<double>(filterMemory * 8) / static_cast<double>(n));
    state.counters["num_items"] = benchmark::Counter(static_cast<double>(n));
    state.counters["fingerprint_bits"] = benchmark::Counter(static_cast<double>(GQF_BITS));
    state.counters["load_factor"] = benchmark::Counter(loadFactor * 100);

    qf_destroy_device(qf);
}

#define FPR_SWEEP_CONFIG     \
    ->Unit(bm::kMillisecond) \
        ->UseManualTime()    \
        ->Iterations(10)     \
        ->Repetitions(5)     \
        ->ReportAggregatesOnly(true)

// Register non-GQF benchmarks (only in the first executable)

#if GQF_BITS == 8

    #define REGISTER_GPUCF_FOR_LOAD_FACTOR(LF)                         \
        BENCHMARK(GPUCF_FPR_Sweep<8, LF>) FPR_SWEEP_CONFIG;            \
        BENCHMARK(GPUCF_FPR_Sweep<16, LF>) FPR_SWEEP_CONFIG;           \
        BENCHMARK(GPUCF_FPR_Sweep<32, LF>) FPR_SWEEP_CONFIG;           \
        BENCHMARK(GPUCF_Insert_Sweep<8, LF>) FPR_SWEEP_CONFIG;         \
        BENCHMARK(GPUCF_Insert_Sweep<16, LF>) FPR_SWEEP_CONFIG;        \
        BENCHMARK(GPUCF_Insert_Sweep<32, LF>) FPR_SWEEP_CONFIG;        \
        BENCHMARK(GPUCF_PositiveQuery_Sweep<8, LF>) FPR_SWEEP_CONFIG;  \
        BENCHMARK(GPUCF_PositiveQuery_Sweep<16, LF>) FPR_SWEEP_CONFIG; \
        BENCHMARK(GPUCF_PositiveQuery_Sweep<32, LF>) FPR_SWEEP_CONFIG;

REGISTER_GPUCF_FOR_LOAD_FACTOR(35)
REGISTER_GPUCF_FOR_LOAD_FACTOR(40)
REGISTER_GPUCF_FOR_LOAD_FACTOR(50)
REGISTER_GPUCF_FOR_LOAD_FACTOR(75)
REGISTER_GPUCF_FOR_LOAD_FACTOR(85)
REGISTER_GPUCF_FOR_LOAD_FACTOR(90)
REGISTER_GPUCF_FOR_LOAD_FACTOR(95)

    #define REGISTER_BLOOM_FOR_LOAD_FACTOR(LF)                         \
        BENCHMARK(Bloom_FPR_Sweep<8, LF>) FPR_SWEEP_CONFIG;            \
        BENCHMARK(Bloom_FPR_Sweep<16, LF>) FPR_SWEEP_CONFIG;           \
        BENCHMARK(Bloom_FPR_Sweep<32, LF>) FPR_SWEEP_CONFIG;           \
        BENCHMARK(Bloom_Insert_Sweep<8, LF>) FPR_SWEEP_CONFIG;         \
        BENCHMARK(Bloom_Insert_Sweep<16, LF>) FPR_SWEEP_CONFIG;        \
        BENCHMARK(Bloom_Insert_Sweep<32, LF>) FPR_SWEEP_CONFIG;        \
        BENCHMARK(Bloom_PositiveQuery_Sweep<8, LF>) FPR_SWEEP_CONFIG;  \
        BENCHMARK(Bloom_PositiveQuery_Sweep<16, LF>) FPR_SWEEP_CONFIG; \
        BENCHMARK(Bloom_PositiveQuery_Sweep<32, LF>) FPR_SWEEP_CONFIG;

REGISTER_BLOOM_FOR_LOAD_FACTOR(35)
REGISTER_BLOOM_FOR_LOAD_FACTOR(40)
REGISTER_BLOOM_FOR_LOAD_FACTOR(50)
REGISTER_BLOOM_FOR_LOAD_FACTOR(75)
REGISTER_BLOOM_FOR_LOAD_FACTOR(85)
REGISTER_BLOOM_FOR_LOAD_FACTOR(90)
REGISTER_BLOOM_FOR_LOAD_FACTOR(95)

    #define REGISTER_TCF_FOR_LOAD_FACTOR(LF)                               \
        BENCHMARK(TCF_FPR_Sweep<uint8_t, LF>) FPR_SWEEP_CONFIG;            \
        BENCHMARK(TCF_FPR_Sweep<uint16_t, LF>) FPR_SWEEP_CONFIG;           \
        BENCHMARK(TCF_FPR_Sweep<uint32_t, LF>) FPR_SWEEP_CONFIG;           \
        BENCHMARK(TCF_Insert_Sweep<uint8_t, LF>) FPR_SWEEP_CONFIG;         \
        BENCHMARK(TCF_Insert_Sweep<uint16_t, LF>) FPR_SWEEP_CONFIG;        \
        BENCHMARK(TCF_Insert_Sweep<uint32_t, LF>) FPR_SWEEP_CONFIG;        \
        BENCHMARK(TCF_PositiveQuery_Sweep<uint8_t, LF>) FPR_SWEEP_CONFIG;  \
        BENCHMARK(TCF_PositiveQuery_Sweep<uint16_t, LF>) FPR_SWEEP_CONFIG; \
        BENCHMARK(TCF_PositiveQuery_Sweep<uint32_t, LF>) FPR_SWEEP_CONFIG;

REGISTER_TCF_FOR_LOAD_FACTOR(35)
REGISTER_TCF_FOR_LOAD_FACTOR(40)
REGISTER_TCF_FOR_LOAD_FACTOR(50)
REGISTER_TCF_FOR_LOAD_FACTOR(75)
REGISTER_TCF_FOR_LOAD_FACTOR(85)
REGISTER_TCF_FOR_LOAD_FACTOR(90)
REGISTER_TCF_FOR_LOAD_FACTOR(95)

#endif  // GQF_BITS == 8

// The bit width is determined by the GQF_BITS macro at compile time

BENCHMARK(GQF_FPR_Sweep<35>) FPR_SWEEP_CONFIG;
BENCHMARK(GQF_FPR_Sweep<40>) FPR_SWEEP_CONFIG;
BENCHMARK(GQF_FPR_Sweep<50>) FPR_SWEEP_CONFIG;
BENCHMARK(GQF_FPR_Sweep<75>) FPR_SWEEP_CONFIG;
BENCHMARK(GQF_FPR_Sweep<85>) FPR_SWEEP_CONFIG;
BENCHMARK(GQF_FPR_Sweep<90>) FPR_SWEEP_CONFIG;
BENCHMARK(GQF_FPR_Sweep<95>) FPR_SWEEP_CONFIG;

BENCHMARK(GQF_Insert_Sweep<35>) FPR_SWEEP_CONFIG;
BENCHMARK(GQF_Insert_Sweep<40>) FPR_SWEEP_CONFIG;
BENCHMARK(GQF_Insert_Sweep<50>) FPR_SWEEP_CONFIG;
BENCHMARK(GQF_Insert_Sweep<75>) FPR_SWEEP_CONFIG;
BENCHMARK(GQF_Insert_Sweep<85>) FPR_SWEEP_CONFIG;
BENCHMARK(GQF_Insert_Sweep<90>) FPR_SWEEP_CONFIG;
BENCHMARK(GQF_Insert_Sweep<95>) FPR_SWEEP_CONFIG;

BENCHMARK(GQF_PositiveQuery_Sweep<35>) FPR_SWEEP_CONFIG;
BENCHMARK(GQF_PositiveQuery_Sweep<40>) FPR_SWEEP_CONFIG;
BENCHMARK(GQF_PositiveQuery_Sweep<50>) FPR_SWEEP_CONFIG;
BENCHMARK(GQF_PositiveQuery_Sweep<75>) FPR_SWEEP_CONFIG;
BENCHMARK(GQF_PositiveQuery_Sweep<85>) FPR_SWEEP_CONFIG;
BENCHMARK(GQF_PositiveQuery_Sweep<90>) FPR_SWEEP_CONFIG;
BENCHMARK(GQF_PositiveQuery_Sweep<95>) FPR_SWEEP_CONFIG;

STANDARD_BENCHMARK_MAIN();
