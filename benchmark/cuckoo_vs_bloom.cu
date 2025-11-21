#include <benchmark/benchmark.h>
#include <cuckoofilter.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <cuco/bloom_filter.cuh>
#include <cuco/utility/cuda_thread_scope.cuh>
#include <cuda/std/cstdint>
#include <hash_strategies.cuh>
#include <helpers.cuh>
#include <random>
#include "benchmark_common.cuh"

namespace bm = benchmark;

using Config = CuckooConfig<uint64_t, 16, 500, 128, 16, XorAltBucketPolicy>;

template <typename Filter>
size_t cucoNumBlocks(size_t n) {
    constexpr auto bitsPerWord = sizeof(typename Filter::word_type) * 8;
    return (n * Config::bitsPerTag) / (Filter::words_per_block * bitsPerWord);
}

template <double loadFactor = 0.95>
class BloomFilterFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    using BloomFilter = cuco::bloom_filter<uint64_t>;

    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(state.range(0), loadFactor);
        capacity = cap;
        n = num;

        const size_t numBlocks = cucoNumBlocks<BloomFilter>(capacity);
        d_keys.resize(n);
        d_output.resize(n);
        generateKeysGPU(d_keys);

        filter = std::make_unique<BloomFilter>(numBlocks);
        filterMemory = filter->block_extent() * BloomFilter::words_per_block *
                       sizeof(typename BloomFilter::word_type);
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
    }

    size_t capacity;
    size_t n;
    size_t filterMemory;
    thrust::device_vector<uint64_t> d_keys;
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<BloomFilter> filter;
    Timer timer;
};

using CFFixture = CuckooFilterFixture<Config>;

BENCHMARK_DEFINE_F(CFFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = adaptiveInsert(*filter, d_keys);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(CFFixture, Query)(bm::State& state) {
    adaptiveInsert(*filter, d_keys);

    for (auto _ : state) {
        timer.start();
        filter->containsMany(d_keys, d_output);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(CFFixture, Delete)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        adaptiveInsert(*filter, d_keys);
        cudaDeviceSynchronize();

        timer.start();
        size_t remaining = filter->deleteMany(d_keys, d_output);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(CFFixture, InsertAndQuery)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = adaptiveInsert(*filter, d_keys);
        filter->containsMany(d_keys, d_output);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(CFFixture, InsertQueryDelete)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = adaptiveInsert(*filter, d_keys);
        filter->containsMany(d_keys, d_output);
        size_t remaining = filter->deleteMany(d_keys, d_output);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

using BBFFixture = BloomFilterFixture<>;

BENCHMARK_DEFINE_F(BBFFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();

        timer.start();
        filter->add(d_keys.begin(), d_keys.end());
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(BBFFixture, Query)(bm::State& state) {
    thrust::device_vector<uint8_t> d_output(n);

    filter->add(d_keys.begin(), d_keys.end());

    for (auto _ : state) {
        timer.start();
        filter->contains(
            d_keys.begin(),
            d_keys.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))
        );
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(BBFFixture, InsertAndQuery)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();

        timer.start();
        filter->add(d_keys.begin(), d_keys.end());
        filter->contains(
            d_keys.begin(),
            d_keys.end(),
            reinterpret_cast<bool*>(thrust::raw_pointer_cast(d_output.data()))
        );
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

static void CF_FPR(bm::State& state) {
    Timer timer;
    auto [capacity, n] = calculateCapacityAndSize(state.range(0), 0.95);

    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU(d_keys);

    auto filter = std::make_unique<CuckooFilter<Config>>(capacity);
    size_t filterMemory = filter->sizeInBytes();
    adaptiveInsert(*filter, d_keys);

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    thrust::device_vector<uint64_t> d_neverInserted(fprTestSize);
    thrust::device_vector<uint8_t> d_output(fprTestSize);

    generateKeysGPURange(
        d_neverInserted, fprTestSize, static_cast<uint64_t>(UINT32_MAX) + 1, UINT64_MAX
    );

    for (auto _ : state) {
        timer.start();
        filter->containsMany(d_neverInserted, d_output);
        double elapsed = timer.stop();

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

static void BBF_FPR(bm::State& state) {
    Timer timer;
    using BloomFilter = cuco::bloom_filter<uint64_t>;
    auto [capacity, n] = calculateCapacityAndSize(state.range(0), 0.95);

    const size_t numBlocks = cucoNumBlocks<BloomFilter>(capacity);
    size_t filterMemory =
        numBlocks * BloomFilter::words_per_block * sizeof(typename BloomFilter::word_type);

    auto filter = std::make_unique<BloomFilter>(numBlocks);
    thrust::device_vector<uint64_t> d_keys(n);
    generateKeysGPU<uint64_t>(d_keys, UINT32_MAX);
    filter->add(d_keys.begin(), d_keys.end());

    size_t fprTestSize = std::min(n, size_t(1'000'000));
    thrust::device_vector<uint64_t> d_neverInserted(fprTestSize);
    thrust::device_vector<uint8_t> d_output(fprTestSize);

    generateKeysGPURange(
        d_neverInserted, fprTestSize, static_cast<uint64_t>(UINT32_MAX) + 1, UINT64_MAX
    );

    for (auto _ : state) {
        timer.start();
        filter->contains(d_neverInserted.begin(), d_neverInserted.end(), d_output.begin());
        double elapsed = timer.stop();

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

REGISTER_BENCHMARK(CFFixture, Insert);
REGISTER_BENCHMARK(BBFFixture, Insert);

REGISTER_BENCHMARK(CFFixture, Query);
REGISTER_BENCHMARK(BBFFixture, Query);

REGISTER_BENCHMARK(CFFixture, Delete);

REGISTER_BENCHMARK(CFFixture, InsertAndQuery);
REGISTER_BENCHMARK(BBFFixture, InsertAndQuery);

REGISTER_BENCHMARK(CFFixture, InsertQueryDelete);

REGISTER_FUNCTION_BENCHMARK(CF_FPR);
REGISTER_FUNCTION_BENCHMARK(BBF_FPR);

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
