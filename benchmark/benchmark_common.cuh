#pragma once

#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <chrono>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <fstream>
#include <bucket_policies.cuh>
#include <limits>
#include <random>
#include <string>

class CPUTimer {
   public:
    CPUTimer() = default;

    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }

    double elapsed() {
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = endTime - startTime;
        return elapsed.count();
    }

   private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
};

class GPUTimer {
   public:
    GPUTimer() = default;

    ~GPUTimer() {
        if (startEvent) cudaEventDestroy(startEvent);
        if (stopEvent) cudaEventDestroy(stopEvent);
    }

    GPUTimer(const GPUTimer&) = delete;
    GPUTimer& operator=(const GPUTimer&) = delete;

    GPUTimer(GPUTimer&& other) noexcept
        : startEvent(other.startEvent), stopEvent(other.stopEvent) {
        other.startEvent = nullptr;
        other.stopEvent = nullptr;
    }

    GPUTimer& operator=(GPUTimer&& other) noexcept {
        if (this != &other) {
            if (startEvent) cudaEventDestroy(startEvent);
            if (stopEvent) cudaEventDestroy(stopEvent);
            startEvent = other.startEvent;
            stopEvent = other.stopEvent;
            other.startEvent = nullptr;
            other.stopEvent = nullptr;
        }
        return *this;
    }

    void start() {
        ensureInitialized();
        cudaEventRecord(startEvent);
    }

    double elapsed() {
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
        return static_cast<double>(milliseconds) / 1000.0;
    }

   private:
    void ensureInitialized() {
        if (!startEvent) {
            cudaEventCreate(&startEvent);
            cudaEventCreate(&stopEvent);
        }
    }

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
};

std::pair<size_t, size_t> calculateCapacityAndSize(size_t capacity, double loadFactor) {
    return {capacity, capacity * loadFactor};
}
template <typename Filter, size_t bitsPerTag>
size_t cucoNumBlocks(size_t n) {
    constexpr auto bitsPerWord = sizeof(typename Filter::word_type) * 8;
    return SDIV(n * bitsPerTag, Filter::words_per_block * bitsPerWord);
}

inline size_t getGPUL2CacheSize() {
    static size_t cachedSize = []() {
        int device;
        cudaGetDevice(&device);

        int l2CacheSize;
        cudaDeviceGetAttribute(&l2CacheSize, cudaDevAttrL2CacheSize, device);

        return static_cast<size_t>(l2CacheSize);
    }();

    return cachedSize;
}

template <typename FilterConfig>
inline size_t adaptiveInsert(
    CuckooFilter<FilterConfig>& filter,
    thrust::device_vector<typename FilterConfig::KeyType>& d_keys
) {
    // static size_t threshold = getGPUL2CacheSize() / (FilterConfig::bitsPerTag / CHAR_BIT);
    static constexpr size_t threshold = 1 << 29;

    if (d_keys.size() < threshold) {
        return filter.insertMany(d_keys);
    } else {
        return filter.insertManySorted(d_keys);
    }
}

template <typename T>
void generateKeysGPURange(
    thrust::device_vector<T>& d_output,
    size_t count,
    T min,
    T max,
    unsigned int seed = 99999
) {
    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(count),
        d_output.begin(),
        [=] __device__(size_t idx) {
            thrust::default_random_engine rng(seed);
            thrust::uniform_int_distribution<T> dist(min, max);
            rng.discard(idx);
            return dist(rng);
        }
    );
}

/**
 * @brief Generate random keys on the GPU
 *
 * @tparam T The key type
 * @param d_keys Device vector to fill with random keys
 * @param max Maximum value for generated keys (default: max value of type T)
 * @param seed Random seed (default: 42)
 */
template <typename T>
void generateKeysGPU(
    thrust::device_vector<T>& d_keys,
    T max = std::numeric_limits<T>::max(),
    unsigned int seed = 42
) {
    generateKeysGPURange(d_keys, d_keys.size(), static_cast<T>(1), max, seed);
}

template <typename T>
std::vector<T> generateKeysCPU(
    size_t n,
    unsigned seed = 42,
    T minVal = 1,
    T maxVal = std::numeric_limits<T>::max()
) {
    std::vector<T> keys(n);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<T> dist(minVal, maxVal);
    std::generate(keys.begin(), keys.end(), [&]() { return dist(rng); });
    return keys;
}

size_t getL2CacheSize() {
    std::ifstream cacheFile("/sys/devices/system/cpu/cpu0/cache/index2/size");
    if (cacheFile.is_open()) {
        std::string size_str;
        std::getline(cacheFile, size_str);
        cacheFile.close();

        // Parse the size string (format: "512K", "1M", etc.)
        size_t size = 0;
        char unit = 0;
        if (std::sscanf(size_str.c_str(), "%zu%c", &size, &unit) >= 1) {
            switch (unit) {
                case 'K':
                case 'k':
                    return size * 1024;
                case 'M':
                case 'm':
                    return size * 1024 * 1024;
                case 'G':
                case 'g':
                    return size * 1024 * 1024 * 1024;
                default:
                    return size;
            }
        }
    }

    // Fallback to a reasonable default
    return 512 * 1024;
}

void setCommonCounters(benchmark::State& state, size_t memory, size_t n) {
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = benchmark::Counter(
        static_cast<double>(memory), benchmark::Counter::kDefaults, benchmark::Counter::kIs1024
    );
    state.counters["bits_per_item"] = benchmark::Counter(
        static_cast<double>(memory * 8) / static_cast<double>(n),
        benchmark::Counter::kDefaults,
        benchmark::Counter::kIs1024
    );
    state.counters["fpr_percentage"] = 0.0;
    state.counters["false_positives"] = 0.0;
}

void setFPRCounters(
    benchmark::State& state,
    size_t memory,
    size_t n,
    double fpr,
    size_t falsePositives,
    size_t testSize
) {
    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * testSize));
    state.counters["memory_bytes"] = benchmark::Counter(
        static_cast<double>(memory), benchmark::Counter::kDefaults, benchmark::Counter::kIs1024
    );
    state.counters["bits_per_item"] = benchmark::Counter(
        static_cast<double>(memory * 8) / static_cast<double>(n),
        benchmark::Counter::kDefaults,
        benchmark::Counter::kIs1024
    );
    state.counters["fpr_percentage"] = benchmark::Counter(fpr * 100);
    state.counters["false_positives"] = benchmark::Counter(static_cast<double>(falsePositives));
    state.counters["num_items"] = benchmark::Counter(static_cast<double>(n));
}

template <typename Fixture>
void benchmarkInsertBody(Fixture& fixture, benchmark::State& state) {
    for (auto _ : state) {
        fixture.filter->clear();
        cudaDeviceSynchronize();

        fixture.timer.start();
        size_t inserted = adaptiveInsert(*fixture.filter, fixture.d_keys);
        double elapsed = fixture.timer.elapsed();

        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(inserted);
    }
    fixture.setCounters(state);
}

template <typename Fixture>
void benchmarkQueryBody(Fixture& fixture, benchmark::State& state) {
    adaptiveInsert(*fixture.filter, fixture.d_keys);
    cudaDeviceSynchronize();

    for (auto _ : state) {
        fixture.timer.start();
        fixture.filter->containsMany(fixture.d_keys, fixture.d_output);
        double elapsed = fixture.timer.elapsed();

        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(fixture.d_output.data().get());
    }
    fixture.setCounters(state);
}

template <typename Fixture>
void benchmarkDeleteBody(Fixture& fixture, benchmark::State& state) {
    for (auto _ : state) {
        fixture.filter->clear();
        adaptiveInsert(*fixture.filter, fixture.d_keys);
        cudaDeviceSynchronize();

        fixture.timer.start();
        size_t remaining = fixture.filter->deleteMany(fixture.d_keys, fixture.d_output);
        double elapsed = fixture.timer.elapsed();

        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(remaining);
        benchmark::DoNotOptimize(fixture.d_output.data().get());
    }
    fixture.setCounters(state);
}

template <typename Fixture>
void benchmarkInsertAndQueryBody(Fixture& fixture, benchmark::State& state) {
    for (auto _ : state) {
        fixture.filter->clear();
        cudaDeviceSynchronize();

        fixture.timer.start();
        size_t inserted = adaptiveInsert(*fixture.filter, fixture.d_keys);
        fixture.filter->containsMany(fixture.d_keys, fixture.d_output);
        double elapsed = fixture.timer.elapsed();

        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(inserted);
        benchmark::DoNotOptimize(fixture.d_output.data().get());
    }
    fixture.setCounters(state);
}

template <typename Fixture>
void benchmarkInsertQueryDeleteBody(Fixture& fixture, benchmark::State& state) {
    for (auto _ : state) {
        fixture.filter->clear();
        cudaDeviceSynchronize();

        fixture.timer.start();
        size_t inserted = adaptiveInsert(*fixture.filter, fixture.d_keys);
        fixture.filter->containsMany(fixture.d_keys, fixture.d_output);
        size_t remaining = fixture.filter->deleteMany(fixture.d_keys, fixture.d_output);
        double elapsed = fixture.timer.elapsed();

        state.SetIterationTime(elapsed);
        benchmark::DoNotOptimize(inserted);
        benchmark::DoNotOptimize(remaining);
        benchmark::DoNotOptimize(fixture.d_output.data().get());
    }
    fixture.setCounters(state);
}

template <typename ConfigType, double loadFactor = 0.95>
class CuckooFilterFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    using KeyType = typename ConfigType::KeyType;

    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(state.range(0), loadFactor);
        capacity = cap;
        n = num;

        d_keys.resize(n);
        d_output.resize(n);
        generateKeysGPU(d_keys);

        filter = std::make_unique<CuckooFilter<ConfigType>>(capacity);
        filterMemory = filter->sizeInBytes();
    }

    void TearDown(const benchmark::State&) override {
        filter.reset();
        d_keys.clear();
        d_output.clear();
        d_keys.shrink_to_fit();
        d_output.shrink_to_fit();
    }

    void setCounters(benchmark::State& state) {
        setCommonCounters(state, filterMemory, n);
    }

    size_t capacity;
    size_t n;
    size_t filterMemory;
    thrust::device_vector<KeyType> d_keys;
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<CuckooFilter<ConfigType>> filter;
    GPUTimer timer;
};

#define BENCHMARK_CONFIG                \
    ->RangeMultiplier(2)                \
        ->Range(1 << 16, 1ULL << 28)    \
        ->Unit(benchmark::kMillisecond) \
        ->UseManualTime()               \
        ->Iterations(10)                \
        ->Repetitions(5)                \
        ->ReportAggregatesOnly(true)

#define REGISTER_BENCHMARK(FixtureName, BenchName) \
    BENCHMARK_REGISTER_F(FixtureName, BenchName)   \
    BENCHMARK_CONFIG

#define REGISTER_FUNCTION_BENCHMARK(FuncName) \
    BENCHMARK(FuncName)                       \
    BENCHMARK_CONFIG

#define DEFINE_FILTER_INSERT_BENCHMARK(FixtureName) \
    BENCHMARK_DEFINE_F(FixtureName, Insert)         \
    (benchmark::State & state) {                    \
        benchmarkInsertBody(*this, state);          \
    }

#define DEFINE_FILTER_QUERY_BENCHMARK(FixtureName) \
    BENCHMARK_DEFINE_F(FixtureName, Query)         \
    (benchmark::State & state) {                   \
        benchmarkQueryBody(*this, state);          \
    }

#define DEFINE_FILTER_DELETE_BENCHMARK(FixtureName) \
    BENCHMARK_DEFINE_F(FixtureName, Delete)         \
    (benchmark::State & state) {                    \
        benchmarkDeleteBody(*this, state);          \
    }

#define DEFINE_FILTER_INSERT_AND_QUERY_BENCHMARK(FixtureName) \
    BENCHMARK_DEFINE_F(FixtureName, InsertAndQuery)           \
    (benchmark::State & state) {                              \
        benchmarkInsertAndQueryBody(*this, state);            \
    }

#define DEFINE_FILTER_INSERT_QUERY_DELETE_BENCHMARK(FixtureName) \
    BENCHMARK_DEFINE_F(FixtureName, InsertQueryDelete)           \
    (benchmark::State & state) {                                 \
        benchmarkInsertQueryDeleteBody(*this, state);            \
    }

#define DEFINE_CORE_BENCHMARKS(FixtureName)     \
    DEFINE_FILTER_INSERT_BENCHMARK(FixtureName) \
    DEFINE_FILTER_QUERY_BENCHMARK(FixtureName)  \
    DEFINE_FILTER_DELETE_BENCHMARK(FixtureName)

#define REGISTER_CORE_BENCHMARKS(FixtureName) \
    REGISTER_BENCHMARK(FixtureName, Insert);  \
    REGISTER_BENCHMARK(FixtureName, Query);   \
    REGISTER_BENCHMARK(FixtureName, Delete);

#define DEFINE_ALL_FILTER_BENCHMARKS(FixtureName)         \
    DEFINE_CORE_BENCHMARKS(FixtureName)                   \
    DEFINE_FILTER_INSERT_AND_QUERY_BENCHMARK(FixtureName) \
    DEFINE_FILTER_INSERT_QUERY_DELETE_BENCHMARK(FixtureName)

#define REGISTER_ALL_FILTER_BENCHMARKS(FixtureName)  \
    REGISTER_CORE_BENCHMARKS(FixtureName)            \
    REGISTER_BENCHMARK(FixtureName, InsertAndQuery); \
    REGISTER_BENCHMARK(FixtureName, InsertQueryDelete)

#define DEFINE_AND_REGISTER_CORE_BENCHMARKS(FixtureName) \
    DEFINE_CORE_BENCHMARKS(FixtureName)                  \
    REGISTER_CORE_BENCHMARKS(FixtureName)

#define DEFINE_INSERT_QUERY(FixtureName)        \
    DEFINE_FILTER_INSERT_BENCHMARK(FixtureName) \
    DEFINE_FILTER_QUERY_BENCHMARK(FixtureName)

#define REGISTER_INSERT_QUERY(FixtureName)   \
    REGISTER_BENCHMARK(FixtureName, Insert); \
    REGISTER_BENCHMARK(FixtureName, Query);

#define DEFINE_AND_REGISTER_INSERT_QUERY(FixtureName) \
    DEFINE_INSERT_QUERY(FixtureName)                  \
    REGISTER_INSERT_QUERY(FixtureName)

#define STANDARD_BENCHMARK_MAIN()                                   \
    int main(int argc, char** argv) {                               \
        ::benchmark::Initialize(&argc, argv);                       \
        if (::benchmark::ReportUnrecognizedArguments(argc, argv)) { \
            return 1;                                               \
        }                                                           \
        ::benchmark::RunSpecifiedBenchmarks();                      \
        ::benchmark::Shutdown();                                    \
        fflush(stdout);                                             \
        std::_Exit(0);                                              \
    }
