#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <CuckooFilterMultiGPU.cuh>
#include <fstream>
#include <helpers.cuh>
#include "benchmark_common.cuh"

namespace bm = benchmark;

using Config = CuckooConfig<uint64_t, 16, 500, 256, 16>;

static constexpr double LOAD_FACTOR = 0.95;

/**
 * @brief Load k-mers from binary file
 *
 * Binary format:
 *   - uint64_t: Number of k-mers (N)
 *   - N x uint64_t: Encoded k-mers
 *
 * @param filename Path to binary k-mer file
 * @return Vector of k-mers
 */
std::vector<uint64_t> loadKmerFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Failed to open k-mer file: " + filename);
    }

    // Read count
    uint64_t count;
    file.read(reinterpret_cast<char*>(&count), sizeof(uint64_t));

    if (!file) {
        throw std::runtime_error("Failed to read k-mer count from: " + filename);
    }

    if (count == 0) {
        throw std::runtime_error("K-mer file is empty: " + filename);
    }

    // Read k-mers
    std::vector<uint64_t> kmers(count);
    file.read(reinterpret_cast<char*>(kmers.data()), count * sizeof(uint64_t));

    if (!file) {
        throw std::runtime_error("Failed to read k-mers from: " + filename);
    }

    return kmers;
}

/**
 * @brief Fixture for k-mer benchmarks with single GPU
 */
template <typename ConfigType>
class KmerFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    using KeyType = typename ConfigType::KeyType;

    void SetUp(const benchmark::State&) override {
        if (kmerData.empty()) {
            throw std::runtime_error("K-mer data not loaded. Call loadDataset() first.");
        }

        n = kmerData.size();
        capacity = static_cast<size_t>(n / LOAD_FACTOR);

        d_keys.resize(n);
        thrust::copy(kmerData.begin(), kmerData.end(), d_keys.begin());

        d_output.resize(n);

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

    static void loadDataset(const std::string& filename) {
        kmerData = loadKmerFile(filename);
    }

    static std::vector<uint64_t> kmerData;
    size_t capacity;
    size_t n;
    size_t filterMemory;
    thrust::device_vector<KeyType> d_keys;
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<CuckooFilter<ConfigType>> filter;
    GPUTimer timer;
};

// Static member initialization
template <typename ConfigType>
std::vector<uint64_t> KmerFixture<ConfigType>::kmerData;

/**
 * @brief Fixture for multi-GPU k-mer benchmarks
 */
template <typename ConfigType>
class KmerMultiGPUFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    using KeyType = typename ConfigType::KeyType;

    void SetUp(const benchmark::State&) override {
        if (kmerData.empty()) {
            throw std::runtime_error("K-mer data not loaded. Call loadDataset() first.");
        }

        n = kmerData.size();

        int deviceCount;
        CUDA_CALL(cudaGetDeviceCount(&deviceCount));
        numGPUs = static_cast<size_t>(deviceCount);

        capacity = static_cast<size_t>(n / LOAD_FACTOR);

        h_keys.resize(n);
        std::copy(kmerData.begin(), kmerData.end(), h_keys.begin());

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

    static void loadDataset(const std::string& filename) {
        kmerData = loadKmerFile(filename);
    }

    static std::vector<uint64_t> kmerData;
    size_t numGPUs;
    size_t capacity;
    size_t n;
    size_t filterMemory;
    thrust::host_vector<KeyType> h_keys;
    thrust::host_vector<bool> h_output;
    std::unique_ptr<CuckooFilterMultiGPU<ConfigType>> filter;
    CPUTimer timer;
};

template <typename ConfigType>
std::vector<uint64_t> KmerMultiGPUFixture<ConfigType>::kmerData;

using SingleGPUFixture = KmerFixture<Config>;
using MultiGPUFixture = KmerMultiGPUFixture<Config>;

BENCHMARK_DEFINE_F(SingleGPUFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = adaptiveInsert(*filter, d_keys);
        double elapsed = timer.elapsed();

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
        double elapsed = timer.elapsed();

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
        double elapsed = timer.elapsed();

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
        double elapsed = timer.elapsed();

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
        double elapsed = timer.elapsed();

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
        double elapsed = timer.elapsed();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(h_output.data());
    }
    setCounters(state);
}

#define KMER_CONFIG                 \
    ->Unit(benchmark::kMillisecond) \
        ->UseManualTime()           \
        ->Iterations(10)            \
        ->Repetitions(5)            \
        ->ReportAggregatesOnly(true);

int main(int argc, char** argv) {
    std::string kmerFile;
    bool useMultiGPU = false;

    // Parse our custom arguments and remove them from argv
    std::vector<char*> remainingArgs;
    remainingArgs.push_back(argv[0]);  // Keep program name

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--kmer-file" && i + 1 < argc) {
            kmerFile = argv[i + 1];
            ++i;  // Skip the value
        } else if (arg == "--multi-gpu") {
            useMultiGPU = true;
        } else {
            // Pass through to Google Benchmark
            remainingArgs.push_back(argv[i]);
        }
    }

    if (kmerFile.empty()) {
        std::cerr << "Error: --kmer-file argument required\n";
        std::cerr << "Usage: " << argv[0]
                  << " --kmer-file <path> [--multi-gpu] [benchmark options]\n";
        return 1;
    }

    try {
        if (useMultiGPU) {
            MultiGPUFixture::loadDataset(kmerFile);
            std::cout << "Loaded " << MultiGPUFixture::kmerData.size()
                      << " k-mers for multi-GPU benchmarking\n";

            BENCHMARK_REGISTER_F(MultiGPUFixture, Insert)
            KMER_CONFIG;

            BENCHMARK_REGISTER_F(MultiGPUFixture, Query)
            KMER_CONFIG

            BENCHMARK_REGISTER_F(MultiGPUFixture, Delete)
            KMER_CONFIG;

        } else {
            SingleGPUFixture::loadDataset(kmerFile);
            std::cout << "Loaded " << SingleGPUFixture::kmerData.size()
                      << " k-mers for single-GPU benchmarking\n";

            BENCHMARK_REGISTER_F(SingleGPUFixture, Insert)
            KMER_CONFIG;

            BENCHMARK_REGISTER_F(SingleGPUFixture, Query)
            KMER_CONFIG;

            BENCHMARK_REGISTER_F(SingleGPUFixture, Delete)
            KMER_CONFIG;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading k-mer file: " << e.what() << "\n";
        return 1;
    }

    // Update argc and argv for Google Benchmark
    int newArgc = static_cast<int>(remainingArgs.size());
    char** newArgv = remainingArgs.data();

    ::benchmark::Initialize(&newArgc, newArgv);
    if (::benchmark::ReportUnrecognizedArguments(newArgc, newArgv)) {
        return 1;
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    fflush(stdout);
    std::_Exit(0);
}
