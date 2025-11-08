#include <benchmark/benchmark.h>
#include <cuda_runtime_api.h>
#include <sys/wait.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <unistd.h>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <CuckooFilterIPC.cuh>
#include <helpers.cuh>
#include "benchmark_common.cuh"

namespace bm = benchmark;

constexpr double TARGET_LOAD_FACTOR = 0.95;
using Config = CuckooConfig<uint32_t, 16, 500, 128, 16, XorHashStrategy>;

static constexpr char SERVER_NAME[] = "benchmark_server";
static pid_t g_serverPid = -1;
static size_t g_currentServerCapacity = 0;

static void startIPCServerProcess(size_t capacity) {
    if (g_serverPid > 0 && g_currentServerCapacity == capacity) {
        return;
    }

    // Stop existing server
    if (g_serverPid > 0) {
        kill(g_serverPid, SIGTERM);
        waitpid(g_serverPid, nullptr, 0);
        g_serverPid = -1;
    }

    g_serverPid = fork();
    if (g_serverPid == 0) {
        std::string capacity_str = std::to_string(capacity);

        char* const argv[] = {
            (char*)"./build/cuckoo-filter-ipc-server", (char*)capacity_str.c_str(), nullptr
        };

        execvp(argv[0], argv);

        perror("execvp failed");
        exit(1);

    } else if (g_serverPid < 0) {
        throw std::runtime_error("Failed to fork server process");
    }

    // Client waits for new server to start
    g_currentServerCapacity = capacity;
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
}

static void cleanupIPCServer() {
    if (g_serverPid > 0) {
        kill(g_serverPid, SIGTERM);
        waitpid(g_serverPid, nullptr, 0);
        g_serverPid = -1;
    }
    shm_unlink(("/cuckoo_filter_" + std::string(SERVER_NAME)).c_str());
}

static void Local_Insert(bm::State& state) {
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

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void IPC_Insert(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    startIPCServerProcess(capacity);

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);

    CuckooFilterIPCClientThrust<Config> client(SERVER_NAME);

    size_t requiredBuckets = std::ceil(static_cast<double>(capacity) / Config::bucketSize);
    size_t numBuckets = 1ULL << static_cast<size_t>(std::ceil(std::log2(requiredBuckets)));

    constexpr size_t tagsPerWord = sizeof(uint64_t) * 8 / Config::bitsPerTag;
    constexpr size_t wordCount = Config::bucketSize / tagsPerWord;
    size_t filterMemory = numBuckets * wordCount * sizeof(uint64_t);

    for (auto _ : state) {
        state.PauseTiming();
        client.clear();
        state.ResumeTiming();

        size_t inserted = client.insertMany(d_keys);
        cudaDeviceSynchronize();
        bm::DoNotOptimize(inserted);
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void Local_Query(bm::State& state) {
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

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void IPC_Query(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    startIPCServerProcess(capacity);

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    thrust::device_vector<uint8_t> d_output(n);

    CuckooFilterIPCClientThrust<Config> client(SERVER_NAME);
    client.clear();
    client.insertMany(d_keys);

    size_t requiredBuckets = std::ceil(static_cast<double>(capacity) / Config::bucketSize);
    size_t numBuckets = 1ULL << static_cast<size_t>(std::ceil(std::log2(requiredBuckets)));

    constexpr size_t tagsPerWord = sizeof(uint64_t) * 8 / Config::bitsPerTag;
    constexpr size_t wordCount = Config::bucketSize / tagsPerWord;
    size_t filterMemory = numBuckets * wordCount * sizeof(uint64_t);

    for (auto _ : state) {
        client.containsMany(d_keys, d_output);
        cudaDeviceSynchronize();
        bm::DoNotOptimize(d_output.data().get());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void Local_Delete(bm::State& state) {
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

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void IPC_Delete(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    startIPCServerProcess(capacity);

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    thrust::device_vector<uint8_t> d_output(n);

    CuckooFilterIPCClientThrust<Config> client(SERVER_NAME);

    size_t requiredBuckets = std::ceil(static_cast<double>(capacity) / Config::bucketSize);
    size_t numBuckets = 1ULL << static_cast<size_t>(std::ceil(std::log2(requiredBuckets)));

    constexpr size_t tagsPerWord = sizeof(uint64_t) * 8 / Config::bitsPerTag;
    constexpr size_t wordCount = Config::bucketSize / tagsPerWord;
    size_t filterMemory = numBuckets * wordCount * sizeof(uint64_t);

    for (auto _ : state) {
        state.PauseTiming();
        client.clear();
        client.insertMany(d_keys);
        state.ResumeTiming();

        size_t remaining = client.deleteMany(d_keys, d_output);
        cudaDeviceSynchronize();
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(d_output.data().get());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void Local_InsertAndQuery(bm::State& state) {
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

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

static void IPC_InsertAndQuery(bm::State& state) {
    auto [capacity, n] = calculateCapacityAndSize<Config>(state.range(0), TARGET_LOAD_FACTOR);

    startIPCServerProcess(capacity);

    thrust::device_vector<uint32_t> d_keys(n);
    generateKeysGPU(d_keys);
    thrust::device_vector<uint8_t> d_output(n);

    CuckooFilterIPCClientThrust<Config> client(SERVER_NAME);

    size_t requiredBuckets = std::ceil(static_cast<double>(capacity) / Config::bucketSize);
    size_t numBuckets = 1ULL << static_cast<size_t>(std::ceil(std::log2(requiredBuckets)));

    constexpr size_t tagsPerWord = sizeof(uint64_t) * 8 / Config::bitsPerTag;
    constexpr size_t wordCount = Config::bucketSize / tagsPerWord;
    size_t filterMemory = numBuckets * wordCount * sizeof(uint64_t);

    for (auto _ : state) {
        state.PauseTiming();
        client.clear();
        state.ResumeTiming();

        size_t inserted = client.insertMany(d_keys);
        client.containsMany(d_keys, d_output);

        cudaDeviceSynchronize();

        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(d_output.data().get());
    }

    state.SetItemsProcessed(static_cast<int64_t>(state.iterations() * n));
    state.counters["memory_bytes"] = bm::Counter(
        static_cast<double>(filterMemory), bm::Counter::kDefaults, bm::Counter::kIs1024
    );
    state.counters["bytes_per_item"] = bm::Counter(
        static_cast<double>(filterMemory) / static_cast<double>(n),
        bm::Counter::kDefaults,
        bm::Counter::kIs1024
    );
}

BENCHMARK(Local_Insert)->RangeMultiplier(2)->Range(1 << 16, 1 << 28)->Unit(bm::kMillisecond);
BENCHMARK(IPC_Insert)->RangeMultiplier(2)->Range(1 << 16, 1 << 28)->Unit(bm::kMillisecond);
BENCHMARK(Local_Query)->RangeMultiplier(2)->Range(1 << 16, 1 << 28)->Unit(bm::kMillisecond);
BENCHMARK(IPC_Query)->RangeMultiplier(2)->Range(1 << 16, 1 << 28)->Unit(bm::kMillisecond);
BENCHMARK(Local_Delete)->RangeMultiplier(2)->Range(1 << 16, 1 << 28)->Unit(bm::kMillisecond);
BENCHMARK(IPC_Delete)->RangeMultiplier(2)->Range(1 << 16, 1 << 28)->Unit(bm::kMillisecond);

BENCHMARK(Local_InsertAndQuery)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

BENCHMARK(IPC_InsertAndQuery)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1 << 28)
    ->Unit(bm::kMillisecond);

int main(int argc, char** argv) {
    bm::Initialize(&argc, argv);
    if (bm::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }
    bm::RunSpecifiedBenchmarks();

    cleanupIPCServer();

    bm::Shutdown();
    return 0;
}
