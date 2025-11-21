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

using Config = CuckooConfig<uint64_t, 16, 500, 128, 16, XorAltBucketPolicy>;

static constexpr char SERVER_NAME[] = "benchmark_server";
static pid_t g_serverPid = -1;
static size_t g_currentServerCapacity = 0;

using LocalCFFixture = CuckooFilterFixture<Config>;

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

class IPCCFFixture : public benchmark::Fixture {
    using benchmark::Fixture::SetUp;
    using benchmark::Fixture::TearDown;

   public:
    static constexpr double TARGET_LOAD_FACTOR = 0.95;

    void SetUp(const benchmark::State& state) override {
        auto [cap, num] = calculateCapacityAndSize(state.range(0), TARGET_LOAD_FACTOR);
        capacity = cap;
        n = num;

        startIPCServerProcess(capacity);

        d_keys.resize(n);
        d_output.resize(n);
        generateKeysGPU(d_keys);

        client = std::make_unique<CuckooFilterIPCClientThrust<Config>>(SERVER_NAME);

        size_t requiredBuckets = std::ceil(static_cast<double>(capacity) / Config::bucketSize);
        size_t numBuckets = 1ULL << static_cast<size_t>(std::ceil(std::log2(requiredBuckets)));

        constexpr size_t tagsPerWord = sizeof(uint64_t) * 8 / Config::bitsPerTag;
        constexpr size_t wordCount = Config::bucketSize / tagsPerWord;
        filterMemory = numBuckets * wordCount * sizeof(uint64_t);
    }

    void TearDown(const benchmark::State&) override {
        client.reset();
        d_keys.clear();
        d_output.clear();
    }

    void setCounters(benchmark::State& state) const {
        setCommonCounters(state, filterMemory, n);
    }

    size_t capacity;
    size_t n;
    size_t filterMemory;
    thrust::device_vector<uint64_t> d_keys;
    thrust::device_vector<uint8_t> d_output;
    std::unique_ptr<CuckooFilterIPCClientThrust<Config>> client;
    Timer timer;
};

BENCHMARK_DEFINE_F(LocalCFFixture, Insert)(bm::State& state) {
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

BENCHMARK_DEFINE_F(IPCCFFixture, Insert)(bm::State& state) {
    for (auto _ : state) {
        client->clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = client->insertMany(d_keys);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(LocalCFFixture, Query)(bm::State& state) {
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

BENCHMARK_DEFINE_F(IPCCFFixture, Query)(bm::State& state) {
    client->clear();
    client->insertMany(d_keys);

    for (auto _ : state) {
        timer.start();
        client->containsMany(d_keys, d_output);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(LocalCFFixture, Delete)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        filter->insertMany(d_keys);
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

BENCHMARK_DEFINE_F(IPCCFFixture, Delete)(bm::State& state) {
    for (auto _ : state) {
        client->clear();
        client->insertMany(d_keys);
        cudaDeviceSynchronize();

        timer.start();
        size_t remaining = client->deleteMany(d_keys, d_output);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(remaining);
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(LocalCFFixture, InsertAndQuery)(bm::State& state) {
    for (auto _ : state) {
        filter->clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = filter->insertMany(d_keys);
        filter->containsMany(d_keys, d_output);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

BENCHMARK_DEFINE_F(IPCCFFixture, InsertAndQuery)(bm::State& state) {
    for (auto _ : state) {
        client->clear();
        cudaDeviceSynchronize();

        timer.start();
        size_t inserted = client->insertMany(d_keys);
        client->containsMany(d_keys, d_output);
        double elapsed = timer.stop();

        state.SetIterationTime(elapsed);
        bm::DoNotOptimize(inserted);
        bm::DoNotOptimize(d_output.data().get());
    }
    setCounters(state);
}

REGISTER_BENCHMARK(LocalCFFixture, Insert);
REGISTER_BENCHMARK(IPCCFFixture, Insert);

REGISTER_BENCHMARK(LocalCFFixture, Query);
REGISTER_BENCHMARK(IPCCFFixture, Query);

REGISTER_BENCHMARK(LocalCFFixture, Delete);
REGISTER_BENCHMARK(IPCCFFixture, Delete);

REGISTER_BENCHMARK(LocalCFFixture, InsertAndQuery);
REGISTER_BENCHMARK(IPCCFFixture, InsertAndQuery);

int main(int argc, char** argv) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        return 1;
    }

    ::benchmark::RunSpecifiedBenchmarks();
    cleanupIPCServer();
    ::benchmark::Shutdown();

    fflush(stdout);
    std::cout << std::flush;

    std::_Exit(0);
}
