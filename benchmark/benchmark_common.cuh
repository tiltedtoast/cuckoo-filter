#pragma once

#include <benchmark/benchmark.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/random.h>
#include <thrust/transform.h>
#include <cstdint>
#include <fstream>
#include <hash_strategies.cuh>
#include <limits>
#include <random>
#include <string>

template <typename ConfigType>
std::pair<size_t, size_t> calculateCapacityAndSize(size_t capacity, double loadFactor) {
    return {capacity, capacity * loadFactor};
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
    size_t n = d_keys.size();
    thrust::transform(
        thrust::counting_iterator<size_t>(0),
        thrust::counting_iterator<size_t>(n),
        d_keys.begin(),
        [max, seed] __device__(size_t idx) {
            thrust::default_random_engine rng(seed);
            thrust::uniform_int_distribution<T> dist(1, max);
            rng.discard(idx);
            return dist(rng);
        }
    );
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
}
