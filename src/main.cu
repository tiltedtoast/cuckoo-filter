#include <chrono>
#include <cstdint>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <iostream>
#include <random>
#include "BucketsTableCpu.cuh"
#include "BucketsTableGpu.cuh"
#include "common.cuh"
#include "NaiveTable.cuh"

template <typename T>
size_t count_ones(T* data, size_t n) {
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (data[i]) {
            count++;
        }
    }
    return count;
}

int main() {
    const size_t n = 1 << 25;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(1, UINT32_MAX);

    auto input = std::vector<uint32_t>(n);
    for (size_t i = 0; i < n; ++i) {
        input[i] = dis(gen);
    }

    auto naive_table = NaiveTable<uint32_t, 32, n * 2>();
    auto start = std::chrono::high_resolution_clock::now();

    size_t naive_count = 0;
    for (size_t i = 0; i < n; ++i) {
        naive_count += size_t(naive_table.insert(input[i]));
    }

    auto naive_mask = naive_table.containsMany(input.data(), n);
    auto end = std::chrono::high_resolution_clock::now();
    auto naive_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    auto buckets_table = BucketsTableCpu<uint32_t, 32, 32, n / 32>();
    auto buckets_table_gpu = BucketsTableGpu<uint32_t, 32, 32, n / 32>();

    start = std::chrono::high_resolution_clock::now();

    size_t buckets_count = 0;
    for (size_t i = 0; i < n; ++i) {
        buckets_count += size_t(buckets_table.insert(input[i]));
    }

    auto buckets_mask = buckets_table.containsMany(input.data(), n);
    end = std::chrono::high_resolution_clock::now();
    auto buckets_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    start = std::chrono::high_resolution_clock::now();

    size_t buckets_gpu_counter = buckets_table_gpu.insertMany(input.data(), n);
    auto buckets_gpu_mask = buckets_table_gpu.containsMany(input.data(), n);

    end = std::chrono::high_resolution_clock::now();
    auto buckets_gpu_duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "NaiveTable:\t\tInserted & Queried " << naive_count << " / "
              << n << " elements in " << naive_duration.count() << " ms"
              << "\t(" << count_ones(naive_mask, n) << " found)" << std::endl;
    std::cout << "BucketsTableCpu:\tInserted & Queried " << buckets_count
              << " / " << n << " elements in " << buckets_duration.count()
              << " ms" << "\t(" << count_ones(buckets_mask, n) << " found)"
              << std::endl;
    std::cout << "BucketsTableGpu:\tInserted & Queried " << buckets_gpu_counter
              << " / " << n << " elements in " << buckets_gpu_duration.count()
              << " ms" << "\t(" << count_ones(buckets_gpu_mask, n) << " found)"
              << std::endl;
}
