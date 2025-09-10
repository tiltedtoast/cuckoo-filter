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

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <table_type> <n_exponent>"
                  << std::endl;
        std::cerr
            << "table_type: 0=NaiveTable, 1=BucketsTableCpu, 2=BucketsTableGpu"
            << std::endl;
        std::cerr << "n_exponent: exponent for n = 2^x" << std::endl;
        return 1;
    }

    int table_type = std::atoi(argv[1]);
    int n_exponent = std::atoi(argv[2]);

    if (table_type < 0 || table_type > 2) {
        std::cerr << "Invalid table type. Use 0, 1, or 2." << std::endl;
        return 1;
    }

    if (n_exponent < 1 || n_exponent > 30) {
        std::cerr << "Invalid exponent. Use 1-30." << std::endl;
        return 1;
    }

    size_t n = 1ULL << n_exponent;

    uint32_t* input;
    std::mt19937 rng;
    std::uniform_int_distribution<uint32_t> dist(1, UINT32_MAX);

    CUDA_CALL(cudaMallocHost(&input, sizeof(uint32_t) * n));

    for (size_t i = 0; i < n; ++i) {
        input[i] = dist(rng);
    }

    if (table_type == 0) {
        auto table = NaiveTable<uint32_t, 32, 1000, 256>(n * 2);

        auto start = std::chrono::high_resolution_clock::now();
        size_t count = 0;
        for (size_t i = 0; i < n; ++i) {
            count += size_t(table.insert(input[i]));
        }

        auto mask = table.containsMany(input, n);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();

        size_t found = count_ones(mask, n);
        std::cout << "NaiveTable: Inserted " << n << " items, found " << found
                  << " items in " << duration << " ms" << std::endl;
    } else if (table_type == 1) {
        auto table = BucketsTableCpu<uint32_t, 32, 32, 1000>(n / 16);

        auto start = std::chrono::high_resolution_clock::now();

        for (size_t i = 0; i < n; ++i) {
            table.insert(input[i]);
        }
        auto mask = table.containsMany(input, n);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();

        size_t found = count_ones(mask, n);
        std::cout << "BucketsTableCpu: Inserted " << n << " items, found "
                  << found << " items in " << duration << " ms" << std::endl;
    } else if (table_type == 2) {
        auto table = BucketsTableCpu<uint32_t, 32, 32, 1000>(n / 16);
    }
}
