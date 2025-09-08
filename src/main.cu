#include <cstdint>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <iostream>
#include <random>
#include "common.cuh"
#include "NaiveTable.cuh"

int main() {
    auto table = NaiveTable<uint32_t, 16, 2048>();

    const size_t n = 1000;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> dis(1, UINT32_MAX);

    auto input = std::vector<uint32_t>(n);

    size_t count = 0;

    for (size_t i = 0; i < n; ++i) {
        input[i] = dis(gen);
        count += size_t(table.insert(input[i]));
    }

    std::cout << "Inserted " << count << " / " << n << " elements."
              << std::endl;

    auto mask = table.containsMany(input.data(), n);
    for (size_t i = 0; i < n; ++i) {
        if (!mask[i]) {
            std::cout << "Error: key " << input[i] << " not found!"
                      << std::endl;
        }
    }
}
