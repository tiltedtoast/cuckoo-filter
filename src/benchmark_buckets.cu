#include <chrono>
#include <cstdint>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <vector>
#include "BucketsTableCpu.cuh"
#include "BucketsTableGpu.cuh"
#include "common.cuh"

template <typename T>
size_t countOnes(T* data, size_t n) {
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) {
        if (data[i]) {
            count++;
        }
    }
    return count;
}

struct BucketSizeBenchmarkResult {
    int exponent{};
    size_t n{};
    size_t bucketSize{};
    std::string tableType;
    double avgsInsertTimeMs{};
    double avgQueryTimeMs{};
    double avgTotalTimeMs{};
    double minTotalTimeMs{};
    double maxTotalTimeMs{};
    size_t itemsInserted{};
    size_t itemsFound{};
    double loadFactor{};
    double falsePositiveRate{};
    size_t memoryUsageBytes{};
    double insertThroughputMops{};
    double queryThroughputMops{};
};

template <typename Func>
std::vector<double> benchmarkFunction(Func func, int iterations = 5) {
    std::vector<double> times(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        double timeMs = static_cast<double>(duration.count()) / 1000.0 / 1000.0;
        times[i] = timeMs;
    }

    return times;
}

template <size_t bucketSize>
BucketSizeBenchmarkResult
benchmarkCpuTableWithBucketSize(uint32_t* input, size_t n, int exponent) {
    BucketSizeBenchmarkResult result;
    result.exponent = exponent;
    result.n = n;
    result.bucketSize = bucketSize;
    result.tableType = "BucketsTableCpu";

    size_t count = 0;
    size_t found = 0;
    double insertTime = 0.0;
    double queryTime = 0.0;

    auto benchmarkFunc = [&]() {
        auto table =
            BucketsTableCpu<uint32_t, 32, bucketSize, 1000>(n / bucketSize);

        auto insert_start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < n; ++i) {
            count += size_t(table.insert(input[i]));
        }
        auto insert_end = std::chrono::high_resolution_clock::now();
        insertTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         insert_end - insert_start
                     )
                         .count() /
                     1000.0 / 1000.0;

        auto query_start = std::chrono::high_resolution_clock::now();
        bool* output = table.containsMany(input, n);
        auto query_end = std::chrono::high_resolution_clock::now();
        queryTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        query_end - query_start
                    )
                        .count() /
                    1000.0 / 1000.0;

        found = countOnes(output, n);
        std::free(output);

        result.loadFactor = table.loadFactor();
        result.falsePositiveRate = table.expectedFalsePositiveRate();
        size_t numBuckets = n / bucketSize;
        result.memoryUsageBytes = numBuckets * (bucketSize * sizeof(uint32_t) +
                                                sizeof(int) + sizeof(size_t));
    };

    auto times = benchmarkFunction(benchmarkFunc);

    result.itemsInserted = count;
    result.itemsFound = found;
    result.avgsInsertTimeMs = insertTime;
    result.avgQueryTimeMs = queryTime;
    result.minTotalTimeMs = *std::min_element(times.begin(), times.end());
    result.maxTotalTimeMs = *std::max_element(times.begin(), times.end());

    double sum = 0.0;
    for (double time : times) {
        sum += time;
    }
    result.avgTotalTimeMs = sum / times.size();

    result.insertThroughputMops =
        (static_cast<double>(n) / 1'000'000.0) / (insertTime / 1'000.0);
    result.queryThroughputMops =
        (static_cast<double>(n) / 1'000'000.0) / (queryTime / 1'000.0);

    return result;
}

template <size_t bucketSize>
BucketSizeBenchmarkResult
benchmarkGpuTableWithBucketSize(uint32_t* input, size_t n, int exponent) {
    BucketSizeBenchmarkResult result;
    result.exponent = exponent;
    result.n = n;
    result.bucketSize = bucketSize;
    result.tableType = "BucketsTableGpu";

    size_t count = 0;
    size_t found = 0;
    bool* output = nullptr;
    CUDA_CALL(cudaMallocHost(&output, sizeof(bool) * n));

    double insertTime = 0.0;
    double queryTime = 0.0;

    auto benchmarkFunc = [&]() {
        auto table =
            BucketsTableGpu<uint32_t, 32, bucketSize, 1000>(n / bucketSize);

        auto insert_start = std::chrono::high_resolution_clock::now();
        count = table.insertMany(input, n);
        auto insert_end = std::chrono::high_resolution_clock::now();

        insertTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         insert_end - insert_start
                     )
                         .count() /
                     1000.0 / 1000.0;

        auto query_start = std::chrono::high_resolution_clock::now();
        table.containsMany(input, n, output);
        auto query_end = std::chrono::high_resolution_clock::now();

        queryTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        query_end - query_start
                    )
                        .count() /
                    1000.0 / 1000.0;

        found = countOnes(output, n);
        result.loadFactor = table.loadFactor();
        result.falsePositiveRate = table.expectedFalsePositiveRate();

        size_t numBuckets = n / bucketSize;
        result.memoryUsageBytes = numBuckets * (bucketSize * sizeof(uint32_t) +
                                                sizeof(int) + sizeof(size_t));
    };

    auto times = benchmarkFunction(benchmarkFunc);

    result.itemsInserted = count;
    result.itemsFound = found;
    result.avgsInsertTimeMs = insertTime;
    result.avgQueryTimeMs = queryTime;
    result.minTotalTimeMs = *std::min_element(times.begin(), times.end());
    result.maxTotalTimeMs = *std::max_element(times.begin(), times.end());

    double sum = 0.0;
    for (double time : times) {
        sum += time;
    }
    result.avgTotalTimeMs = sum / times.size();

    result.insertThroughputMops =
        (static_cast<double>(n) / 1000000.0) / (insertTime / 1000.0);
    result.queryThroughputMops =
        (static_cast<double>(n) / 1000000.0) / (queryTime / 1000.0);

    CUDA_CALL(cudaFreeHost(output));
    return result;
}

template <typename BenchmarkFunc>
void runBucketSizeBenchmarks(
    BenchmarkFunc benchmarkFunc,
    uint32_t* input,
    size_t n,
    int exponent,
    std::vector<BucketSizeBenchmarkResult>& results
) {
    results.push_back(benchmarkFunc.template operator()<4>(input, n, exponent));
    results.push_back(benchmarkFunc.template operator()<8>(input, n, exponent));
    results.push_back(
        benchmarkFunc.template operator()<16>(input, n, exponent)
    );
    results.push_back(
        benchmarkFunc.template operator()<32>(input, n, exponent)
    );
    results.push_back(
        benchmarkFunc.template operator()<64>(input, n, exponent)
    );
    results.push_back(
        benchmarkFunc.template operator()<128>(input, n, exponent)
    );
    results.push_back(
        benchmarkFunc.template operator()<256>(input, n, exponent)
    );
}

void writeResultsToCsv(
    const std::vector<BucketSizeBenchmarkResult>& results,
    const std::string& filename
) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing"
                  << std::endl;
        return;
    }

    file << "exponent,n,bucketSize,tableType,avgsInsertTimeMs,avgQueryTimeMs,"
         << "avgTotalTimeMs,minTotalTimeMs,maxTotalTimeMs,itemsInserted,"
         << "itemsFound,loadFactor,falsePositiveRate,memoryUsageMB,"
         << "insertThroughputMops,queryThroughputMops\n";

    for (const auto& result : results) {
        file << result.exponent << "," << result.n << "," << result.bucketSize
             << "," << result.tableType << "," << std::fixed
             << std::setprecision(6) << result.avgsInsertTimeMs << ","
             << std::fixed << std::setprecision(6) << result.avgQueryTimeMs
             << "," << std::fixed << std::setprecision(6)
             << result.avgTotalTimeMs << "," << std::fixed
             << std::setprecision(6) << result.minTotalTimeMs << ","
             << std::fixed << std::setprecision(6) << result.maxTotalTimeMs
             << "," << result.itemsInserted << "," << result.itemsFound << ","
             << std::fixed << std::setprecision(4) << result.loadFactor << ","
             << std::fixed << std::setprecision(8) << result.falsePositiveRate
             << "," << std::fixed << std::setprecision(2)
             << (result.memoryUsageBytes / (1024.0 * 1024.0)) << ","
             << std::fixed << std::setprecision(2)
             << result.insertThroughputMops << "," << std::fixed
             << std::setprecision(2) << result.queryThroughputMops << "\n";
    }

    file.close();
    std::cout << "Results written to " << filename << std::endl;
}

void printSummaryTable(const std::vector<BucketSizeBenchmarkResult>& results) {
    std::cout << "\n" << std::string(120, '=') << std::endl;
    std::cout << "BUCKET SIZE ANALYSIS SUMMARY" << std::endl;
    std::cout << std::string(120, '=') << std::endl;

    std::map<
        std::string,
        std::map<size_t, std::vector<BucketSizeBenchmarkResult>>>
        groupedResults;

    for (const auto& result : results) {
        groupedResults[result.tableType][result.n].push_back(result);
    }

    for (const auto& [tableType, sizeMap] : groupedResults) {
        std::cout << "\n" << tableType << " Performance Analysis:" << std::endl;
        std::cout << std::string(60, '-') << std::endl;

        std::cout << std::left << std::setw(12) << "Data Size" << std::setw(12)
                  << "Bucket Size" << std::setw(15) << "Insert (MOPS)"
                  << std::setw(15) << "Query (MOPS)" << std::setw(12)
                  << "Load Factor" << std::setw(12) << "Memory (MB)"
                  << std::endl;
        std::cout << std::string(78, '-') << std::endl;

        for (const auto& [data_size, results_vec] : sizeMap) {
            for (const auto& result : results_vec) {
                std::cout << std::left << std::setw(12)
                          << ("2^" + std::to_string(result.exponent))
                          << std::setw(12) << result.bucketSize << std::setw(15)
                          << std::fixed << std::setprecision(2)
                          << result.insertThroughputMops << std::setw(15)
                          << std::fixed << std::setprecision(2)
                          << result.queryThroughputMops << std::setw(12)
                          << std::fixed << std::setprecision(3)
                          << result.loadFactor << std::setw(12) << std::fixed
                          << std::setprecision(1)
                          << (result.memoryUsageBytes / (1024.0 * 1024.0))
                          << std::endl;
            }

            auto best_insert = std::max_element(
                results_vec.begin(),
                results_vec.end(),
                [](const auto& a, const auto& b) {
                    return a.insertThroughputMops < b.insertThroughputMops;
                }
            );
            auto best_query = std::max_element(
                results_vec.begin(),
                results_vec.end(),
                [](const auto& a, const auto& b) {
                    return a.queryThroughputMops < b.queryThroughputMops;
                }
            );

            std::cout << "    → Best insert performance: bucket size "
                      << best_insert->bucketSize << " (" << std::fixed
                      << std::setprecision(2)
                      << best_insert->insertThroughputMops << " MOPS)"
                      << std::endl;
            std::cout << "    → Best query performance: bucket size "
                      << best_query->bucketSize << " (" << std::fixed
                      << std::setprecision(2) << best_query->queryThroughputMops
                      << " MOPS)" << std::endl;
            std::cout << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    std::string output_file = "benchmark_results_bucketSize.csv";

    if (argc > 1) {
        output_file = argv[1];
    }

    const int min_exponent = 10;
    const int max_exponent = 30;

    std::cout << "Generating test data..." << std::endl;

    std::mt19937 rng(42);
    std::uniform_int_distribution<uint32_t> dist(1, UINT32_MAX);

    size_t max_n = 1ULL << max_exponent;
    uint32_t* input = nullptr;
    CUDA_CALL(cudaMallocHost(&input, sizeof(uint32_t) * max_n));

    for (size_t i = 0; i < max_n; ++i) {
        input[i] = dist(rng);
    }

    std::vector<BucketSizeBenchmarkResult> results;

    std::cout << "\nRunning bucket size benchmarks..." << std::endl;
    std::cout << "Testing bucket sizes: 4, 8, 16, 32, 64, 128, 256"
              << std::endl;
    std::cout << std::string(70, '-') << std::endl;

    for (int exponent = min_exponent; exponent <= max_exponent; ++exponent) {
        size_t n = 1ULL << exponent;

        std::cout << "\nTesting size 2^" << exponent << " (" << n
                  << " elements):" << std::endl;

        size_t starting_index = results.size();

        // GPU Benchmarks
        results.push_back(
            benchmarkGpuTableWithBucketSize<4>(input, n, exponent)
        );
        results.push_back(
            benchmarkGpuTableWithBucketSize<8>(input, n, exponent)
        );
        results.push_back(
            benchmarkGpuTableWithBucketSize<16>(input, n, exponent)
        );
        results.push_back(
            benchmarkGpuTableWithBucketSize<32>(input, n, exponent)
        );
        results.push_back(
            benchmarkGpuTableWithBucketSize<64>(input, n, exponent)
        );
        results.push_back(
            benchmarkGpuTableWithBucketSize<128>(input, n, exponent)
        );
        results.push_back(
            benchmarkGpuTableWithBucketSize<256>(input, n, exponent)
        );

        // CPU Benchmarks
        results.push_back(
            benchmarkCpuTableWithBucketSize<4>(input, n, exponent)
        );
        results.push_back(
            benchmarkCpuTableWithBucketSize<8>(input, n, exponent)
        );
        results.push_back(
            benchmarkCpuTableWithBucketSize<16>(input, n, exponent)
        );
        results.push_back(
            benchmarkCpuTableWithBucketSize<32>(input, n, exponent)
        );
        results.push_back(
            benchmarkCpuTableWithBucketSize<64>(input, n, exponent)
        );
        results.push_back(
            benchmarkCpuTableWithBucketSize<128>(input, n, exponent)
        );
        results.push_back(
            benchmarkCpuTableWithBucketSize<256>(input, n, exponent)
        );

        for (size_t i = starting_index; i < results.size(); ++i) {
            const auto& result = results[i];
            std::cout << "    " << std::setw(15) << std::left
                      << result.tableType << " Bucket " << std::setw(3)
                      << std::right << result.bucketSize << ": " << std::setw(7)
                      << std::fixed << std::setprecision(2)
                      << result.insertThroughputMops << " MOPS (insert), "
                      << std::setw(7) << result.queryThroughputMops
                      << " MOPS (query)" << std::endl;

            if (i == starting_index + 6) {
                std::cout << std::endl;
            }
        }
    }

    writeResultsToCsv(results, output_file);
    printSummaryTable(results);

    CUDA_CALL(cudaFreeHost(input));

    std::cout << "\nBucket size benchmark completed successfully!" << std::endl;

    return 0;
}