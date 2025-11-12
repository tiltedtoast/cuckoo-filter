#pragma once

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/partition.h>
#include <cstddef>
#include <cstdint>
#include <CuckooFilter.cuh>
#include <memory>
#include <vector>
#include "helpers.cuh"

template <typename Config>
class CuckooFilterMultiGPU {
   public:
    using KeyType = typename Config::KeyType;
    static constexpr size_t STREAMS_PER_GPU = 4;
    static constexpr size_t CHUNK_SIZE = 1 << 20;

    struct Partitioner {
        size_t numGPUs;

        __host__ __device__ size_t operator()(const KeyType& key) const {
            uint64_t hash = CuckooFilter<Config>::hash64(key);
            return hash % numGPUs;
        }
    };

   private:
    size_t numGPUs;
    size_t capacityPerGPU;
    std::vector<CuckooFilter<Config>*> filters;
    std::vector<cudaStream_t> streams;

   public:
    CuckooFilterMultiGPU(size_t numGPUs, size_t capacity)
        : numGPUs(numGPUs), capacityPerGPU(static_cast<size_t>(SDIV(capacity, numGPUs) * 1.05)) {
        assert(numGPUs > 0 && "Number of GPUs must be at least 1");

        int deviceCount;
        CUDA_CALL(cudaGetDeviceCount(&deviceCount));
        assert(
            numGPUs <= static_cast<size_t>(deviceCount) && "Requested more GPUs than available"
        );

        filters.reserve(numGPUs);
        streams.reserve(numGPUs);

        for (size_t i = 0; i < numGPUs; ++i) {
            cudaSetDevice(i);
            filters[i] = new CuckooFilter<Config>(capacityPerGPU);

            CUDA_CALL(cudaStreamCreate(&streams[i]));
        }

        cudaSetDevice(0);
    }

    ~CuckooFilterMultiGPU() {
        for (size_t i = 0; i < numGPUs; ++i) {
            cudaSetDevice(i);
            cudaStreamDestroy(streams[i]);
            delete filters[i];
        }

        cudaSetDevice(0);
    }

    CuckooFilterMultiGPU(const CuckooFilterMultiGPU&) = delete;
    CuckooFilterMultiGPU& operator=(const CuckooFilterMultiGPU&) = delete;

    size_t insertMany(const thrust::host_vector<KeyType>& h_keys) {
        size_t n = h_keys.size();

        // Partition keys on host by computing which GPU each belongs to
        Partitioner partitioner{numGPUs};
        std::vector<std::vector<KeyType>> gpuPartitions(numGPUs);

        for (size_t i = 0; i < numGPUs; ++i) {
            gpuPartitions[i].reserve(capacityPerGPU);
        }

        for (const auto& key : h_keys) {
            size_t gpuId = partitioner(key);
            gpuPartitions[gpuId].push_back(key);
        }

        std::vector<thrust::device_vector<KeyType>> d_gpuKeys(numGPUs);

        for (size_t gpuId = 0; gpuId < numGPUs; ++gpuId) {
            if (gpuPartitions[gpuId].empty()) {
                continue;
            }

            CUDA_CALL(cudaSetDevice(gpuId));

            d_gpuKeys[gpuId] = thrust::device_vector<KeyType>(
                gpuPartitions[gpuId].begin(), gpuPartitions[gpuId].end()
            );

            filters[gpuId]->insertMany(d_gpuKeys[gpuId], streams[gpuId]);
        }

        CUDA_CALL(cudaSetDevice(0));

        return totalOccupiedSlots();
    }

    void
    containsMany(const thrust::host_vector<KeyType>& h_keys, thrust::host_vector<bool>& h_output) {
        if (h_output.size() != h_keys.size()) {
            h_output.resize(h_keys.size());
        }

        size_t n = h_keys.size();

        Partitioner partitioner{numGPUs};
        std::vector<std::vector<std::pair<size_t, KeyType>>> gpuPartitions(numGPUs);

        for (size_t i = 0; i < numGPUs; ++i) {
            gpuPartitions[i].reserve(capacityPerGPU);
        }

        for (size_t i = 0; i < n; ++i) {
            size_t gpuId = partitioner(h_keys[i]);
            gpuPartitions[gpuId].push_back({i, h_keys[i]});
        }

        std::vector<thrust::device_vector<KeyType>> d_gpuKeys(numGPUs);
        std::vector<thrust::device_vector<bool>> d_gpuResults(numGPUs);

        for (size_t gpuId = 0; gpuId < numGPUs; ++gpuId) {
            if (gpuPartitions[gpuId].empty()) {
                continue;
            }

            CUDA_CALL(cudaSetDevice(gpuId));

            std::vector<KeyType> gpuKeys;
            gpuKeys.reserve(gpuPartitions[gpuId].size());
            for (const auto& [idx, key] : gpuPartitions[gpuId]) {
                gpuKeys.push_back(key);
            }

            d_gpuKeys[gpuId] = thrust::device_vector<KeyType>(gpuKeys.begin(), gpuKeys.end());
            filters[gpuId]->containsMany(d_gpuKeys[gpuId], d_gpuResults[gpuId], streams[gpuId]);
        }

        for (size_t gpuId = 0; gpuId < numGPUs; ++gpuId) {
            if (gpuPartitions[gpuId].empty()) {
                continue;
            }

            CUDA_CALL(cudaSetDevice(gpuId));

            thrust::host_vector<bool> h_gpuResults = d_gpuResults[gpuId];

            // Scatter results back to original positions
            for (size_t i = 0; i < gpuPartitions[gpuId].size(); ++i) {
                h_output[gpuPartitions[gpuId][i].first] = h_gpuResults[i];
            }
        }

        CUDA_CALL(cudaSetDevice(0));
    }

    size_t
    deleteMany(const thrust::host_vector<KeyType>& h_keys, thrust::host_vector<bool>& h_output) {
        if (h_output.size() != h_keys.size()) {
            h_output.resize(h_keys.size());
        }

        size_t n = h_keys.size();

        Partitioner partitioner{numGPUs};
        std::vector<std::vector<std::pair<size_t, KeyType>>> gpuPartitions(numGPUs);

        for (size_t i = 0; i < numGPUs; ++i) {
            gpuPartitions[i].reserve(capacityPerGPU);
        }

        for (size_t i = 0; i < n; ++i) {
            size_t gpuId = partitioner(h_keys[i]);
            gpuPartitions[gpuId].push_back({i, h_keys[i]});
        }

        // Store device vectors to keep them alive until synchronization
        std::vector<thrust::device_vector<KeyType>> d_gpuKeys(numGPUs);
        std::vector<thrust::device_vector<bool>> d_gpuResults(numGPUs);

        for (size_t gpuId = 0; gpuId < numGPUs; ++gpuId) {
            if (gpuPartitions[gpuId].empty()) {
                continue;
            }

            CUDA_CALL(cudaSetDevice(gpuId));

            std::vector<KeyType> gpuKeys;
            gpuKeys.reserve(gpuPartitions[gpuId].size());
            for (const auto& [idx, key] : gpuPartitions[gpuId]) {
                gpuKeys.push_back(key);
            }

            d_gpuKeys[gpuId] = thrust::device_vector<KeyType>(gpuKeys.begin(), gpuKeys.end());
            filters[gpuId]->deleteMany(d_gpuKeys[gpuId], d_gpuResults[gpuId], streams[gpuId]);
        }

        // Copy results back to host (stream sync happens internally)
        for (size_t gpuId = 0; gpuId < numGPUs; ++gpuId) {
            if (gpuPartitions[gpuId].empty()) {
                continue;
            }

            CUDA_CALL(cudaSetDevice(gpuId));

            thrust::host_vector<bool> h_gpuResults = d_gpuResults[gpuId];

            // Scatter results back to original positions
            for (size_t i = 0; i < gpuPartitions[gpuId].size(); ++i) {
                h_output[gpuPartitions[gpuId][i].first] = h_gpuResults[i];
            }
        }

        CUDA_CALL(cudaSetDevice(0));

        return totalOccupiedSlots();
    }

    float loadFactor() {
        size_t totalOccupied = totalOccupiedSlots();
        size_t totalCapacity = 0;
        for (size_t i = 0; i < numGPUs; ++i) {
            totalCapacity += filters[i]->capacity();
        }
        return static_cast<float>(totalOccupied) / static_cast<float>(totalCapacity);
    }

    size_t totalOccupiedSlots() {
        size_t total = 0;
        for (size_t i = 0; i < numGPUs; ++i) {
            total += filters[i]->occupiedSlots();
        }
        return total;
    }

    void clear() {
        for (size_t i = 0; i < numGPUs; ++i) {
            CUDA_CALL(cudaSetDevice(i));
            filters[i]->clear();
        }
        CUDA_CALL(cudaSetDevice(0));
    }

    [[nodiscard]] size_t totalCapacity() const {
        size_t total = 0;
        for (size_t i = 0; i < numGPUs; ++i) {
            total += filters[i]->capacity();
        }
        return total;
    }
};