#pragma once

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include "CuckooFilter.cuh"
#include "helpers.cuh"

template <typename Config>
class CuckooFilterMultiGPU {
   public:
    using T = typename Config::KeyType;
    static constexpr size_t CHUNK_SIZE = 1 << 22;

    struct Partitioner {
        size_t numGPUs;

        __host__ __device__ size_t operator()(const T& key) const {
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

        for (size_t i = 0; i < numGPUs; ++i) {
            CUDA_CALL(cudaSetDevice(i));
            for (size_t j = 0; j < numGPUs; ++j) {
                if (i != j) {
                    int canAccessPeer = 0;
                    cudaDeviceCanAccessPeer(&canAccessPeer, i, j);
                    if (canAccessPeer) {
                        cudaDeviceEnablePeerAccess(j, 0);
                    } else {
                        std::cerr << "Warning: P2P access not supported between GPU " << i
                                  << " and GPU " << j << std::endl;
                    }
                }
            }
        }

        filters.reserve(numGPUs);
        streams.reserve(numGPUs);

        for (size_t i = 0; i < numGPUs; ++i) {
            cudaSetDevice(i);
            filters.push_back(new CuckooFilter<Config>(capacityPerGPU));

            cudaStream_t newStream;
            CUDA_CALL(cudaStreamCreate(&newStream));
            streams.push_back(newStream);
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

    size_t insertMany(const thrust::host_vector<T>& h_keys) {
        size_t n = h_keys.size();
        if (n == 0) {
            return totalOccupiedSlots();
        }

        std::vector<thrust::device_vector<T>> d_stagingVectors(numGPUs);

        for (size_t chunkOffset = 0; chunkOffset < n; chunkOffset += CHUNK_SIZE) {
            size_t chunkSize = std::min(CHUNK_SIZE, n - chunkOffset);

            CUDA_CALL(cudaSetDevice(0));
            thrust::device_vector<T> d_chunkKeys(
                h_keys.begin() + chunkOffset, h_keys.begin() + chunkOffset + chunkSize
            );
            thrust::device_vector<size_t> d_gpuIndices(chunkSize);

            Partitioner partitioner{numGPUs};
            thrust::transform(
                thrust::device,
                d_chunkKeys.begin(),
                d_chunkKeys.end(),
                d_gpuIndices.begin(),
                partitioner
            );
            thrust::sort_by_key(
                thrust::device, d_gpuIndices.begin(), d_gpuIndices.end(), d_chunkKeys.begin()
            );

            thrust::device_vector<size_t> d_counts(numGPUs);
            thrust::device_vector<size_t> d_offsets(numGPUs);
            thrust::device_vector<size_t> d_uniqueKeys(numGPUs);

            thrust::reduce_by_key(
                thrust::device,
                d_gpuIndices.begin(),
                d_gpuIndices.end(),
                thrust::make_constant_iterator<size_t>(1),
                d_uniqueKeys.begin(),
                d_counts.begin()
            );

            thrust::exclusive_scan(
                thrust::device, d_counts.begin(), d_counts.end(), d_offsets.begin(), 0
            );

            thrust::host_vector<size_t> h_counts = d_counts;
            thrust::host_vector<size_t> h_offsets = d_offsets;

            for (size_t gpuId = 0; gpuId < numGPUs; ++gpuId) {
                if (h_counts[gpuId] > 0) {
                    CUDA_CALL(cudaSetDevice(gpuId));
                    size_t oldSize = d_stagingVectors[gpuId].size();
                    d_stagingVectors[gpuId].resize(oldSize + h_counts[gpuId]);
                    CUDA_CALL(cudaMemcpyPeerAsync(
                        d_stagingVectors[gpuId].data().get() + oldSize,
                        gpuId,
                        d_chunkKeys.data().get() + h_offsets[gpuId],
                        0,
                        h_counts[gpuId] * sizeof(T),
                        streams[gpuId]
                    ));
                }
            }
            for (size_t i = 0; i < numGPUs; ++i) {
                CUDA_CALL(cudaSetDevice(i));
                CUDA_CALL(cudaStreamSynchronize(streams[i]));
            }
        }

        for (size_t gpuId = 0; gpuId < numGPUs; ++gpuId) {
            CUDA_CALL(cudaSetDevice(gpuId));
            if (!d_stagingVectors[gpuId].empty()) {
                filters[gpuId]->insertMany(d_stagingVectors[gpuId], streams[gpuId]);
            }
        }

        for (size_t gpuId = 0; gpuId < numGPUs; ++gpuId) {
            CUDA_CALL(cudaSetDevice(gpuId));
            CUDA_CALL(cudaStreamSynchronize(streams[gpuId]));
        }

        CUDA_CALL(cudaSetDevice(0));
        return totalOccupiedSlots();
    }

    void containsMany(const thrust::host_vector<T>& h_keys, thrust::host_vector<bool>& h_output) {
        size_t n = h_keys.size();
        if (h_output.size() != n) {
            h_output.resize(n);
        }

        if (n == 0) {
            return;
        }

        for (size_t chunkOffset = 0; chunkOffset < n; chunkOffset += CHUNK_SIZE) {
            size_t chunkSize = std::min(CHUNK_SIZE, n - chunkOffset);

            CUDA_CALL(cudaSetDevice(0));
            thrust::device_vector<T> d_chunkKeys(
                h_keys.begin() + chunkOffset, h_keys.begin() + chunkOffset + chunkSize
            );
            thrust::device_vector<size_t> d_originalIndices(chunkSize);
            thrust::sequence(thrust::device, d_originalIndices.begin(), d_originalIndices.end());

            thrust::device_vector<size_t> d_gpuIndices(chunkSize);
            Partitioner partitioner{numGPUs};
            thrust::transform(
                thrust::device,
                d_chunkKeys.begin(),
                d_chunkKeys.end(),
                d_gpuIndices.begin(),
                partitioner
            );
            thrust::sort_by_key(
                thrust::device,
                d_gpuIndices.begin(),
                d_gpuIndices.end(),
                thrust::make_zip_iterator(
                    thrust::make_tuple(d_chunkKeys.begin(), d_originalIndices.begin())
                )
            );

            thrust::device_vector<size_t> d_counts(numGPUs);
            thrust::device_vector<size_t> d_offsets(numGPUs);
            thrust::device_vector<size_t> d_uniqueKeys(numGPUs);

            thrust::reduce_by_key(
                thrust::device,
                d_gpuIndices.begin(),
                d_gpuIndices.end(),
                thrust::make_constant_iterator<size_t>(1),
                d_uniqueKeys.begin(),
                d_counts.begin()
            );

            thrust::exclusive_scan(
                thrust::device, d_counts.begin(), d_counts.end(), d_offsets.begin(), 0
            );

            thrust::host_vector<size_t> h_counts = d_counts;
            thrust::host_vector<size_t> h_offsets = d_offsets;

            thrust::device_vector<bool> d_chunkResultsSorted(chunkSize);
            for (size_t gpuId = 0; gpuId < numGPUs; ++gpuId) {
                if (h_counts[gpuId] > 0) {
                    CUDA_CALL(cudaSetDevice(gpuId));
                    thrust::device_vector<T> d_receivedKeys(h_counts[gpuId]);
                    thrust::device_vector<bool> d_gpuResults(h_counts[gpuId]);

                    CUDA_CALL(cudaMemcpyPeerAsync(
                        d_receivedKeys.data().get(),
                        gpuId,
                        d_chunkKeys.data().get() + h_offsets[gpuId],
                        0,
                        h_counts[gpuId] * sizeof(T),
                        streams[gpuId]
                    ));
                    filters[gpuId]->containsMany(d_receivedKeys, d_gpuResults, streams[gpuId]);
                    CUDA_CALL(cudaMemcpyPeerAsync(
                        d_chunkResultsSorted.data().get() + h_offsets[gpuId],
                        0,
                        d_gpuResults.data().get(),
                        gpuId,
                        h_counts[gpuId] * sizeof(bool),
                        streams[gpuId]
                    ));
                }
            }

            for (size_t gpuId = 0; gpuId < numGPUs; ++gpuId) {
                CUDA_CALL(cudaSetDevice(gpuId));
                CUDA_CALL(cudaStreamSynchronize(streams[gpuId]));
            }

            CUDA_CALL(cudaSetDevice(0));
            thrust::device_vector<bool> d_chunkOutput(chunkSize);
            thrust::scatter(
                thrust::device,
                d_chunkResultsSorted.begin(),
                d_chunkResultsSorted.end(),
                d_originalIndices.begin(),
                d_chunkOutput.begin()
            );
            thrust::copy(
                d_chunkOutput.begin(), d_chunkOutput.end(), h_output.begin() + chunkOffset
            );
        }
        CUDA_CALL(cudaSetDevice(0));
    }

    size_t deleteMany(const thrust::host_vector<T>& h_keys, thrust::host_vector<bool>& h_output) {
        size_t n = h_keys.size();
        if (h_output.size() != n) {
            h_output.resize(n);
        }

        if (n == 0) {
            return totalOccupiedSlots();
        }

        for (size_t chunkOffset = 0; chunkOffset < n; chunkOffset += CHUNK_SIZE) {
            size_t chunkSize = std::min(CHUNK_SIZE, n - chunkOffset);

            CUDA_CALL(cudaSetDevice(0));
            thrust::device_vector<T> d_chunkKeys(
                h_keys.begin() + chunkOffset, h_keys.begin() + chunkOffset + chunkSize
            );
            thrust::device_vector<size_t> d_originalIndices(chunkSize);
            thrust::sequence(thrust::device, d_originalIndices.begin(), d_originalIndices.end());

            thrust::device_vector<size_t> d_gpuIndices(chunkSize);
            Partitioner partitioner{numGPUs};
            thrust::transform(
                thrust::device,
                d_chunkKeys.begin(),
                d_chunkKeys.end(),
                d_gpuIndices.begin(),
                partitioner
            );
            thrust::sort_by_key(
                thrust::device,
                d_gpuIndices.begin(),
                d_gpuIndices.end(),
                thrust::make_zip_iterator(
                    thrust::make_tuple(d_chunkKeys.begin(), d_originalIndices.begin())
                )
            );

            thrust::device_vector<size_t> d_counts(numGPUs);
            thrust::device_vector<size_t> d_offsets(numGPUs);
            thrust::device_vector<size_t> d_uniqueKeys(numGPUs);

            thrust::reduce_by_key(
                thrust::device,
                d_gpuIndices.begin(),
                d_gpuIndices.end(),
                thrust::make_constant_iterator<size_t>(1),
                d_uniqueKeys.begin(),
                d_counts.begin()
            );

            thrust::exclusive_scan(
                thrust::device, d_counts.begin(), d_counts.end(), d_offsets.begin(), 0
            );

            thrust::host_vector<size_t> h_counts = d_counts;
            thrust::host_vector<size_t> h_offsets = d_offsets;

            thrust::device_vector<bool> d_chunkResultsSorted(chunkSize);
            for (size_t gpuId = 0; gpuId < numGPUs; ++gpuId) {
                if (h_counts[gpuId] > 0) {
                    CUDA_CALL(cudaSetDevice(gpuId));
                    thrust::device_vector<T> d_receivedKeys(h_counts[gpuId]);
                    thrust::device_vector<bool> d_gpuResults(h_counts[gpuId]);

                    CUDA_CALL(cudaMemcpyPeerAsync(
                        d_receivedKeys.data().get(),
                        gpuId,
                        d_chunkKeys.data().get() + h_offsets[gpuId],
                        0,
                        h_counts[gpuId] * sizeof(T),
                        streams[gpuId]
                    ));
                    filters[gpuId]->deleteMany(d_receivedKeys, d_gpuResults, streams[gpuId]);
                    CUDA_CALL(cudaMemcpyPeerAsync(
                        d_chunkResultsSorted.data().get() + h_offsets[gpuId],
                        0,
                        d_gpuResults.data().get(),
                        gpuId,
                        h_counts[gpuId] * sizeof(bool),
                        streams[gpuId]
                    ));
                }
            }

            for (size_t gpuId = 0; gpuId < numGPUs; ++gpuId) {
                CUDA_CALL(cudaSetDevice(gpuId));
                CUDA_CALL(cudaStreamSynchronize(streams[gpuId]));
            }

            CUDA_CALL(cudaSetDevice(0));
            thrust::device_vector<bool> d_chunkOutput(chunkSize);
            thrust::scatter(
                thrust::device,
                d_chunkResultsSorted.begin(),
                d_chunkResultsSorted.end(),
                d_originalIndices.begin(),
                d_chunkOutput.begin()
            );
            thrust::copy(
                d_chunkOutput.begin(), d_chunkOutput.end(), h_output.begin() + chunkOffset
            );
        }

        CUDA_CALL(cudaSetDevice(0));
        return totalOccupiedSlots();
    }

    float loadFactor() {
        return static_cast<float>(totalOccupiedSlots()) / static_cast<float>(totalCapacity());
    }

    size_t totalOccupiedSlots() {
        size_t total = 0;
        for (size_t i = 0; i < numGPUs; ++i) {
            CUDA_CALL(cudaSetDevice(i));
            total += filters[i]->occupiedSlots();
        }
        CUDA_CALL(cudaSetDevice(0));
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