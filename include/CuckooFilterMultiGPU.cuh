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
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>
#include "CuckooFilter.cuh"
#include "helpers.cuh"

template <typename Config>
class CuckooFilterMultiGPU {
   public:
    using T = typename Config::KeyType;

    static constexpr double CHUNK_SIZE_FACTOR = 0.8;
    static constexpr double CAPACITY_HEADROOM = 1.05;

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
    size_t primaryGPU;
    size_t chunkSize;
    std::vector<CuckooFilter<Config>*> filters;
    std::vector<cudaStream_t> streams;

    /**
     * @brief Partitions a device vector of keys by their target GPU.
     *
     * This function takes a vector of keys, determines the target GPU for each key
     * and then sorts the keys according to their destination GPU.
     * It can optionally sort a companion vector of indices at the same time.
     * The results (counts and offsets for each GPU's data) are returned via output parameters.
     *
     * @param d_keys The device vector of keys to be partitioned and sorted. Modified in place.
     * @param d_indicesToSort (Optional) A pointer to a device vector of indices to be sorted
     *        along with the keys. Modified in place.
     * @param h_counts A host vector to be filled with the number of keys for each GPU.
     * @param h_offsets A host vector to be filled with the starting offset for each GPU's data.
     */
    void partitionKeysByGpu(
        thrust::device_vector<T>& d_keys,
        thrust::device_vector<size_t>* d_indicesToSort,
        thrust::host_vector<size_t>& h_counts,
        thrust::host_vector<size_t>& h_offsets
    ) {
        size_t n = d_keys.size();
        thrust::device_vector<size_t> d_gpuIndices(n);
        Partitioner partitioner{numGPUs};
        thrust::transform(
            thrust::device, d_keys.begin(), d_keys.end(), d_gpuIndices.begin(), partitioner
        );

        if (d_indicesToSort) {
            thrust::sort_by_key(
                thrust::device,
                d_gpuIndices.begin(),
                d_gpuIndices.end(),
                thrust::make_zip_iterator(
                    thrust::make_tuple(d_keys.begin(), d_indicesToSort->begin())
                )
            );
        } else {
            thrust::sort_by_key(
                thrust::device, d_gpuIndices.begin(), d_gpuIndices.end(), d_keys.begin()
            );
        }

        thrust::device_vector<size_t> d_counts(numGPUs);
        thrust::device_vector<size_t> d_offsets(numGPUs);
        thrust::device_vector<size_t> d_uniqueGpuIds(numGPUs);
        thrust::reduce_by_key(
            thrust::device,
            d_gpuIndices.begin(),
            d_gpuIndices.end(),
            thrust::make_constant_iterator<size_t>(1),
            d_uniqueGpuIds.begin(),
            d_counts.begin()
        );
        thrust::exclusive_scan(
            thrust::device, d_counts.begin(), d_counts.end(), d_offsets.begin(), 0
        );

        h_counts = d_counts;
        h_offsets = d_offsets;
    }

    template <typename FilterFunctor>
    void queryAndReorder(
        const thrust::host_vector<T>& h_keys,
        thrust::host_vector<bool>& h_output,
        FilterFunctor filterOp
    ) {
        size_t n = h_keys.size();
        h_output.resize(n);

        if (n == 0) {
            return;
        }

        for (size_t chunkOffset = 0; chunkOffset < n; chunkOffset += chunkSize) {
            size_t currentChunkSize = std::min(chunkSize, n - chunkOffset);

            CUDA_CALL(cudaSetDevice(primaryGPU));
            thrust::device_vector<T> d_chunkKeys(
                h_keys.begin() + chunkOffset, h_keys.begin() + chunkOffset + currentChunkSize
            );
            thrust::device_vector<size_t> d_originalIndices(currentChunkSize);
            thrust::sequence(thrust::device, d_originalIndices.begin(), d_originalIndices.end());

            thrust::host_vector<size_t> h_counts(numGPUs);
            thrust::host_vector<size_t> h_offsets(numGPUs);
            partitionKeysByGpu(d_chunkKeys, &d_originalIndices, h_counts, h_offsets);

            thrust::device_vector<bool> d_chunkResultsSorted(currentChunkSize);

            parallelForGPUs([&](size_t gpuId) {
                if (h_counts[gpuId] > 0) {
                    thrust::device_vector<T> d_receivedKeys(h_counts[gpuId]);
                    thrust::device_vector<bool> d_gpuResults(h_counts[gpuId]);

                    CUDA_CALL(cudaMemcpyPeerAsync(
                        d_receivedKeys.data().get(),
                        gpuId,
                        d_chunkKeys.data().get() + h_offsets[gpuId],
                        primaryGPU,
                        h_counts[gpuId] * sizeof(T),
                        streams[gpuId]
                    ));

                    filterOp(filters[gpuId], d_receivedKeys, d_gpuResults, streams[gpuId]);

                    CUDA_CALL(cudaMemcpyPeerAsync(
                        d_chunkResultsSorted.data().get() + h_offsets[gpuId],
                        primaryGPU,
                        d_gpuResults.data().get(),
                        gpuId,
                        h_counts[gpuId] * sizeof(bool),
                        streams[primaryGPU]
                    ));
                }
            });

            synchronizeAllGPUs();

            CUDA_CALL(cudaSetDevice(primaryGPU));
            thrust::device_vector<bool> d_chunkOutput(currentChunkSize);
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
        CUDA_CALL(cudaSetDevice(primaryGPU));
    }

   public:
    CuckooFilterMultiGPU(size_t numGPUs, size_t capacity, size_t primaryGPU = 0)
        : numGPUs(numGPUs),
          capacityPerGPU(static_cast<size_t>(SDIV(capacity, numGPUs) * CAPACITY_HEADROOM)),
          primaryGPU(primaryGPU) {
        assert(numGPUs > 0 && "Number of GPUs must be at least 1");
        assert(primaryGPU < numGPUs && "Invalid primary GPU index");

        streams.resize(numGPUs);
        parallelForGPUs([&](size_t i) {
            CUDA_CALL(cudaStreamCreate(&streams[i]));
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
        });

        filters.resize(numGPUs);
        parallelForGPUs([&](size_t i) { filters[i] = new CuckooFilter<Config>(capacityPerGPU); });

        CUDA_CALL(cudaSetDevice(primaryGPU));
        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        chunkSize =
            static_cast<size_t>(static_cast<double>(totalMem) * CHUNK_SIZE_FACTOR / sizeof(T));
    }

    ~CuckooFilterMultiGPU() {
        parallelForGPUs([&](size_t i) {
            delete filters[i];
            CUDA_CALL(cudaStreamDestroy(streams[i]));
        });

        CUDA_CALL(cudaSetDevice(primaryGPU));
    }

    CuckooFilterMultiGPU(const CuckooFilterMultiGPU&) = delete;
    CuckooFilterMultiGPU& operator=(const CuckooFilterMultiGPU&) = delete;

    size_t insertMany(const thrust::host_vector<T>& h_keys) {
        size_t n = h_keys.size();
        if (n == 0) {
            return totalOccupiedSlots();
        }

        for (size_t chunkOffset = 0; chunkOffset < n; chunkOffset += chunkSize) {
            size_t currentChunkSize = std::min(chunkSize, n - chunkOffset);

            CUDA_CALL(cudaSetDevice(primaryGPU));
            thrust::device_vector<T> d_chunkKeys(
                h_keys.begin() + chunkOffset, h_keys.begin() + chunkOffset + currentChunkSize
            );

            thrust::host_vector<size_t> h_counts;
            thrust::host_vector<size_t> h_offsets;
            partitionKeysByGpu(d_chunkKeys, nullptr, h_counts, h_offsets);

            parallelForGPUs([&](size_t gpuId) {
                if (h_counts[gpuId] > 0) {
                    thrust::device_vector<T> d_receivedKeys(h_counts[gpuId]);
                    CUDA_CALL(cudaMemcpyPeerAsync(
                        d_receivedKeys.data().get(),
                        gpuId,
                        d_chunkKeys.data().get() + h_offsets[gpuId],
                        primaryGPU,
                        h_counts[gpuId] * sizeof(T),
                        streams[gpuId]
                    ));
                    filters[gpuId]->insertMany(d_receivedKeys, streams[gpuId]);
                }
            });

            synchronizeAllGPUs();
        }

        CUDA_CALL(cudaSetDevice(primaryGPU));
        return totalOccupiedSlots();
    }

    void containsMany(const thrust::host_vector<T>& h_keys, thrust::host_vector<bool>& h_output) {
        queryAndReorder(
            h_keys,
            h_output,
            [](CuckooFilter<Config>* filter,
               const thrust::device_vector<T>& keys,
               thrust::device_vector<bool>& results,
               cudaStream_t stream) { filter->containsMany(keys, results, stream); }
        );
    }

    size_t deleteMany(const thrust::host_vector<T>& h_keys, thrust::host_vector<bool>& h_output) {
        queryAndReorder(
            h_keys,
            h_output,
            [](CuckooFilter<Config>* filter,
               const thrust::device_vector<T>& keys,
               thrust::device_vector<bool>& results,
               cudaStream_t stream) { filter->deleteMany(keys, results, stream); }
        );
        return totalOccupiedSlots();
    }

    float loadFactor() {
        return static_cast<float>(totalOccupiedSlots()) / static_cast<float>(totalCapacity());
    }

    template <typename Func>
    void parallelForGPUs(Func func) const {
        for (size_t i = 0; i < numGPUs; ++i) {
            CUDA_CALL(cudaSetDevice(i));
            func(i);
        }
    }

    void synchronizeAllGPUs() {
        parallelForGPUs([&](size_t i) { CUDA_CALL(cudaStreamSynchronize(streams[i])); });
    }

    size_t totalOccupiedSlots() {
        std::atomic<size_t> total(0);
        parallelForGPUs([&](size_t i) {
            total.fetch_add(filters[i]->occupiedSlots(), std::memory_order_relaxed);
        });
        return total;
    }

    void clear() {
        parallelForGPUs([&](size_t i) { filters[i]->clear(); });
    }

    [[nodiscard]] size_t totalCapacity() const {
        std::atomic<size_t> total(0);

        parallelForGPUs([&](size_t i) {
            total.fetch_add(filters[i]->capacity(), std::memory_order_relaxed);
        });

        return total;
    }
};