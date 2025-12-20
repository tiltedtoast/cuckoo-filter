#pragma once

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>
#include <type_traits>
#include <vector>
#include "CuckooFilter.cuh"
#include "helpers.cuh"

#include <gossip.cuh>
#include <plan_parser.hpp>

/**
 * @brief A multi-GPU implementation of the Cuckoo Filter.
 *
 * This class partitions keys across multiple GPUs using the gossip library
 * for efficient multi-GPU communication. It handles data distribution using
 * gossip's multisplit and all-to-all primitives, and aggregates results.
 *
 * @tparam Config The configuration structure for the Cuckoo Filter.
 */
template <typename Config>
class CuckooFilterMultiGPU {
   public:
    using T = typename Config::KeyType;

    /**
     * @brief Functor for partitioning keys across GPUs.
     *
     * Uses a hash function to assign each key to a specific GPU index.
     * Compatible with gossip's multisplit which requires __host__ __device__ functor.
     */
    struct Partitioner {
        size_t numGPUs;

        __host__ __device__ gossip::gpu_id_t operator()(const T& key) const {
            uint64_t hash = CuckooFilter<Config>::hash64(key);
            return static_cast<gossip::gpu_id_t>(hash % numGPUs);
        }
    };

   private:
    size_t numGPUs;
    size_t capacityPerGPU;
    std::vector<CuckooFilter<Config>*> filters;

    gossip::context_t gossipContext;
    gossip::multisplit_t multisplit;
    gossip::all2all_t all2all;
    gossip::all2all_t all2allResults;

    // Pre-allocated per-GPU buffers for gossip operations
    std::vector<T*> srcBuffers;
    std::vector<T*> dstBuffers;
    std::vector<size_t> bufferCapacities;

    std::vector<size_t> resultBufferCapacities;
    std::vector<bool*> resultSrcBuffers;
    std::vector<bool*> resultDstBuffers;

    /**
     * @brief Gets the available free memory for each GPU.
     * @return A vector where the i-th element is the free memory on GPU i.
     */
    [[nodiscard]] std::vector<size_t> getGpuMemoryInfo() const {
        std::vector<size_t> freeMem(numGPUs);
        parallelForGPUs([&](size_t gpuId) {
            size_t free, total;
            CUDA_CALL(cudaMemGetInfo(&free, &total));
            freeMem[gpuId] = free;
        });
        return freeMem;
    }

    /**
     * @brief Ensures per-GPU buffers have sufficient capacity.
     * @param requiredPerGPU Required capacity for each GPU's buffer.
     */
    void ensureBufferCapacity(size_t requiredPerGPU) {
        for (size_t gpu = 0; gpu < numGPUs; ++gpu) {
            cudaSetDevice(gossipContext.get_device_id(gpu));

            // Key buffers
            if (bufferCapacities[gpu] < requiredPerGPU) {
                if (srcBuffers[gpu]) {
                    cudaFree(srcBuffers[gpu]);
                }
                if (dstBuffers[gpu]) {
                    cudaFree(dstBuffers[gpu]);
                }

                CUDA_CALL(cudaMalloc(&srcBuffers[gpu], requiredPerGPU * sizeof(T)));
                CUDA_CALL(cudaMalloc(&dstBuffers[gpu], requiredPerGPU * sizeof(T)));
                bufferCapacities[gpu] = requiredPerGPU;
            }

            // Result buffers
            if (resultBufferCapacities[gpu] < requiredPerGPU) {
                if (resultSrcBuffers[gpu]) {
                    cudaFree(resultSrcBuffers[gpu]);
                }
                if (resultDstBuffers[gpu]) {
                    cudaFree(resultDstBuffers[gpu]);
                }

                CUDA_CALL(cudaMalloc(&resultSrcBuffers[gpu], requiredPerGPU * sizeof(bool)));
                CUDA_CALL(cudaMalloc(&resultDstBuffers[gpu], requiredPerGPU * sizeof(bool)));
                resultBufferCapacities[gpu] = requiredPerGPU;
            }
        }
    }

    /**
     * @brief Free all pre-allocated buffers.
     */
    void freeBuffers() {
        for (size_t gpu = 0; gpu < numGPUs; ++gpu) {
            cudaSetDevice(gossipContext.get_device_id(gpu));
            if (srcBuffers[gpu]) {
                cudaFree(srcBuffers[gpu]);
                srcBuffers[gpu] = nullptr;
            }
            if (dstBuffers[gpu]) {
                cudaFree(dstBuffers[gpu]);
                dstBuffers[gpu] = nullptr;
            }
            if (resultSrcBuffers[gpu]) {
                cudaFree(resultSrcBuffers[gpu]);
                resultSrcBuffers[gpu] = nullptr;
            }
            if (resultDstBuffers[gpu]) {
                cudaFree(resultDstBuffers[gpu]);
                resultDstBuffers[gpu] = nullptr;
            }
            bufferCapacities[gpu] = 0;
            resultBufferCapacities[gpu] = 0;
        }
    }

    /**
     * @brief Generic function for processing keys using gossip primitives.
     *
     * The workflow is:
     * 1. Distribute input keys to GPUs
     * 2. Use gossip multisplit to partition keys by target GPU
     * 3. Use gossip all2all to shuffle keys to correct GPUs
     * 4. Execute filter operation locally
     * 5. Use gossip all2all to return results (if hasOutput)
     * 6. Reorder results to match original input order
     *
     * @param h_keys Host vector of keys to process.
     * @param h_output Optional host vector to store the boolean results.
     * @param filterOp The specific CuckooFilter operation to execute.
     * @tparam returnOccupied If true, returns total occupied slots after operation.
     * @tparam hasOutput If true, expects an output buffer for results.
     * @return Total occupied slots if returnOccupied is true, otherwise 0.
     */
    template <bool returnOccupied, bool hasOutput, typename FilterFunc>
    size_t executeOperation(
        const thrust::host_vector<T>& h_keys,
        thrust::host_vector<bool>* h_output,
        FilterFunc filterOp
    ) {
        size_t n = h_keys.size();

        if constexpr (hasOutput) {
            h_output->resize(n);
        }

        if (n == 0) {
            return returnOccupied ? totalOccupiedSlots() : 0;
        }

        // gossip recommends this for random distributions
        const float memoryFactor = 1.5f;
        const auto perGPUCapacity =
            static_cast<size_t>(std::ceil(static_cast<double>(n) / numGPUs * memoryFactor));

        ensureBufferCapacity(perGPUCapacity);

        // Distribute input keys evenly across GPUs initially
        std::vector<size_t> inputLens(numGPUs);
        std::vector<size_t> inputOffsets(numGPUs + 1, 0);
        size_t keysPerGPU = n / numGPUs;
        size_t remainder = n % numGPUs;

        for (size_t gpu = 0; gpu < numGPUs; ++gpu) {
            inputLens[gpu] = keysPerGPU + (gpu < remainder ? 1 : 0);
            inputOffsets[gpu + 1] = inputOffsets[gpu] + inputLens[gpu];
        }

        // Copy input data to source buffers on each GPU
        parallelForGPUs([&](size_t gpuId) {
            if (inputLens[gpuId] > 0) {
                CUDA_CALL(cudaMemcpy(
                    srcBuffers[gpuId],
                    h_keys.data() + inputOffsets[gpuId],
                    inputLens[gpuId] * sizeof(T),
                    cudaMemcpyHostToDevice
                ));
            }
        });
        gossipContext.sync_hard();

        // Phase 1: Multisplit - partition keys by target GPU
        std::vector<std::vector<size_t>> partitionTable(numGPUs, std::vector<size_t>(numGPUs));
        std::vector<size_t> dstLens(numGPUs, perGPUCapacity);

        Partitioner partitioner{numGPUs};
        multisplit.execAsync(
            srcBuffers,      // source pointers (per GPU)
            inputLens,       // source lengths (per GPU)
            dstBuffers,      // destination pointers (per GPU)
            dstLens,         // destination capacities (per GPU)
            partitionTable,  // output: partition counts [src][dst]
            partitioner
        );
        multisplit.sync();

        std::swap(srcBuffers, dstBuffers);

        // Calculate how many keys each GPU will receive after all2all
        std::vector<size_t> recvCounts(numGPUs, 0);
        for (size_t dst = 0; dst < numGPUs; ++dst) {
            for (size_t src = 0; src < numGPUs; ++src) {
                recvCounts[dst] += partitionTable[src][dst];
            }
        }

        // Phase 2: shuffle partitioned keys to correct GPUs
        all2all.execAsync(
            srcBuffers,     // partitioned source data
            dstLens,        // source buffer capacities
            dstBuffers,     // destination for received data
            dstLens,        // destination buffer capacities
            partitionTable  // partition counts from multisplit
        );
        all2all.sync();

        // If no output is required, execute filter ops and we're done
        if constexpr (!hasOutput) {
            parallelForGPUs([&](size_t gpuId) {
                size_t localCount = recvCounts[gpuId];
                if (localCount == 0) {
                    return;
                }
                auto stream = gossipContext.get_streams(gpuId)[0];
                filterOp(filters[gpuId], dstBuffers[gpuId], nullptr, localCount, stream);
            });
            gossipContext.sync_all_streams();
            return returnOccupied ? totalOccupiedSlots() : 0;
        }

        // Phase 3: Prepare table for reverse all-to-all
        std::vector<std::vector<size_t>> reverseTable(numGPUs, std::vector<size_t>(numGPUs));
        for (size_t src = 0; src < numGPUs; ++src) {
            for (size_t dst = 0; dst < numGPUs; ++dst) {
                reverseTable[dst][src] = partitionTable[src][dst];
            }
        }

        // Phase 4: Execute filter operations
        parallelForGPUs([&](size_t gpuId) {
            size_t localCount = recvCounts[gpuId];
            if (localCount == 0) {
                return;
            }
            auto stream = gossipContext.get_streams(gpuId)[0];
            filterOp(
                filters[gpuId], dstBuffers[gpuId], resultSrcBuffers[gpuId], localCount, stream
            );
        });
        gossipContext.sync_all_streams();

        all2allResults.execAsync(
            resultSrcBuffers, recvCounts, resultDstBuffers, dstLens, reverseTable
        );
        all2allResults.sync();

        // Phase 5: Copy results back to host and reorder
        std::vector<size_t> returnCounts(numGPUs);
        for (size_t gpu = 0; gpu < numGPUs; ++gpu) {
            returnCounts[gpu] = 0;
            for (size_t src = 0; src < numGPUs; ++src) {
                returnCounts[gpu] += reverseTable[src][gpu];
            }
        }

        parallelForGPUs([&](size_t gpuId) {
            size_t localCount = returnCounts[gpuId];
            if (localCount == 0) {
                return;
            }

            std::vector<uint8_t> h_localResults(localCount);
            CUDA_CALL(cudaMemcpy(
                h_localResults.data(),
                resultDstBuffers[gpuId],
                localCount * sizeof(bool),
                cudaMemcpyDeviceToHost
            ));

            for (size_t i = 0; i < localCount; ++i) {
                (*h_output)[inputOffsets[gpuId] + i] = static_cast<bool>(h_localResults[i]);
            }
        });
        gossipContext.sync_hard();

        return returnOccupied ? totalOccupiedSlots() : 0;
    }

   public:
    /**
     * @brief Constructs a new CuckooFilterMultiGPU with default transfer plan.
     *
     * Initializes gossip context, multisplit, all-to-all primitives, and CuckooFilter instances
     * on each available GPU.
     *
     * @param numGPUs Number of GPUs to use.
     * @param capacity Total capacity of the distributed filter.
     */
    CuckooFilterMultiGPU(size_t numGPUs, size_t capacity)
        : numGPUs(numGPUs),
          capacityPerGPU(static_cast<size_t>(SDIV(capacity, numGPUs) * 1.02)),
          gossipContext(numGPUs),
          multisplit(gossipContext),
          all2all(gossipContext, gossip::all2all::default_plan(numGPUs)),
          all2allResults(gossipContext, gossip::all2all::default_plan(numGPUs)),
          srcBuffers(numGPUs, nullptr),
          dstBuffers(numGPUs, nullptr),
          bufferCapacities(numGPUs, 0),
          resultSrcBuffers(numGPUs, nullptr),
          resultDstBuffers(numGPUs, nullptr),
          resultBufferCapacities(numGPUs, 0) {
        assert(numGPUs > 0 && "Number of GPUs must be at least 1");

        filters.resize(numGPUs);

        for (size_t i = 0; i < numGPUs; ++i) {
            CUDA_CALL(cudaSetDevice(gossipContext.get_device_id(i)));
            CuckooFilter<Config>* filter;
            CUDA_CALL(cudaMallocManaged(&filter, sizeof(CuckooFilter<Config>)));
            new (filter) CuckooFilter<Config>(capacityPerGPU);
            filters[i] = filter;
        }
        gossipContext.sync_hard();
    }

    /**
     * @brief Constructs a new CuckooFilterMultiGPU with custom transfer plan.
     *
     * Initializes gossip context, multisplit, all-to-all primitives with provided
     * transfer plan loaded from file, and CuckooFilter instances on each available GPU.
     *
     * @param numGPUs Number of GPUs to use.
     * @param capacity Total capacity of the distributed filter.
     * @param transferPlanPath Path to gossip transfer plan file for optimized topology-aware
     * transfers.
     */
    CuckooFilterMultiGPU(size_t numGPUs, size_t capacity, const char* transferPlanPath)
        : numGPUs(numGPUs),
          capacityPerGPU(static_cast<size_t>(SDIV(capacity, numGPUs) * 1.02)),
          gossipContext(numGPUs),
          multisplit(gossipContext),
          all2all(
              gossipContext,
              [&]() {
                  auto plan = parse_plan(transferPlanPath);
                  if (plan.num_gpus() == 0) {
                      return gossip::all2all::default_plan(numGPUs);
                  }
                  return plan;
              }()
          ),
          all2allResults(
              gossipContext,
              [&]() {
                  auto plan = parse_plan(transferPlanPath);
                  if (plan.num_gpus() == 0) {
                      return gossip::all2all::default_plan(numGPUs);
                  }
                  return plan;
              }()
          ),
          srcBuffers(numGPUs, nullptr),
          dstBuffers(numGPUs, nullptr),
          bufferCapacities(numGPUs, 0),
          resultSrcBuffers(numGPUs, nullptr),
          resultDstBuffers(numGPUs, nullptr),
          resultBufferCapacities(numGPUs, 0) {
        assert(numGPUs > 0 && "Number of GPUs must be at least 1");

        filters.resize(numGPUs);

        for (size_t i = 0; i < numGPUs; ++i) {
            CUDA_CALL(cudaSetDevice(gossipContext.get_device_id(i)));
            CuckooFilter<Config>* filter;
            CUDA_CALL(cudaMallocManaged(&filter, sizeof(CuckooFilter<Config>)));
            new (filter) CuckooFilter<Config>(capacityPerGPU);
            filters[i] = filter;
        }
        gossipContext.sync_hard();
    }

    /**
     * @brief Destroys the CuckooFilterMultiGPU.
     *
     * Cleans up filter instances and pre-allocated buffers.
     */
    ~CuckooFilterMultiGPU() {
        freeBuffers();
        for (size_t i = 0; i < numGPUs; ++i) {
            CUDA_CALL(cudaSetDevice(gossipContext.get_device_id(i)));
            filters[i]->~CuckooFilter<Config>();
            CUDA_CALL(cudaFree(filters[i]));
        }
    }

    CuckooFilterMultiGPU(const CuckooFilterMultiGPU&) = delete;
    CuckooFilterMultiGPU& operator=(const CuckooFilterMultiGPU&) = delete;

    /**
     * @brief Inserts a batch of keys into the distributed filter.
     * Uses gossip primitives for efficient multi-GPU data distribution.
     * @param h_keys The keys to insert.
     * @return The total number of occupied slots across all GPUs after insertion.
     */
    size_t insertMany(const thrust::host_vector<T>& h_keys) {
        return executeOperation<true, false>(
            h_keys,
            nullptr,
            [](CuckooFilter<Config>* filter,
               const T* keys,
               bool* /*unused_results*/,
               size_t count,
               cudaStream_t stream) { filter->insertMany(keys, count, stream); }
        );
    }

    /**
     * @brief Checks for the presence of multiple keys in the filter.
     * @param h_keys The keys to check.
     * @param h_output A host vector to store the results (true if present, false otherwise).
     */
    void containsMany(const thrust::host_vector<T>& h_keys, thrust::host_vector<bool>& h_output) {
        executeOperation<false, true>(
            h_keys,
            &h_output,
            [](CuckooFilter<Config>* filter,
               const T* keys,
               bool* results,
               size_t count,
               cudaStream_t stream) { filter->containsMany(keys, count, results, stream); }
        );
    }

    /**
     * @brief Deletes multiple keys from the filter.
     * @param h_keys The keys to delete.
     * @param h_output A host vector to store the results (true if a key was found and deleted).
     * @return The total number of occupied slots across all GPUs after deletion.
     */
    size_t deleteMany(const thrust::host_vector<T>& h_keys, thrust::host_vector<bool>& h_output) {
        return executeOperation<true, true>(
            h_keys,
            &h_output,
            [](CuckooFilter<Config>* filter,
               const T* keys,
               bool* results,
               size_t count,
               cudaStream_t stream) { filter->deleteMany(keys, count, results, stream); }
        );
    }

    /**
     * @brief Deletes multiple keys from the filter without returning per-key results.
     * @param h_keys The keys to delete.
     * @return The total number of occupied slots across all GPUs after deletion.
     */
    size_t deleteMany(const thrust::host_vector<T>& h_keys) {
        return executeOperation<true, false>(
            h_keys,
            nullptr,
            [](CuckooFilter<Config>* filter,
               const T* keys,
               bool* /*unused_results*/,
               size_t count,
               cudaStream_t stream) { filter->deleteMany(keys, count, nullptr, stream); }
        );
    }

    /**
     * @brief Calculates the global load factor.
     * @return float Load factor (total occupied / total capacity).
     */
    [[nodiscard]] float loadFactor() const {
        return static_cast<float>(totalOccupiedSlots()) / static_cast<float>(totalCapacity());
    }

    /**
     * @brief Executes a function in parallel across all GPUs.
     *
     * Spawns a thread for each GPU to run the provided function.
     *
     * @tparam Func Type of the function to execute.
     * @param func The function to execute, taking the GPU index as an argument.
     */
    template <typename Func>
    void parallelForGPUs(Func func) const {
        std::vector<std::thread> threads;
        for (size_t i = 0; i < numGPUs; ++i) {
            threads.emplace_back([=, this]() {
                CUDA_CALL(cudaSetDevice(gossipContext.get_device_id(i)));
                func(i);
            });
        }

        for (auto& t : threads) {
            t.join();
        }
    }

    /**
     * @brief Synchronizes all GPU streams used by this filter.
     */
    void synchronizeAllGPUs() {
        gossipContext.sync_all_streams();
    }

    /**
     * @brief Returns the total number of occupied slots across all GPUs.
     * @return size_t Total occupied slots.
     */
    [[nodiscard]] size_t totalOccupiedSlots() const {
        std::atomic<size_t> total(0);
        parallelForGPUs([&](size_t i) {
            total.fetch_add(filters[i]->occupiedSlots(), std::memory_order_relaxed);
        });

        return total.load();
    }

    /**
     * @brief Clears all filters on all GPUs.
     */
    void clear() {
        parallelForGPUs([&](size_t i) { filters[i]->clear(); });
    }

    /**
     * @brief Returns the total capacity of the distributed filter.
     * @return size_t Total capacity.
     */
    [[nodiscard]] size_t totalCapacity() const {
        std::atomic<size_t> total(0);
        parallelForGPUs([&](size_t i) {
            total.fetch_add(filters[i]->capacity(), std::memory_order_relaxed);
        });
        return total.load();
    }

    [[nodiscard]] size_t sizeInBytes() const {
        std::atomic<size_t> total(0);
        parallelForGPUs([&](size_t i) {
            total.fetch_add(filters[i]->sizeInBytes(), std::memory_order_relaxed);
        });
        return total.load();
    }
};