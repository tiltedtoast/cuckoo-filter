#include <thrust/host_vector.h>
#include <chrono>
#include <iostream>
#include <random>
#include <thread>
#include <vector>
#include "CuckooFilterIPC.cuh"

using Config = CuckooConfig<uint64_t, 16, 500, 256, 16>;

constexpr double TARGET_LOAD_FACTOR = 0.95;

void runServer(const std::string& name, size_t capacity, bool forceShutdown = false) {
    std::cout << "Starting server with capacity: " << capacity << std::endl;
    if (forceShutdown) {
        std::cout << "Force shutdown mode enabled (pending requests will be cancelled)"
                  << std::endl;
    }

    try {
        CuckooFilterIPCServer<Config> server(name, capacity);
        server.start();

        std::cout << "Server running. Press Enter to stop..." << std::endl;
        std::cin.get();

        server.stop(forceShutdown);
        std::cout << "Server stopped." << std::endl;

        auto filter = server.getFilter();
        std::cout << "Final load factor: " << filter->loadFactor() << std::endl;
        std::cout << "Occupied slots: " << filter->occupiedSlots() << std::endl;
        std::cout << "Capacity: " << filter->capacity() << std::endl;
        std::cout << "Size in bytes: " << filter->sizeInBytes() << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Server error: " << e.what() << std::endl;
    }
}

void runClient(const std::string& name, int clientId, size_t numKeys) {
    std::cout << "Client " << clientId << " starting..." << std::endl;

    try {
        // Give the server some time to initialise
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        CuckooFilterIPCClient<Config> client(name);

        std::vector<uint64_t> h_keys(numKeys);
        std::random_device rd;
        std::mt19937_64 gen(rd() + clientId);
        std::uniform_int_distribution<uint64_t> dis(1, UINT32_MAX);
        for (size_t i = 0; i < numKeys; i++) {
            h_keys[i] = dis(gen);
        }

        uint64_t* d_keys;
        bool* d_results;
        CUDA_CALL(cudaMalloc(&d_keys, numKeys * sizeof(uint64_t)));
        CUDA_CALL(cudaMalloc(&d_results, numKeys * sizeof(bool)));
        CUDA_CALL(
            cudaMemcpy(d_keys, h_keys.data(), numKeys * sizeof(uint64_t), cudaMemcpyHostToDevice)
        );

        auto start = std::chrono::high_resolution_clock::now();

        size_t occupiedAfterInsert = client.insertMany(d_keys, numKeys);
        std::cout << "Client " << clientId << " inserted " << numKeys << " keys (filter now has "
                  << occupiedAfterInsert << " occupied slots)" << std::endl;

        client.containsMany(d_keys, numKeys, d_results);

        std::vector<uint8_t> h_results(numKeys);
        CUDA_CALL(
            cudaMemcpy(h_results.data(), d_results, numKeys * sizeof(bool), cudaMemcpyDeviceToHost)
        );

        size_t found = 0;
        for (bool result : h_results) {
            if (result) {
                found++;
            }
        }
        std::cout << "Client " << clientId << " found " << found << "/" << numKeys << " keys"
                  << std::endl;

        size_t deleteCount = numKeys / 2;
        size_t occupiedAfterDelete = client.deleteMany(d_keys, deleteCount, d_results);
        std::cout << "Client " << clientId << " deleted " << deleteCount << " keys (filter now has "
                  << occupiedAfterDelete << " occupied slots)" << std::endl;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Client " << clientId << " completed in " << duration.count() << "ms"
                  << std::endl;

        CUDA_CALL(cudaFree(d_keys));
        CUDA_CALL(cudaFree(d_results));

    } catch (const std::exception& e) {
        std::cerr << "Client " << clientId << " error: " << e.what() << std::endl;
    }
}

void runClientThrust(const std::string& name, int clientId, size_t numKeys) {
    std::cout << "Thrust Client " << clientId << " starting..." << std::endl;

    try {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));

        CuckooFilterIPCClientThrust<Config> client(name);

        thrust::host_vector<uint64_t> h_keys(numKeys);
        std::random_device rd;
        std::mt19937_64 gen(rd() + clientId);
        std::uniform_int_distribution<uint64_t> dis(1, UINT32_MAX);
        for (size_t i = 0; i < numKeys; i++) {
            h_keys[i] = dis(gen);
        }

        thrust::device_vector<uint64_t> d_keys = h_keys;
        thrust::device_vector<bool> d_results(numKeys);

        auto start = std::chrono::high_resolution_clock::now();

        size_t occupiedAfterInsert = client.insertMany(d_keys);
        std::cout << "Thrust Client " << clientId << " inserted " << numKeys
                  << " keys (filter now has " << occupiedAfterInsert << " occupied slots)"
                  << std::endl;

        client.containsMany(d_keys, d_results);

        thrust::host_vector<bool> h_results = d_results;
        size_t found = 0;
        for (bool result : h_results) {
            if (result) {
                found++;
            }
        }
        std::cout << "Thrust Client " << clientId << " found " << found << "/" << numKeys << " keys"
                  << std::endl;

        size_t deleteCount = numKeys / 2;
        thrust::device_vector<uint64_t> d_keysToDelete(
            d_keys.begin(), d_keys.begin() + deleteCount
        );
        size_t occupiedAfterDelete = client.deleteMany(d_keysToDelete);
        std::cout << "Thrust Client " << clientId << " deleted " << deleteCount
                  << " keys (filter now has " << occupiedAfterDelete << " occupied slots)"
                  << std::endl;

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        std::cout << "Thrust Client " << clientId << " completed in " << duration.count() << "ms"
                  << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Thrust Client " << clientId << " error: " << e.what() << std::endl;
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cout << "Usage:" << std::endl;
        std::cout << "    Server: " << argv[0] << " server <name> [capacity_exponent] [--force]"
                  << std::endl;
        std::cout << "            capacity_exponent: capacity = 2^x (default: 25)" << std::endl;
        std::cout << "            --force: Force shutdown (cancel pending requests)" << std::endl
                  << std::endl;
        std::cout << "    Client: " << argv[0]
                  << " client <name> <client_type> <num_clients> [capacity_exponent]" << std::endl;
        std::cout << "            client_type: 'normal' or 'thrust' (required)" << std::endl;
        std::cout << "            num_clients: number of concurrent clients to launch (default: 1)"
                  << std::endl;
        std::cout << "            capacity_exponent: same as server (default: 25)" << std::endl;
        std::cout << "            Keys to insert per client = 2^capacity_exponent * "
                  << TARGET_LOAD_FACTOR << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    std::string name = argv[2];

    if (mode == "server") {
        int capacityExponent = 25;
        bool forceShutdown = false;

        for (int i = 3; i < argc; i++) {
            std::string arg = argv[i];
            if (arg == "--force") {
                forceShutdown = true;
            } else {
                capacityExponent = std::stoi(arg);
            }
        }

        size_t capacity = 1ULL << capacityExponent;
        runServer(name, capacity, forceShutdown);
    } else if (mode == "client") {
        if (argc < 4) {
            std::cerr << "Error: client_type is required for client mode" << std::endl;
            std::cerr << "Usage: " << argv[0]
                      << " client <name> <client_type> <num_clients> [capacity_exponent]"
                      << std::endl;
            return 1;
        }

        std::string clientType = argv[3];
        if (clientType != "normal" && clientType != "thrust") {
            std::cerr << "Error: client_type must be 'normal' or 'thrust', got: " << clientType
                      << std::endl;
            return 1;
        }

        int numClients = (argc > 4) ? std::stoi(argv[4]) : 1;
        int capacityExponent = (argc > 5) ? std::stoi(argv[5]) : 25;
        size_t numKeys = (1ULL << capacityExponent) * TARGET_LOAD_FACTOR;

        auto clientFunc = (clientType == "normal") ? runClient : runClientThrust;

        std::vector<std::thread> clientThreads;
        for (int i = 0; i < numClients; i++) {
            clientThreads.emplace_back(clientFunc, name, i, numKeys);
        }

        for (auto& thread : clientThreads) {
            thread.join();
        }
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        return 1;
    }

    return 0;
}