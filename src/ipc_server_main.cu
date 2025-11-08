#include <chrono>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>
#include "benchmark_common.cuh"
#include "CuckooFilterIPC.cuh"

using Config = CuckooConfig<uint32_t, 16, 500, 128, 16, XorHashStrategy>;
static constexpr char SERVER_NAME[] = "benchmark_server";

CuckooFilterIPCServer<Config>* g_server = nullptr;

void handleSignal(int signal) {
    (void)signal;

    if (g_server) {
        g_server->stop();
    }
    exit(0);
}

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <capacity>" << std::endl;
        return 1;
    }

    size_t capacity = std::stoul(argv[1]);

    signal(SIGTERM, handleSignal);
    signal(SIGINT, handleSignal);

    try {
        // Unlink any old shared memory just in case
        shm_unlink(("/cuckoo_filter_" + std::string(SERVER_NAME)).c_str());

        CuckooFilterIPCServer<Config> server(SERVER_NAME, capacity);
        g_server = &server;
        server.start();

        while (true) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }

    } catch (const std::exception& e) {
        std::cerr << "Server failed to start: " << e.what() << std::endl;
        return 1;
    }
}