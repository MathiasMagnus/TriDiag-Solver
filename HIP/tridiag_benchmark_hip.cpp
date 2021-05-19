#include "tridiag_benchmark_hip.hpp"

#include "tridiag_solver.hpp"

#include <algorithm>
#include <functional>
#include <iostream>
#include <type_traits>
#include <tuple>
#include <vector>
#include <utility>
#include <random>
#include <cmath>
#include <complex>

std::initializer_list<int> system_counts = { 1, 2, 3, 4 };
std::initializer_list<int> system_sizes = {
    127, 128, 129,
    255, 256, 257,
    500, 768, 1000,
    1023, 1024, 1025,
    1234,
    2047, 2048, 2049,
    4096, 4097,
    32768
};
constexpr size_t warmup_count = 5;               // Used to dispatch binaries, etc.
constexpr unsigned int batch_size = 10;          // Used to inflate runtime
constexpr benchmark::IterationCount trials = 10; // Used to improve statistics (minimum 1)

template<class T>
void run(benchmark::State& state,
         int size,
         int sys_count)
{
    // Generate data
    seed_type seed = 1;
    std::vector<T> dl = bench::get_random_data<T>(size, 0, 32768, seed++),
                   d  = bench::get_random_data<T>(size, 0, 32768, seed++),
                   du = bench::get_random_data<T>(size, 0, 32768, seed++),
                   b  = bench::get_random_data<T>(size*sys_count, 0, 32768, seed++);

    T* dev_dl; HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dev_dl), dl.size() * sizeof(T)));
    T* dev_d;  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dev_d ), d .size() * sizeof(T)));
    T* dev_du; HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dev_du), du.size() * sizeof(T)));
    T* dev_b;  HIP_CHECK(hipMalloc(reinterpret_cast<void**>(&dev_b ), b .size() * sizeof(T)));

    HIP_CHECK(hipMemcpy(dev_dl, dl.data(), dl.size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_d , d .data(), d .size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_du, du.data(), du.size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dev_b,  b .data(), b .size() * sizeof(T), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    tridiag_solver<T> tds{size, sys_count};
    HIP_CHECK(hipDeviceSynchronize());

    // Warm-up
    for(size_t i = 0; i < warmup_count; i++)
        tds.solve(dev_dl, dev_d, dev_du, dev_b);
    HIP_CHECK(hipDeviceSynchronize());

    for(auto _ : state)
    {
        auto start = std::chrono::high_resolution_clock::now();
        for(unsigned int i = 0; i < batch_size; i++) {
            tds.solve(dev_dl, dev_d, dev_du, dev_b);
            HIP_CHECK(hipDeviceSynchronize());
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds =
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
        state.SetIterationTime(elapsed_seconds.count());
    }
    state.SetBytesProcessed(state.iterations() * batch_size * (3/*du,d,dl*/+sys_count) * size * sizeof(T));
    state.SetItemsProcessed(state.iterations() * batch_size * (3/*du,d,dl*/+sys_count) * size);

    HIP_CHECK(hipFree(dev_dl));
    HIP_CHECK(hipFree(dev_d));
    HIP_CHECK(hipFree(dev_du));
    HIP_CHECK(hipFree(dev_b));
}

int main(int argc, char *argv[])
{
    // Parse argv
    benchmark::Initialize(&argc, argv);

    // HIP
    int device_id;
    hipDeviceProp_t device_props;
    HIP_CHECK(hipGetDevice(&device_id));
    HIP_CHECK(hipGetDeviceProperties(&device_props, device_id));
    std::cout << "[HIP] Device name: " << device_props.name << std::endl;
    HIP_CHECK(hipSetDevice(device_id));

    // Add benchmarks
    std::vector<benchmark::internal::Benchmark*> benchmarks;
    for (auto size : system_sizes)
        for (auto sys_count : system_counts)
        {
            bench::reg("solve_float", run<float>, size, sys_count);
            bench::reg("solve_double", run<double>, size, sys_count);
            bench::reg("solve_cuComplex", run<cuComplex>, size, sys_count);
            bench::reg("solve_cuDoubleComplex", run<cuDoubleComplex>, size, sys_count);
        }

    // Set manual timing amd iterations
    for (auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
        b->Iterations(trials);
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();

    return 0;
}