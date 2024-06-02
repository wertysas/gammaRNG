//
// Created by Johan Ericsson on 2024-04-07.
//

#ifndef CUGAMMA_GAMMABENCHMARK_CUH
#define CUGAMMA_GAMMABENCHMARK_CUH

#include <iostream>
#include <sstream>
#include <chrono>
#include <utility>
#include <iomanip>
#include "gamma_generators.cuh"
#include "cuda_error_helpers.h"
#include "curand_util.cuh"
#include "util.hpp"


template<typename RealType, auto GammaKernel>
class GammaBenchmark {
public:
    GammaBenchmark(const char* name) : name(name) {
        // Get device properties
        cuda_result = cudaGetDeviceProperties(&deviceProperties, 0);
        check_cuda_result(cuda_result, "Could not get device properties: ");

        // Attach to GPU
        cuda_result = cudaSetDevice(0);
        check_cuda_result(cuda_result, "Could not set CUDA device: ");

    }
    void operator()(RealType alpha=1.0) {
        add_header(alpha);
        uint step_size = 1<<24;
        for (size_t n=1<<22; n < 1<<28; n+=step_size) {
        //for (size_t n=1<<20; n < 1L<<32; n*=2) {
            benchmark(n, alpha);
        }
        std::cout << ss.str() << std::endl;
    }

private:
    const std::string name;
    std::stringstream ss;
    cudaError_t cuda_result;
    cudaDeviceProp deviceProperties;
    uint device = 0;
    uint thread_block_size = 64;
    RealType* d_gammas = nullptr;
    curandState* d_rng_states = nullptr;
    dim3 block, grid;

    void benchmark(size_t nrands,  RealType alpha=1.0, int iterations = 10) {
        set_up(nrands);

        float time_ms;
        cudaEvent_t  start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        ss << nrands;
        for (int i=0; i<iterations+1; i++) {
            cudaEventRecord(start);
            //marsaglia_tsang_pt_strided<<<grid, block>>>(d_gammas, d_rng_states, nrands, alpha);
            gamma_pt_strided<RealType, GammaKernel><<<grid, block>>>(d_gammas, d_rng_states, nrands, alpha);
            cudaEventRecord(stop);
            check_cuda_result(cudaDeviceSynchronize(), "Failed to synchronize with device: ");
            cudaEventElapsedTime(&time_ms, start, stop) ;
            ss << ", " << time_ms;
        }
        ss << "\n";
        tear_down();
    }

    void set_up(size_t nrands) {

        // Use persistent threads for generating random numbers
        int max_active_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, gamma_pt_strided<RealType, GammaKernel>, thread_block_size, device);
        block.x = thread_block_size;
        grid.x = max_active_blocks*deviceProperties.multiProcessorCount;
        // verifies kernel dims are supported by initRNG and device
        verify_kernel_dims(block, grid, deviceProperties, &initRNG);

        // RNG states allocation on device
        cuda_result = cudaMalloc(&d_rng_states, grid.x * block.x * sizeof(curandState));
        check_cuda_result(cuda_result, "Could not allocate RNG states on device: ");
        initRNG<<<grid, block>>>(d_rng_states, 0);
        check_cuda_result(cudaDeviceSynchronize(), "Failed to synchronize with device: ");

        // Random number arrays
        cuda_result = cudaMalloc(&d_gammas,  nrands * sizeof(RealType));
        check_cuda_result(cuda_result, "Could not allocate array for rng output on device: ");
    }

    void tear_down() {
        // clean up device memory
        check_cuda_result(cudaFree(d_gammas));
        check_cuda_result(cudaFree(d_rng_states));
    }

    void add_header(RealType alpha) {
        // add test header
        ss.str("");
        ss.clear();
        ss << "time: ";
        auto time = std::chrono::system_clock::now();
        auto in_time_t = std::chrono::system_clock::to_time_t(time);
        ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X") << "\n";

        ss << "device type: " << deviceProperties.name << "\n";
        ss << "kernel: " << name << "\n";
        ss << "alpha: " << alpha << "\n";
        ss << "measurement unit: milliseconds\n";
        ss << std::string(80, '-') << "\n";
    }
};


#endif //CUGAMMA_GAMMABENCHMARK_CUH
