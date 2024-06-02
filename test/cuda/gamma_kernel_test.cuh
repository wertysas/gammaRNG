//
// Created by Johan Ericsson on 2024-03-20.
//

#ifndef CUGAMMA_GAMMAKERNELTEST_CUH
#define CUGAMMA_GAMMAKERNELTEST_CUH

#include <iostream>
#include <utility>
#include "gamma_generators.cuh"
#include "cuda_error_helpers.h"
#include "curand_util.cuh"
#include "util.hpp"


struct TestVisitor {
    template<typename Test>
    void visit(Test& test) {
        test();
    }
};

class TestBase {
public:
    virtual ~TestBase() = default;
    virtual void accept(TestVisitor& visitor) = 0;

};

template<typename RealType, auto GammaKernel>
class GammaKernelTest : public TestBase {
public:
    GammaKernelTest(unsigned int num_rands, const char *name, unsigned int seed = 123)
            : nrands(num_rands),
              seed(seed),
              name(name) {};

    void operator()(RealType alpha=1.0001) {
        set_up();
        gamma_pt_strided<RealType, GammaKernel><<<grid, block>>>(d_gammas, d_rng_states, nrands, alpha);

        // Copy memory
        std::vector<RealType> gammas;
        try {
            gammas.resize(nrands);
        } catch (const std::bad_alloc &e) {
            std::cerr << "Vector Allocation failed: " << e.what() << '\n';
        }
        check_cuda_result(cudaDeviceSynchronize(), "Failed to synchronize with device: ");
        cuda_result = cudaMemcpy(gammas.data(), d_gammas, nrands * sizeof(RealType),
                                 cudaMemcpyDeviceToHost);
        check_cuda_result(cuda_result, "Failed to copy gamma array from device to host: ");

        for (int i=0; i<100; i++) {
            std::cout << gammas[i] << std::endl;
        }
        std::cout << "Gammas size:" << gammas.size() << std::endl;
        write_binary(gammas, std::string(CUGAMMA_TEST_OUTPUT_DIR) + "/gammas"+"#"+name+"#"+".bin");
        tear_down();
    }

    void accept(TestVisitor& visitor) override {
        visitor.visit(*this);
    }

private:
    unsigned int seed;
    unsigned int nrands;
    unsigned int device=0;
    dim3 block;
    dim3 grid;
    uint thread_block_size=64;
    std::string name;
    RealType *d_gammas = nullptr;
    curandState *d_rng_states = nullptr;
    cudaError_t cuda_result = cudaSuccess;
    cudaDeviceProp device_properties;


    void set_up() {

        // Get device properties
        cuda_result = cudaGetDeviceProperties(&device_properties, device);
        check_cuda_result(cuda_result, "Could not get device properties: ");

        // Attach to GPU
        cuda_result = cudaSetDevice(device);
        check_cuda_result(cuda_result, "Could not set CUDA device: ");


        // Use persistent threads for generating random numbers
        int max_active_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, gamma_pt_strided<RealType, GammaKernel>, thread_block_size, device);
        block.x = thread_block_size;
        grid.x = max_active_blocks*device_properties.multiProcessorCount;
        // verifies kernel dims are supported by initRNG and device
        verify_kernel_dims(block, grid, device_properties, &initRNG);

        // RNG states allocation on device
        cuda_result = cudaMalloc(&d_rng_states, grid.x * block.x * sizeof(curandState));
        check_cuda_result(cuda_result, "Could not allocate RNG states on device: ");
        initRNG<<<grid, block>>>(d_rng_states, seed);
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

    void write_output() {

    }


};


#endif //CUGAMMA_GAMMAKERNELTEST_CUH
