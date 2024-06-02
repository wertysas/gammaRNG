//
// Created by Johan Ericsson on 2024-03-13.
//



#include "../../include/cuda/curand_util.cuh"

// RNG init kernel
__global__ void initRNG(curandState *const rngStates, const unsigned int seed) {
    // Determine thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialise the RNG
    curand_init(seed, tid, 0, &rngStates[tid]);
}
