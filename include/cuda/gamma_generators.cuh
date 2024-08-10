//
// Created by Johan Ericsson on 2024-03-13.
//

#ifndef CUGAMMA_GAMMA_GENERATORS_CUH
#define CUGAMMA_GAMMA_GENERATORS_CUH

#include <math_constants.h>
#include <cooperative_groups.h>
#include "curand_util.cuh"




template<typename RealType>
__forceinline__ __device__  RealType marsaglia_tsang(curandState &rng_state, const RealType alpha) {
    const RealType d = alpha - RealType(1)/RealType(3);
    const RealType c = invsqrt<RealType>(9.0 * d);
    RealType Z, U, V, v;
    bool accept;
    do {
        Z = normal<RealType>(rng_state);
        U = uniform<RealType>(rng_state);
        v = (1.0 + c * Z);
        V = v * v * v;
        accept = ((V > 0.0) && log(U) < (0.5 * Z * Z + d- d*V + d*log(V)));
    } while (!accept);
    return d * V;
}


////////////////////////////////////////////////////////////////////////////////
// Cheng, R. C. H. “The Generation of Gamma Variables with Non-Integral Shape
// Parameter.” Journal of the Royal Statistical Society. Series C
// (Applied Statistics) 26, no. 1 (1977): 71–75. https://doi.org/10.2307/2346871.
////////////////////////////////////////////////////////////////////////////////

// Algorithm (GA) corresponding to steps 1-3 from the paper. We don't use the
// squeeze corresponding to step 3' from the paper. Most suitable for alpha > 1
template<typename RealType, typename CurandState=curandState>
__forceinline__ __device__  RealType cheng1977(CurandState &rng_state, const RealType alpha) {
    const RealType a = invsqrt(RealType(2) * alpha - RealType(1));
    const RealType b = alpha - log<RealType>(4);
    const RealType c = alpha + sqrt(RealType(2) * alpha - RealType(1));
    RealType U1, U2, V, X;
    bool accept;
    do {
        U1 = uniform<RealType>(rng_state);
        U2 = uniform<RealType>(rng_state);
        V = a * log(U1 / (RealType(1.0) - U1));
        X = alpha * exp(V);
        accept = b + c*V - X >= log(U1*U1*U2);
    } while (!accept);
    return X;
}


////////////////////////////////////////////////////////////////////////////////
// Cheng, R. C. H., and G. M. Feast. “Some Simple Gamma Variate Generators.”
// Journal of the Royal Statistical Society. Series C (Applied Statistics) 28,
// no. 3 (1979): 290–95. https://doi.org/10.2307/2347200.
////////////////////////////////////////////////////////////////////////////////

// Algorithm GKM1 most efficient for alpha in (1,3)
template<typename RealType, typename CurandState=curandState>
__forceinline__ __device__  RealType GKM1(CurandState &rng_state, const RealType alpha) {
    const RealType a = alpha - RealType(1.);
    const RealType b = (alpha - 1 / (RealType(6) * alpha)) / a;
    const RealType c = RealType(2.)/a;
    const RealType d = c+ RealType(2.);
    RealType U1, U2, W;
    bool accept;
    do {
        U1 = uniform<RealType>(rng_state);
        U2 = uniform<RealType>(rng_state);
        W = b*U1/U2;
        accept = c*log(U2) - log(W) + W < 0;
    } while (!accept);
    return a*W;
}

// Algorithm GKM2 most efficient for alpha in (3, infty)
template<typename RealType, typename CurandState=curandState>
__forceinline__ __device__  RealType GKM2(CurandState &rng_state, const RealType alpha) {
    const RealType a = alpha - RealType(1.);
    const RealType b = (alpha - 1 / (RealType(6) * alpha)) / a;
    const RealType c = RealType(2.)/a;
    const RealType d = c+ RealType(2.);
    const RealType finv = invsqrt(alpha);
    RealType U, U1, U2, W;
    bool accept;
    do {
        U = uniform<RealType>(rng_state);
        U1 = uniform<RealType>(rng_state);
        U2 = U1 + finv*(1.-1.86*U);
        W = b*U1/U2;
        // if (c*U2-d + W + 1./W <= 0) {
        //     return a*W;
        // }
        accept = (0 < U2 && U2 < 1 && c*log(U2)-log(W)+W< 0);
    } while (!accept);
    return a*W;
}


// Algorithm GKM3 as suggested by the authors dispatches to either
// GKM1 or GKM2 based on alpha value, with cutoff at 2.5.
template<typename RealType, typename CurandState=curandState>
__forceinline__ __device__  RealType GKM3(CurandState &rng_state, const RealType alpha) {
    const RealType cutoff_alpha = 2.5;
    if (alpha < cutoff_alpha) {
        return GKM1(rng_state, alpha);
    }
    return GKM2(rng_state, alpha);
}




////////////////////////////////////////////////////////////////////////////////
// D. J. Best. “Letters to the Editors”. eng. In: Journal of the Royal
// Statistical Society: Series C (Applied Statistics) 27.2 (1978), pp. 181–182.
////////////////////////////////////////////////////////////////////////////////

//
template<typename RealType, typename CurandState=curandState>
__forceinline__ __device__  RealType best1978(CurandState &rng_state, const RealType alpha) {
    const RealType b = alpha - RealType(1.0);
    const RealType c = 3 * alpha - RealType(3.0) / RealType(4.0);
    RealType X, Y, Z, U, V, W;
    bool accept;
    do {
        U = uniform<RealType>(rng_state);
        V = uniform<RealType>(rng_state);
        W = U * (1 - U);
        Y = sqrt(c / W) * (U - RealType(0.5));
        X = b + Y;
        Z = 64.*W*W*W*V*V;
        accept = log(Z) <= 2.*(b*log(X/b)-Y);
    } while (!accept);
    return X;
}


////////////////////////////////////////////////////////////////////////////////
// Joachim H. Ahrens and Ulrich Dieter. “Computer methods for sampling from
// gamma, beta, poisson and bionomial distributions”. In: Computing 12.3 (1974),
// pp. 223–246.
////////////////////////////////////////////////////////////////////////////////

// Algorithm GC
template<typename RealType, typename CurandState=curandState>
__forceinline__ __device__  RealType GC(CurandState &rng_state, const RealType alpha) {
    const RealType b = alpha-1.0f;
    const RealType A = alpha + b;
    const RealType s = sqrt(A);
    RealType u, ustar, t, x;
    bool accept;
    do {
        u = uniform<RealType>(rng_state);
        ustar = uniform<RealType>(rng_state);
        t = s*tan(CUDART_PI_F*(u-0.5f));
        x = b+t;
        accept = ustar <= exp(b*log(x/b) - t + log(1+t*t/A));
    } while (!accept);
    return x;

}




////////////////////////////////////////////////////////////////////////////////
// Other Kernels
////////////////////////////////////////////////////////////////////////////////

// Normal kernels used for reference purpose in benchmark script.
template<typename RealType>
__forceinline__ __device__  RealType normal_kernel(curandState &rng_state, RealType alpha) {
    return normal<RealType>(rng_state);
}


////////////////////////////////////////////////////////////////////////////////
// Global Kernels
////////////////////////////////////////////////////////////////////////////////


// This function assumes each thread has its own curandState and generates multiple random numbers/thread (see usage in benchmark class for example on how to use)
template<typename RealType, auto GammaKernel>
__global__ void
gamma_pt_strided(RealType *gammas, curandState *rng_states, const size_t num_rands, const RealType alpha = 1.0) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState rng_state = rng_states[tid];
    uint nthreads = blockDim.x * gridDim.x;

    // main loop body
    uint main_iter_per_thread = num_rands / nthreads;
    for (uint i = 0; i < main_iter_per_thread; i++) {
        uint idx = nthreads * i + tid;
        gammas[idx] = GammaKernel(rng_state, alpha);
    }
    // tail loop
    uint tail_rands = num_rands - main_iter_per_thread * nthreads;
    if (tid < tail_rands) {
        gammas[main_iter_per_thread * nthreads + tid] = GammaKernel(rng_state, alpha);
    }
    rng_states[tid] = rng_state;
}

// This function assumes each thread has its own curandState and generates multiple random numbers/thread
template<typename RealType, auto GammaKernel>
__global__ void gamma_pt(RealType *gammas, curandState *rng_states, const uint num_rands, RealType alpha = 1.0) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState rng_state = rng_states[tid];
    uint nthreads = blockDim.x * gridDim.x;

    // main loop body
    uint main_iter_per_thread = num_rands / nthreads;
    uint idx = tid * main_iter_per_thread;
    for (uint i = 0; i < main_iter_per_thread; i++) {
        gammas[idx + i] = GammaKernel(rng_state, alpha);
    }
    // tail loop
    uint tail_rands = num_rands - main_iter_per_thread * nthreads;
    if (tid < tail_rands) {
        gammas[main_iter_per_thread * nthreads + tid] = GammaKernel(rng_state, alpha);
    }
    rng_states[tid] = rng_state;
}

// This function assumes each thread has its own curandState (NOT RECOMMENDED)
template<typename RealType, auto GammaKernel>
__global__ void gamma(RealType *gammas, curandState *rng_states, const uint size, RealType alpha = 1.0) {
    unsigned int blockid = blockIdx.x;
    unsigned int tid = blockid * blockDim.x + threadIdx.x;
    if (tid >= size) return;

    curandState rng_state = rng_states[tid];
    gammas[tid] = GammaKernel(rng_state, alpha);
    rng_states[tid] = rng_state;
}


#endif //CUGAMMA_GAMMA_GENERATORS_CUH
