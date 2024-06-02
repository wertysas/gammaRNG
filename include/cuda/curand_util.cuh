//
// Created by Johan Ericsson on 2024-03-13.
//

#ifndef CUGAMMA_CURAND_UTIL_CUH
#define CUGAMMA_CURAND_UTIL_CUH

#include <vector>
#include <type_traits>
#include <curand_kernel.h>

// RNG init kernel
__global__ void initRNG(curandState *const rngStates, const unsigned int seed);


// Templated version for cuRAND's uniform functions
template<typename RealType>
inline __device__ RealType uniform(curandState &state) {
    RealType x;
    if constexpr (std::is_same_v<RealType, float>) {
        return curand_uniform(&state);
        // x = curand_uniform(&state);
        // if (x < 1.0f) {
        //     return x;
        // } else {
        //     return uniform<RealType>(state);
        // }
    }
    if constexpr (std::is_same_v<RealType, double>) {
        return curand_uniform_double(&state);
    }
    return 0;
}

template<typename RealType>
inline __device__ RealType normal(curandState &state) {
    if constexpr (std::is_same_v<RealType, float>) {
        return curand_normal(&state);
    }
    if constexpr (std::is_same_v<RealType, double>) {
        return curand_normal_double(&state);
    }
    return 0;
}



template <typename RealType>
inline __device__ RealType log(RealType x) {
    if constexpr (std::is_same_v<RealType, float>) {
        return logf(x);
    }
    if constexpr (std::is_same_v<RealType, double>) {
        return log(x);
    }
}


template<typename RealType>
inline __device__ RealType rsqrt(RealType x) {
    if constexpr (std::is_same_v<RealType, float>) {
        return rsqrtf(9.0 * x);
    }
    if constexpr (std::is_same_v<RealType, double>) {
        return rsqrt(9.0 * x);
    }
    return 0;
}

template<typename RealType>
inline __device__ RealType invsqrt(RealType x) {
    if constexpr (std::is_same_v<RealType, float>) {
        return 1.0f/sqrt(x);
    }
    if constexpr (std::is_same_v<RealType, double>) {
        return 1.0/sqrt(x);
    }
    return 0;
}

// }
//
// namespace cug = cugamma;




#endif //CUGAMMA_CURAND_UTIL_CUH
