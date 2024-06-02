//
// Created by Johan Ericsson on 3/16/24.
//

#ifndef CUGAMMA_GAMMA_GENERATORS_HPP
#define CUGAMMA_GAMMA_GENERATORS_HPP

#include <random>
#include <cmath>

template<typename RealType>
RealType invsqrt(RealType x) {
    return RealType(1)/std::sqrt(x);
}


template<typename RealType, typename UniformRandomBitGenerator>
RealType marsaglia_tsang(UniformRandomBitGenerator& generator, RealType alpha) {
    const RealType d = alpha - 1.0 / 3.0;
    const RealType c = invsqrt<RealType>(9.0 * d);
    auto uniform = std::uniform_real_distribution<RealType>();
    auto normal = std::normal_distribution<RealType>();
    RealType Z, U, V, v;
    bool accept;
    do {
        Z = normal(generator);
        U = uniform(generator);
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
template<typename RealType, typename UniformRandomBitGenerator>
  RealType cheng1977(UniformRandomBitGenerator& generator, const RealType alpha) {
    const RealType a = invsqrt(RealType(2) * alpha - RealType(1));
    const RealType b = alpha - log<RealType>(4);
    const RealType c = alpha + sqrt(RealType(2) * alpha - RealType(1));
    auto uniform = std::uniform_real_distribution<RealType>();
    auto normal = std::normal_distribution<RealType>();
    RealType U1, U2, V, X;
    bool accept;
    do {
        U1 = uniform(generator);
        U2 = uniform(generator);
        V = a * log(U1 / (RealType(1) - U1));
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

// // Algorithm GKM1 most efficient for alpha in (1,3)
// template<typename RealType, typename CurandState=curandState>
//   RealType GKM1(CurandState &rng_state, const RealType alpha) {
//     const RealType a = alpha - RealType(1.);
//     const RealType b = (alpha - 1 / (RealType(6) * alpha)) / a;
//     const RealType c = RealType(2.)/a;
//     const RealType d = c+ RealType(2.);
//     RealType U1, U2, W;
//     bool accept;
//     do {
//         U1 = uniform<RealType>(rng_state);
//         U2 = uniform<RealType>(rng_state);
//         W = b*U1/U2;
//         accept = c*log(U2) - log(W) + W < 0;
//     } while (!accept);
//     return a*W;
// }
//
// // Algorithm GKM2 most efficient for alpha in (3, infty)
// template<typename RealType, typename CurandState=curandState>
//   RealType GKM2(CurandState &rng_state, const RealType alpha) {
//     const RealType a = alpha - RealType(1.);
//     const RealType b = (alpha - 1 / (RealType(6) * alpha)) / a;
//     const RealType c = RealType(2.)/a;
//     const RealType d = c+ RealType(2.);
//     const RealType finv = invsqrt(alpha);
//     RealType U, U1, U2, W;
//     bool accept;
//     do {
//         U = uniform<RealType>(rng_state);
//         U1 = uniform<RealType>(rng_state);
//         U2 = U1 + finv*(1.-1.86*U);
//         W = b*U1/U2;
//         // if (c*U2-d + W + 1./W <= 0) {
//         //     return a*W;
//         // }
//         accept = (0 < U2 && U2 < 1 && c*log(U2)-log(W)+W< 0);
//     } while (!accept);
//     return a*W;
// }
//
//
// // Algorithm GKM3 as suggested by the authors dispatches to either
// // GKM1 or GKM2 based on alpha value, with cutoff at 2.5.
// template<typename RealType, typename CurandState=curandState>
//   RealType GKM3(CurandState &rng_state, const RealType alpha) {
//     const RealType cutoff_alpha = 2.5;
//     if (alpha < cutoff_alpha) {
//         return GKM1(rng_state, alpha);
//     }
//     return GKM2(rng_state, alpha);
// }




////////////////////////////////////////////////////////////////////////////////
// Best (XG)
/// ////////////////////////////////////////////////////////////////////////////////
template<typename RealType, typename UniformRandomBitGenerator>
RealType best1978(UniformRandomBitGenerator& generator, const RealType alpha) {
    const RealType b = alpha - RealType(1);
    const RealType c = 3 * alpha - RealType(3) / RealType(4);

    auto uniform = std::uniform_real_distribution<RealType>();
    RealType X, Y, Z, U, V, W;
    bool accept;
    do {
        U = uniform(generator);
        V = uniform(generator);
        W = U * (RealType(1) - U);
        Y = sqrt(c / W) * (U - RealType(0.5));
        X = b + Y;
        Z = RealType(64)*W*W*W*V*V;
        accept = log(Z) <= RealType(2)*(b*log(X/b)-Y);
    } while (!accept);
    return X;
}




////////////////////////////////////////////////////////////////////////////////
// GammaDistribution Functor Wrapper
////////////////////////////////////////////////////////////////////////////////

template<typename RealType, auto GammaGenerator>
class GammaDistribution {
    public:
        GammaDistribution(RealType alpha) : alpha(alpha) {}
        RealType operator()(auto& generator) {
            return GammaGenerator(generator, alpha);
        }
    private:
        RealType alpha;

    };


#endif //CUGAMMA_GAMMA_GENERATORS_HPP
