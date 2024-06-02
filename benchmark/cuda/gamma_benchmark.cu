//
// Created by Johan Ericsson on 2024-04-07.
//

#include "gamma_benchmark.cuh"
#include "cuda_runtime.h"



int main() {

    GammaBenchmark<float, marsaglia_tsang<float>> gmt{"Marsaglia-Tsang"};
    GammaBenchmark<float, GC<float>> GC{"Ahrens-Dieter (GC)"};
    GammaBenchmark<float, cheng1977<float>> cheng{"Cheng (GA)"};
    GammaBenchmark<float, GKM1<float>> GKM1{"Cheng-Feast (GKM1)"};
    GammaBenchmark<float, GKM2<float>> GKM2{"Cheng-Feast (GKM2)"};
    GammaBenchmark<float, GKM3<float>> GKM3{"Cheng-Feast (GKM3)"};
    GammaBenchmark<float, best1978<float>> best{"Best (XG)"};
    GammaBenchmark<float, normal_kernel<float>> normal_bench{"cuRAND normal (device API)"};
    gmt(1.0001);
    gmt(2.0);
    gmt(4.0);
    gmt(10.0);
    GC(1.0001);
    GC(2.0);
    GC(4.0);
    GC(10.0);
    cheng(1.0001);
    cheng(2.0);
    cheng(4.0);
    cheng(10.0);
    // GKM3(1.0001);
    // GKM3(2.0);
    // GKM3(4.0);
    // GKM3(10.0);
    best(1.0001);
    best(2.0);
    best(4.0);
    best(10.0);
    normal_bench();

    return 0;
}