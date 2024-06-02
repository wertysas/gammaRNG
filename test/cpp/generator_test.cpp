/*
 * Benchmark Code
 */

#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include "util.hpp"
#include "gamma_generators.hpp"

using RealType =  double;

int main(int argc, const char **argv) {
    std::cout << "Cuda Gamma Generation Benchmark\n";
    std::cout << "==========================\n\n";

    const int N =  1000000;

    // STL reference timing
    std::vector<RealType> generated_data(N);
    //std::random_device random_device();
    std::mt19937 mt{};
    // auto lambda = [](auto& generator) -> decltype(auto) {
    //     return gamma_marsaglia_tsangd(generator, 1.0);
    // };

    GammaDistribution<RealType, marsaglia_tsang<RealType, decltype(mt)>> marsaglia_generator(7.32);
    GammaDistribution<RealType, cheng1977<RealType, decltype(mt)>> cheng_generator(7.32);
    GammaDistribution<RealType, best1978<RealType, decltype(mt)>> best_generator(7.32);
    auto std_gamma_generator = std::gamma_distribution<RealType>(7.32);


    auto t = std::chrono::steady_clock::now();
    for (auto &e: generated_data) {
        e = std_gamma_generator(mt);
    }
    auto cpu_time = std::chrono::steady_clock::now() - t;
    std::cout << "Generated " << N << " random numbers using" << typeid(std_gamma_generator).name() << "in: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(cpu_time).count() << " milliseconds." << std::endl;


    t = std::chrono::steady_clock::now();
    for (auto &e: generated_data) {
        e = marsaglia_generator(mt);
    }
    cpu_time = std::chrono::steady_clock::now() - t;
    std::cout << "Generated " << N << " random numbers using" << typeid(marsaglia_generator).name() << "in: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(cpu_time).count() << " milliseconds." << std::endl;

    t = std::chrono::steady_clock::now();
    for (auto &e: generated_data) {
        e = cheng_generator(mt);
    }
    cpu_time = std::chrono::steady_clock::now() - t;
    std::cout << "Generated " << N << " random numbers using" << typeid(cheng_generator).name() << "in: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(cpu_time).count() << " milliseconds." << std::endl;

    t = std::chrono::steady_clock::now();
    for (auto &e: generated_data) {
        e = best_generator(mt);
    }
    cpu_time = std::chrono::steady_clock::now() - t;
    std::cout << "Generated " << N << " random numbers using" << typeid(cheng_generator).name() << "in: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(cpu_time).count() << " milliseconds." << std::endl;

    return 0;
}
