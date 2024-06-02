//
// Created by Johan Ericsson on 2024-03-20.
//


#include <variant>
#include <map>
#include "gamma_kernel_test.cuh"
#include "util.hpp"


using TestMap = std::map<std::string, std::unique_ptr<TestBase>>;

int main () {
    size_t num_rands = 1000000;
    TestMap tests;
    tests["MT"] = std::make_unique<GammaKernelTest<float, marsaglia_tsang<float>>>(num_rands, "Marsaglia-Tsang");
    tests["GC"] = std::make_unique<GammaKernelTest<float, GC<float>>>(num_rands, "Ahrens-Dieter (GC)");
    tests["GA"] = std::make_unique<GammaKernelTest<float, cheng1977<float>>>(num_rands, "Cheng (GA)");
    tests["GKM3"] = std::make_unique<GammaKernelTest<float, GKM3<float>>>(num_rands, "Cheng-Feast (GKM3)");
    tests["XG"] = std::make_unique<GammaKernelTest<float, best1978<float>>>(num_rands, "Best (XG)");
//     auto test = tests.find("marsaglia_tsang2000");
//     std::visit(TestVisitor{}, test->second);
    // auto test = GammaKernelTest<float, marsaglia_tsang2000<float>>(num_rands, "marsaglia_tsang2000");
    //auto test = GammaKernelTest<float, marsaglia_tsangA<float>>(num_rands, "marsaglia_tsangA");
    //auto test = GammaKernelTest<float, cheng1977a<float>>(num_rands, "cheng1977", 1.5);
    // auto test = GammaKernelTest<float, GKM1<float>>(num_rands, "GMK1", 1.5);
    TestVisitor visitor{};
    for (const auto& test: tests)
        test.second->accept(visitor);

    return 0;

}
