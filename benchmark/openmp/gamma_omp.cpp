//
// Created by Johan Ericsson on 2024-03-15.
//

#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <omp.h>


int main(int argc, char *argv[]) {


    std::stringstream ss;
    // add test header
    ss.str("");
    ss.clear();
    ss << "time: ";
    auto time = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(time);
    ss << std::put_time(std::localtime(&in_time_t), "%Y-%m-%d %X") << "\n";
    ss << "device type: " << "CPU";
    ss << "kernel: " << "std::gamma_distribution" << "\n";
    ss << "measurement unit: milliseconds\n";
    ss << std::string(80, '-') << "\n";


    int iterations = 10;
    uint step_size = 1<<24;
    for (size_t n=1<<22; n < 1<<28; n+=step_size) {
        ss << n;
        std::vector<float> generated_data(n);
        const int seed = 1234;
        for (int k = 0; k < iterations + 1; k++) {
            auto t = std::chrono::steady_clock::now();
#pragma omp parallel default(none), shared(n, std::cout, generated_data)
            {
                int id = omp_get_thread_num();
                std::mt19937 mt(id * 979 + seed);
                std::gamma_distribution<float> gamma_generator(10.0f);
#pragma omp for
                for (size_t i = 0; i<n; i++) {
                    generated_data[i] = gamma_generator(mt);
                } // end of omp for
            } // end of parallel region
            auto cpu_time = std::chrono::steady_clock::now() - t;
            ss << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(cpu_time).count();
        }
        ss << "\n";
    }
    std::cout << ss.str() << std::endl;

    return 0;
}