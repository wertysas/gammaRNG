/*
 * Benchmark Code
 */

#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>


int main(int argc, const char **argv) {
    const int N = 385875968;

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


    std::random_device random_device;
    std::mt19937 mt(random_device());
    std::gamma_distribution<float> gamma_generator(10.0f);
    int iterations = 10;
    uint step_size = 1<<24;
    for (size_t n=1<<22; n < 1<<28; n+=step_size) {
        ss << n;
        std::vector<float> generated_data(n);
        for (int i = 0; i < iterations + 1; i++) {
            auto t = std::chrono::steady_clock::now();
            for (auto &e: generated_data) {
                e = gamma_generator(mt);
            }
            auto cpu_time = std::chrono::steady_clock::now() - t;
            ss << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(cpu_time).count();
        }
        ss << "\n";
    }
    std::cout << ss.str() << std::endl;

    return 0;
}
