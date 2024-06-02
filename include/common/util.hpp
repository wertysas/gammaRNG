//
// Created by Johan Ericsson on 2024-03-14.
//

#ifndef CUGAMMA_UTIL_HPP
#define CUGAMMA_UTIL_HPP

#include <iostream>
#include <fstream>

template<typename T>
void write_binary(std::vector<T> &vector, const std::string& file_name = "output.bin") {
    std::ofstream os{file_name, std::ios::binary};  // open file
    if (os.good()) {    // if stream is OK we can write to file
        os.write(reinterpret_cast<char const *>(vector.data()), sizeof(T) * vector.size());
    } else {
        std::cout << "Failed to write to file: " << file_name << " check that the specified path exists" << std::endl;
    }
}



#endif //CUGAMMA_UTIL_HPP
