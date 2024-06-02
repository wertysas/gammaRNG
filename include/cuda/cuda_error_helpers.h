#ifndef CUDA_ERROR_HELPERS_H

#include <string>
#include <stdexcept>
#include <cuda_runtime.h>


inline void check_cuda_result(cudaError_t cuda_result, const char* error_msg="CUDA Error String: ") {
    if (cuda_result != cudaSuccess) {
        std::string msg(error_msg);
        msg += cudaGetErrorString(cuda_result);
        throw std::runtime_error(msg);
    }
}

template <typename T>
__host__ void verify_kernel_dims(dim3 block, dim3 grid, cudaDeviceProp device_properties, T* kernel) {
    cudaFuncAttributes funcAttributes;
    cudaError_t cudaResult;
    cudaResult = cudaFuncGetAttributes(&funcAttributes, kernel);

    if (cudaResult != cudaSuccess) {
        std::string msg("Could not get function attributes");
        msg += "(kernel ";
        msg += typeid(T).name();
        msg += "): ";
        msg += cudaGetErrorString(cudaResult);
        throw std::runtime_error(msg);
    }
    
    // Check block size is supported by kernel
    if (block.x*block.y*block.z > static_cast<unsigned int>(funcAttributes.maxThreadsPerBlock)) {
        std::string msg("Block dimensions are too large for kernel: ");
        msg += typeid(T).name();
        throw std::runtime_error(msg);
    }
    
    // Check all block dimensions are supported by device
    if (block.x > static_cast<unsigned int>(device_properties.maxThreadsDim[0])) {
        throw std::runtime_error("Block x dimension is too large for device");
    }
    if (block.y > static_cast<unsigned int>(device_properties.maxThreadsDim[1])) {
        throw std::runtime_error("Block y dimension is too large for device");
    }
    if (block.z > static_cast<unsigned int>(device_properties.maxThreadsDim[2])) {
        throw std::runtime_error("Block z dimension is too large for device");
    }
    // Check grid dimensions are supported by device
    if (grid.x > static_cast<unsigned int>(device_properties.maxGridSize[0])) {
        throw std::runtime_error("grid x dimension is too large for device");
    }
    if (grid.y > static_cast<unsigned int>(device_properties.maxGridSize[1])) {
        throw std::runtime_error("grid y dimension is too large for device");
    }
    if (grid.z > static_cast<unsigned int>(device_properties.maxGridSize[2])) {
        throw std::runtime_error("grid z dimension is too large for device");
    }

}




#endif //CUDA_ERROR_HELPERS_H

