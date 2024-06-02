add_executable(gamma_cpp_test cpp/generator_test.cpp)
target_include_directories(gamma_cpp_test PRIVATE ${CXX_INCLUDE_DIR})

# OpenMP Tests
#if(OpenMP_CXX_FOUND)
    # add_executable(gamma_omp_test openmp/gamma_omp.cpp)
    # target_include_directories(gamma_omp_test PRIVATE ${OPENMP_INCLUDE_DIR})
    # target_include_directories(gamma_omp_test PRIVATE ${CXX_INCLUDE_DIR})
    # target_link_libraries(gamma_omp_test OpenMP::OpenMP_CXX)
#endif()



# CUDA Tests
if(CUDAToolkit_FOUND)
    add_executable(gamma_kernel_test
            cuda/gamma_kernel_test.cu
            cuda/gamma_kernel_test.cuh
            ${CUDA_SOURCE_DIR}/curand_util.cu)
    target_include_directories(gamma_kernel_test PRIVATE ${CMAKE_SOURCE_DIR}/include/cuda)
    target_link_libraries(gamma_kernel_test CUDA::cudart)
endif()

