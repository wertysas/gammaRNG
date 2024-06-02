
include_directories(../Common)


add_executable(gamma_stl cpp/cpu_benchmark.cpp)


# OpenMP Benchmarks
if(OpenMP_CXX_FOUND)
    add_executable(gamma_omp openmp/gamma_omp.cpp
    )
    target_include_directories(gamma_omp PRIVATE ${OPENMP_INCLUDE_DIR})
    target_link_libraries(gamma_omp OpenMP::OpenMP_CXX)
endif()



# CUDA Benchmarks
if(CUDAToolkit_FOUND)
    add_executable(gamma_generator
            ${GAMMARNG_SOURCE_DIR}/cuda/curand_util.cu
    )
    target_include_directories(gamma_generator PRIVATE ${CMAKE_SOURCE_DIR}/include/cuda)
    target_link_libraries(gamma_generator CUDA::cudart)


    add_executable(gamma_kernel_benchmark
            cuda/gamma_benchmark.cu
            ${GAMMARNG_SOURCE_DIR}/cuda/curand_util.cu
    )
    target_include_directories(gamma_kernel_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/include/cuda)
    target_link_libraries(gamma_kernel_benchmark CUDA::cudart)
endif()

