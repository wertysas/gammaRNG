cmake_minimum_required(VERSION 3.20)
project(GAMMARNG CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS  "-O3")

# For language servers
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

################################################################################
# Enable OpenMP if available on system
################################################################################
find_package(OpenMP)
if(OpenMP_CXX_FOUND) # Use OpenMP_FOUND for CMake < 3.9
    message(STATUS "OpenMP found.")
else()
    message(STATUS "OpenMP not found.")
endif()


################################################################################
# Enable CUDA if available on system
################################################################################
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.2") # Uncomment to specify custom cuda toolkit root directory
# set(ENV{CUDACXX} "/usr/local/cuda-12.2/bin/nvcc") # Uncomment to specify custom CUDA compiler path
find_package(CUDAToolkit) # defines imported targets for all CUDA libs
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA Toolkit found.")
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES 89)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_FLAGS "-O3")
else()
    message(STATUS "NO CUDA Toolkit found, proceeding without CUDA support.")
endif()


################################################################################
# Source and include directory variables
################################################################################
# Top level src/include
set(GAMMARNG_SOURCE_DIR "${CMAKE_SOURCE_DIR}/src" CACHE INTERNAL "Project source directory")
set(GAMMARNG_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include" CACHE INTERNAL "Project include directory")
# Cuda
set(CUDA_SOURCE_DIR "${GAMMARNG_SOURCE_DIR}/cuda" CACHE INTERNAL "CUDA source directory")
set(CUDA_INCLUDE_DIR "${GAMMARNG_INCLUDE_DIR}/cuda" CACHE INTERNAL "CUDA include directory")
# OpenMP
set(OPENMP_SOURCE_DIR "${GAMMARNG_SOURCE_DIR}/openmp" CACHE INTERNAL "OpenMP source directory")
set(OPENMP_INCLUDE_DIR "${GAMMARNG_INCLUDE_DIR}/openmp" CACHE INTERNAL "OpenMP include directory")
# CXX
set(CXX_SOURCE_DIR "${GAMMARNG_SOURCE_DIR}/cpp" CACHE INTERNAL "CXX source directory")
set(CXX_INCLUDE_DIR "${GAMMARNG_INCLUDE_DIR}/cpp" CACHE INTERNAL "CXX include directory")
message(STATUS "CXX_INCLUDE_DIR set to ${CXX_INCLUDE_DIR}")
# Common includes
include_directories("${GAMMARNG_INCLUDE_DIR}/common")

################################################################################
# Tests
################################################################################
# Default output dir for test, data can be changed by using cmake with the command
# cmake -DPROJECT_TEST_OUTPUT_DIR="/custom/output/path"
set(TEST_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/test/data" CACHE PATH "Path to the output directory for tests")
add_definitions(-DGAMMARNG_TEST_OUTPUT_DIR="${TEST_OUTPUT_DIR}")

add_subdirectory(test)
################################################################################
# Benchmarks
################################################################################
add_subdirectory(benchmark)
