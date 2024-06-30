# gammaRNG

This repo contains the CUDA implementation of five gamma generator kernels and benchmarking code for the M.Sc. thesis : [**Gamma Random Number Generation on GPUs using CUDA**](wertysas.github.io/documents/gamma_rng_on_gpus.pdf)

**The best performing kernels on all GPUs tested are ``cheng1977`` and ``marsaglia_tsang``**

More information about the the theory of RNGs and the gamma distribution, and further references can be found in the thesis.



## Gamma Generation Kernels

The following gamma generators were selected as potential candidates for effective generators on GPU architectures and implemented in CUDA ([gamma_generators.cuh](include/cuda/gamma_generators.cuh)):

``marsaglia_tsang`` - The generator (without squeeze step) from:
 - **George Marsaglia and Wai Wan Tsang**. A simple method for generating gamma variables. ACM Trans. Math. Softw. 26, 3 (Sept. 2000): 363–372. https://doi.org/10.1145/358407.358414 

``cheng1977`` - The generator (GA) from:
- **Cheng, R. C. H.** The Generation of Gamma Variables with Non-Integral Shape Parameter. Journal of the Royal Statistical Society. Series C (Applied Statistics) 26, no. 1 (1977): 71–75. https://doi.org/10.2307/2346871


## Build Instructions

The build system used is cmake and the target ``gamma_kernel_benchmark`` correspond to the benchmark executable. An example of how to build and execute the code, and analyze the benchmark output can be found in the google COLAB notebook:
[gammaRNG](https://colab.research.google.com/drive/1nUAEsqd1u1J5OVQPbihVZ8OKk_tsHEro?usp=sharing).


