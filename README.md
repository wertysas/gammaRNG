# gammaRNG

This repo contains the CUDA implementation of five gamma generator kernels and benchmarking code for the M.Sc. thesis : [**Gamma Random Number Generation on GPUs using CUDA**](https://wertysas.github.io/documents/gamma_rng_on_gpus.pdf)

**The best performing kernels on all GPUs tested are ``cheng1977`` and ``marsaglia_tsang``**

For usage on GPUs I recommend using either ``cheng1977`` or ``marsaglia_tsang``, the best performing kernel
can depend on your GPU type, but they should outperform all other kernels for all values of alpha larger than 1.
More information about the the theory of RNGs and the gamma distribution and further references can be found in the thesis.

## Gamma Generation Kernels

The following gamma generators were selected as potential candidates for effective generators on GPU architectures and implemented in CUDA ([gamma_generators.cuh](include/cuda/gamma_generators.cuh)):

``marsaglia_tsang`` - The generator (without squeeze step) from:
 - **George Marsaglia and Wai Wan Tsang**. A simple method for generating gamma variables. *ACM Trans. Math. Softw.* 26, 3 (Sept. 2000): pp. 363–372. https://doi.org/10.1145/358407.358414 

``cheng1977`` - The generator (GA) from:
- **R. C. H. Cheng.** The Generation of Gamma Variables with Non-Integral Shape Parameter. *Journal of the Royal Statistical Society. Series C (Applied Statistics)* 26, no. 1 (1977): pp. 71–75. https://doi.org/10.2307/2346871

``GKM1``, ``GKM2``, ``GKM3`` - The generators with corresponding names from:
- **R. C. H. Cheng and G. M. Feast.** Some Simple Gamma Variate Generators. *Journal of the Royal Statistical Society. Series C (Applied Statistics)* 28.3 (1979): pp. 290–295. https://doi.org/10.2307/2347200.

``GC`` - The generator (GA) from:
- **J.H. Ahrens and U. Dieter.** Computer methods for sampling from gamma, beta, poisson and bionomial distributions. *ComputingÄ 12.3 (1974): pp. 223–246. https://doi.org/10.1007/BF02293108.

``best1978`` - The generator (XG) from:
- **D. J. Best.** Letters to the Editors. *Journal of the Royal Statistical Society: Series C (Applied Statistics)* 27.2 (1978): pp. 181–182. https://doi.org/10.1111/j.1467-9876.1978.tb01041.x.

## Build Instructions and Usage

The build system used is cmake and the target ``gamma_kernel_benchmark`` correspond to the benchmark executable. An example of how to build and execute the code, and analyze the benchmark output can be found in the google COLAB notebook:
[gammaRNG](https://colab.research.google.com/drive/1nUAEsqd1u1J5OVQPbihVZ8OKk_tsHEro?usp=sharing).

**The kernels are written to be used inlined and to be compiled with the ``O3`` optimization flag. If you would like to use the kernel with lower optimization or run out of registers (and are unable to inline), then I highly recommend you to split up the kernels into:**

1. a **setup step** (that initialize the constants before the do loop).
2. a **rejection loop** (coresponding to the do loop).
