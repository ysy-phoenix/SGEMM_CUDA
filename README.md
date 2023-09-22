# Fast CUDA SGEMM from Scratch

Step-by-step optimization of matrix multiplication, implemented in CUDA.
For an explanation of each kernel, see [siboehm.com/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM).

## Overview

Running the kernels on a NVIDIA A6000 (Ampere):

![](benchmark_results.png)

GFLOPs at matrix size 4096x4096:
<!-- benchmark_results -->
| Kernel                              |   GFLOPs/s | Performance relative to cuBLAS   |
|:------------------------------------|-----------:|:---------------------------------|
| 1: Naive                            |      291.7 | 1.6%                             |
| 2: GMEM Coalescing                  |     3425.5 | 18.2%                            |
| 3: SMEM Caching                     |     5012.1 | 26.7%                            |
| 4: 1D Blocktiling                   |    10717.7 | 57.1%                            |
| 5: 2D Blocktiling                   |    14174.3 | 75.5%                            |
| 8: Avoid Bank Conflicts (Offset)    |    14764.8 | 78.6%                            |
| 7: Avoid Bank Conflicts (Linearize) |    15269.4 | 81.3%                            |
| 6: Vectorized Mem Access            |    15404.1 | 82.0%                            |
| 9: Autotuning                       |    16376.8 | 87.2%                            |
| 10: Warptiling                      |    17565   | 93.6%                            |
| 0: cuBLAS                           |    18774.9 | 100.0%                           |
<!-- benchmark_results -->

## Setup

1. Install dependencies: CUDA toolkit 12, Python (+ Seaborn), CMake, Ninja. See [environment.yml](environment.yml).
1. Configure NVCC compilation parameters. Look up your GPUs compute
   capability [here](https://developer.nvidia.com/cuda-gpus). Then configure the `CMakeLists.txt` and change:
    ```cmake
    set(CUDA_COMPUTE_CAPABILITY 80)
    ```
1. Build: `mkdir build && cd build && cmake .. && cmake --build .`
1. Run one of the kernels: `DEVICE=<device_id> ./sgemm <kernel number>`
1. Profiling via [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) (ncu): `make profile KERNEL=<kernel number>`

Credit goes to [wangzyon/NVIDIA_SGEMM_PRACTICE](https://github.com/wangzyon/NVIDIA_SGEMM_PRACTICE) for the benchmarking setup.
