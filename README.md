# CUDA-GEMM-Kernels
Custom GEMM kernels with the goal of beating cuBLAS

INTRO:
General matrix multiplication (GEMM) is a fundemental linear algebra operation that consititutes the foundation of many critical areas in computing including AI/ML, robotics, high performance computing, modeling, etc; therefore, the creation of programs that can increase thoroughput and decrease latency for GEMM operations even by only a few percentage points are highly valuable. Currently, NVIDIA's cuBLAS library is by far the gold standard in this domain, especially when it comes to large, dense, and square matrices. While major improvements have been made in optimizing the computation of oddly shaped matrices (skinny/tall matrices) as well as sparse matrices (improvements that beat cuBLAS, sometimes by a signicant degree) rarely are there custom/research kernels that can beat cuBLAS when it comes to computing large, dense, and square matrices. The goal of this repository is develop a GEMM program that can beat cuBLAS at computing large, dense, and square matrices.

PROJECT OUTLINE:
The goal is to beat cuBLAS at one specific task on specific hardware and then hopefully exptrapolate any gains made to other matrix shapes and hardware. Currently, I'm developing kernels that multiply together two 1024 x 1024 matrices together in mixed precision (FP16 multiply and F32 accumulate) on NVIDIA's T4 GPU (SM verison 7.5).

GENERAL INFO:
This repository is composed of three folders each containing two files. One file consists of code for GEMM functions that I'm testing, experimenting, and benchmarking with and the second file consists of terminal instructions that compile and run the code as well as display NCU compute metrics and SASS. The main functions of all files containing GEMM function code include means for testing latency over thousands of runs as well as code that runs certain functions only once in order to properly outpute ncu compute statistics. 
The first folder, titled "GEMM" consists of GEMM functions I wrote to test out various well known strategies used by all high performance GEMM libraries such as shared memory tiling, vectorized loads, double buffering, etc. The second folder, titled "PTX_GEMM" includes GEMM functions that take advantage of PTX instructions such as mma.sync.aligned.m16n8n8.row.col.f32.f16.f16.f32 in order to test more novel strategies of closing the gap with cuBLAS such as novel strategies for loading global memory into shared memory, register reuse strategies, and playing around with strategies that might allow standard 128 bit vector loads to replace the PTX instruction ldmatrix. The third folder titled "cuBLAS_GEMM" has a cuBLAS implementation written by ChatGPT which I use as a benchmark against my own code.

WARNING:
Much of the code is extremely messy and may take some time to understand. Not all GEMM functions work properly and many exists purely to test and experiment with different strategies. 
