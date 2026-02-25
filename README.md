# CUDA-GEMM-Kernels
Custom GEMM kernels with the goal of beating cuBLAS

INTRO:
General matrix multiplication (GEMM) is a fundemental linear algebra operation that consititutes the foundation of many critical areas in computing including AI/ML, robotics, high performance computing, modeling, etc; therefore, the creation of programs that can increase thoroughput and decrease latency for GEMM operations even by only a few percentage points are highly valuable. Currently, NVIDIA's cuBLAS library is by far the gold standard in this domain, especially when it comes to large, dense, and square matrices. While major improvements have been made in optimizing the computation of oddly shaped matrices (skinny/tall matrices) as well as sparse matrices (improvements that beat cuBLAS, sometimes by a signicant degree) rarely are there custom/research kernels that can beat cuBLAS when it comes to computing large, dense, and square matrices. The goal of this repository is develop a GEMM program that can beat cuBLAS at computing large, dense, and square matrices.

GENERAL INFO:
