#!nvidia-smi
#!nvcc --version
%%writefile cuBLAS_implementation.cu

// cublas_hgemm_1024_bench.cu
//
// Fair, steady-state cuBLAS HGEMM benchmark for:
//   C (FP32) = A (FP16) * B (FP16)   with FP32 accumulate
//
// Key fairness points:
//  - Single explicit stream used everywhere (cuBLAS + events + warmup + timing)
//  - Warmup on the same stream and synchronized before timing
//  - Events recorded on the same stream
//  - Tensor Core math mode explicitly enabled
//  - Optional: verify TC usage with Nsight Compute / SASS (HMMA) if desired

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#define CUDA_CALL(x) do {                                  \
  cudaError_t err = (x);                                   \
  if (err != cudaSuccess) {                                \
    printf("CUDA error %s at %s:%d\n",                      \
           cudaGetErrorString(err), __FILE__, __LINE__);    \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
} while (0)

#define CUBLAS_CALL(x) do {                                \
  cublasStatus_t st = (x);                                 \
  if (st != CUBLAS_STATUS_SUCCESS) {                       \
    printf("cuBLAS error %d at %s:%d\n",                    \
           (int)st, __FILE__, __LINE__);                   \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
} while (0)

static void fill_half(__half* h, size_t n, unsigned seed) {
  std::srand(seed);
  for (size_t i = 0; i < n; ++i) {
    float v = float(std::rand() % 100) / 100.0f; // [0,1)
    h[i] = __float2half_rn(v);
  }
}

int main() {
  printf("cuBLAS HGEMM 1024x1024x1024 benchmark (FP16 inputs, FP32 accumulate/output)\n");

  // Problem size
  constexpr int M = 1024, N = 1024, K = 1024;
  constexpr int WARMUP = 1000;
  constexpr int RUNS   = 10000;

  // Column-major (cuBLAS native):
  // A is MxK with lda = M
  // B is KxN with ldb = K
  // C is MxN with ldc = M
  const int lda = M, ldb = K, ldc = M;

  // Host buffers
  const size_t elemsA = (size_t)M * K;
  const size_t elemsB = (size_t)K * N;
  const size_t elemsC = (size_t)M * N;

  __half* h_A = (__half*)std::malloc(elemsA * sizeof(__half));
  __half* h_B = (__half*)std::malloc(elemsB * sizeof(__half));
  float*  h_C = (float*) std::malloc(elemsC * sizeof(float));
  if (!h_A || !h_B || !h_C) {
    printf("Host malloc failed\n");
    return 1;
  }

  fill_half(h_A, elemsA, 0);
  fill_half(h_B, elemsB, 1);
  //std::memset(h_C, 0, elemsC * sizeof(float));
  memset(h_C, 0, elemsC * sizeof(float));

  // Device buffers
  __half* d_A = nullptr;
  __half* d_B = nullptr;
  float*  d_C = nullptr;
  CUDA_CALL(cudaMalloc(&d_A, elemsA * sizeof(__half)));
  CUDA_CALL(cudaMalloc(&d_B, elemsB * sizeof(__half)));
  CUDA_CALL(cudaMalloc(&d_C, elemsC * sizeof(float)));

  CUDA_CALL(cudaMemcpy(d_A, h_A, elemsA * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_B, h_B, elemsB * sizeof(__half), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_C, h_C, elemsC * sizeof(float),  cudaMemcpyHostToDevice));

  // Create one explicit stream and use it for everything
  cudaStream_t stream;
  CUDA_CALL(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // cuBLAS setup
  cublasHandle_t handle;
  CUBLAS_CALL(cublasCreate(&handle));
  CUBLAS_CALL(cublasSetStream(handle, stream));

  // Explicitly enable Tensor Core math (where applicable)
  CUBLAS_CALL(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

  // GEMM: C = alpha*A*B + beta*C
  const float alpha = 1.0f;
  const float beta  = 0.0f;

  auto gemm = [&]() {
    CUBLAS_CALL(cublasGemmEx(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      M, N, K,
      &alpha,
      d_A, CUDA_R_16F, lda,
      d_B, CUDA_R_16F, ldb,
      &beta,
      d_C, CUDA_R_32F, ldc,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
  };

  // Warmup on the same stream we will time
  for (int i = 0; i < WARMUP; ++i) gemm();
  CUDA_CALL(cudaStreamSynchronize(stream));

  // Timing events on the same stream
  cudaEvent_t start, stop;
  CUDA_CALL(cudaEventCreate(&start));
  CUDA_CALL(cudaEventCreate(&stop));

  CUDA_CALL(cudaEventRecord(start, stream));
  for (int i = 0; i < RUNS; ++i) gemm();
  CUDA_CALL(cudaEventRecord(stop, stream));
  CUDA_CALL(cudaEventSynchronize(stop));

  float ms = 0.0f;
  CUDA_CALL(cudaEventElapsedTime(&ms, start, stop));

  printf("Avg time per GEMM: %.3f microseconds (runs=%d, warmup=%d)\n",
         (ms * 1000.0f) / RUNS, RUNS, WARMUP);

  // Cleanup
  CUDA_CALL(cudaEventDestroy(start));
  CUDA_CALL(cudaEventDestroy(stop));

  CUBLAS_CALL(cublasDestroy(handle));
  CUDA_CALL(cudaStreamDestroy(stream));

  CUDA_CALL(cudaFree(d_A));
  CUDA_CALL(cudaFree(d_B));
  CUDA_CALL(cudaFree(d_C));

  std::free(h_A);
  std::free(h_B);
  std::free(h_C);

  return 0;
}
