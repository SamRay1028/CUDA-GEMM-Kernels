#include <iostream>
#include <math.h>
#include <functional>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>


using namespace std;
using namespace nvcuda::wmma;

#define WARP_SIZE 32
#define TENSOR_TILE_SIZE 16
#define TILE_SIZE_6 64
#define MICRO_M 16
#define MICRO_N 8
#define MICRO_K 8
#define MACRO_M 64
#define MACRO_N 64
#define MACRO_K 64
#define TILE_M 256
#define TILE_N 128
#define TILE_K 128
#define VEC_SIZE 8




typedef struct half4{
  half w, x, y, z;
}half4;




typedef struct half8{
  half a, b, c, d, e, f, g, h;
}half8;




typedef struct half_2{
  half x, y;
}half_2;




typedef union {
  uint4 temp_u;
  uint32_t u[4];
}u4;




typedef union {
  uint2 temp_u;
  uint32_t u[2];
}u2;




typedef struct uint32_t2{
  uint32_t x, y;
}uint32_t2;




typedef union{
  uint32_t2 u;
  half4 h;
}hu2;




typedef union{
  //half2 h;
  uint32_t u;
}hu1;




__device__ __forceinline__ unsigned shfl_idx(unsigned v, int srcLane, unsigned mask=0xffffffffu) {
  unsigned out;
  asm volatile(
    "shfl.sync.idx.b32 %0, %1, %2, 0x1f, %3;\n"
    : "=r"(out) : "r"(v), "r"(srcLane), "r"(mask)
  );
  return out;
}





__device__ __forceinline__ uint32_t pack_half2(float x, float y) {
    // make a half2, then bitcast to u32
    half2 h2 = __floats2half2_rn(x, y);
    union { half2 h; uint32_t u; } v;
    v.h = h2;
    return v.u;
}




__host__ uint32_t pack2(half x, half y){
  uint32_t package = (uint32_t)x;
  package <<= 16;
  package |= (uint32_t)y;
  return package;
}




__device__ __forceinline__ uint32_t pack(float lo, float hi){
  __half hlo = __float2half_rn(lo);
  __half hhi = __float2half_rn(hi);
  uint16_t u_lo, u_hi;
  memcpy(&u_lo, &hlo, sizeof(uint16_t));
  memcpy(&u_hi, &hhi, sizeof(uint16_t));
  return uint32_t(u_lo) | (uint32_t(u_hi) << 16);
}




__global__ void swap_test(){
  __shared__ int flag;
  flag = 0;
  int lane = threadIdx.x % WARP_SIZE;
  int partner_lane = abs(lane - 31);
  int val = lane;
  val = __shfl_sync(1 << lane, val, partner_lane);
  while(flag == 1);
  atomicAdd(&flag, 1);
  printf("lane %d val: %d\n", lane, val);
  atomicAdd(&flag, -1);
}




__global__ void tc_test(float* out) {
    int lane = threadIdx.x & 31;
    if (threadIdx.x >= 32) return;

    // For mma.m16n8k8.f16 on sm_75:
    // A = {a0, a1} where each is .f16x2 (packed in a .b32 reg)
    // B = {b0}     where b0 is .f16x2 (packed in a .b32 reg)
    uint32_t a0 = pack_half2(1.f, 1.f);
    uint32_t a1 = pack_half2(1.f, 1.f);
    uint32_t b0 = pack_half2(1.f, 1.f);

    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;

    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
        : "r"(a0), "r"(a1),
          "r"(b0)
    );

    if (lane == 0) out[0] = c0;
}




__global__ void naive_tensor_mat_mul_kernel(half *d_A_ptr, half *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols)
{
  int warpM = blockIdx.x; //(blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
  int warpN = blockIdx.y;
  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag;
  fill_fragment(c_frag, 0.0f);
  for (int i = 0; i < A_n_cols; i += 16) {
    int aRow = warpM * 16;
    int aCol = i;
    int bRow = i;
    int bCol = warpN * 16;
    load_matrix_sync(a_frag, &d_A_ptr[aRow * A_n_cols + aCol], A_n_cols);
    load_matrix_sync(b_frag, &d_B_ptr[bRow * C_n_cols + bCol], C_n_cols);
    mma_sync(c_frag, a_frag, b_frag, c_frag);
  }
  int cRow = warpM * 16;
  int cCol = warpN * 16;
  store_matrix_sync(&d_C_ptr[cRow * C_n_cols + cCol], c_frag, C_n_cols, mem_row_major);
}




__global__ void asm_gemm(half* A, half* B, float* C, int M, int N, int K){
  uint32_t a[2];
  uint32_t b[1];
  float c[4];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int lane = threadIdx.x / WARP_SIZE;

  //Don't need to worry about memory access offset right now because shared memory is not incorporated

  int Amem_per_thread = (TENSOR_TILE_SIZE * TENSOR_TILE_SIZE / 2) / blockDim.x;
  int A_division = lane / TENSOR_TILE_SIZE;
  int A_level = lane & (TENSOR_TILE_SIZE - 1);
  int A_buf_bump;

  int Bmem_per_thread = (TENSOR_TILE_SIZE / 2 * TENSOR_TILE_SIZE / 2) / blockDim.x;
  int B_division = lane / (TENSOR_TILE_SIZE / 2);
  int B_level = lane & (TENSOR_TILE_SIZE / 2 - 1);
  int B_buf_bump;

  for(int i = 0; i < TENSOR_TILE_SIZE / 2; ++i){
    A_buf_bump = (by * TENSOR_TILE_SIZE + A_level) * K + (i * (TENSOR_TILE_SIZE / 2) + A_division * Amem_per_thread);
    for(int j = 0; j < Amem_per_thread; j += 2){
      a[j] = pack(A[A_buf_bump + j], A[A_buf_bump + j + 1]);
    }
    for(int j = 0; j < 2; ++j){
      B_buf_bump = (i * (TENSOR_TILE_SIZE / 2) + B_level) * N + (bx * TENSOR_TILE_SIZE + j * (TENSOR_TILE_SIZE / 2) + B_division * Bmem_per_thread);
      for(int k = 0; k < Bmem_per_thread; k += 2){
        b[k] = pack(B[B_buf_bump + k], B[B_buf_bump + k + 1]);
      }

      /*
      asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5}, "
        "{%6}, "
        "{%0, %1, %2, %3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(b[0])
      );
      */
    }
  }

  //for(int i = 0; i < 4; ++i) C[()]
}




__global__ void asm_gemm_2(half* __restrict__ A, half* __restrict__ B, float* __restrict__ C, int M, int N, int K){
  A = (half*)__builtin_assume_aligned(A, 16);
  B = (half*)__builtin_assume_aligned(B, 16);
  C = (float*)__builtin_assume_aligned(C, 16);

  uint32_t a[2];
  uint32_t b[1];
  float c[4];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int lane = threadIdx.x % WARP_SIZE;

  //Don't need to worry about memory access offset right now because shared memory is not incorporated

  int Amem_per_thread = (TENSOR_TILE_SIZE * TENSOR_TILE_SIZE / 2) / blockDim.x;
  int A_division = lane / TENSOR_TILE_SIZE;
  int A_level = lane & (TENSOR_TILE_SIZE - 1);

  int Bmem_per_thread = (TENSOR_TILE_SIZE / 2 * TENSOR_TILE_SIZE / 2) / blockDim.x;
  int B_division = lane / (TENSOR_TILE_SIZE / 2);
  int B_level = lane & (TENSOR_TILE_SIZE / 2 - 1);

  int Cmem_per_thread = (TENSOR_TILE_SIZE * TENSOR_TILE_SIZE / 2) / blockDim.x;
  int C_division = lane / TENSOR_TILE_SIZE;
  int C_level = lane & (TENSOR_TILE_SIZE - 1);

  for(int i = 0; i < 4; ++i) c[i] = 0.0f;

  for(int i = 0; i < K / (TENSOR_TILE_SIZE / 2); ++i){
    a[0] = pack(A[(by * TENSOR_TILE_SIZE + A_level) * K + (i * (TENSOR_TILE_SIZE / 2) + A_division * Amem_per_thread + 0)], A[(by * TENSOR_TILE_SIZE + A_level) * K + (i * (TENSOR_TILE_SIZE / 2) + A_division * Amem_per_thread + 1)]);
    a[1] = pack(A[(by * TENSOR_TILE_SIZE + A_level) * K + (i * (TENSOR_TILE_SIZE / 2) + A_division * Amem_per_thread + 2)], A[(by * TENSOR_TILE_SIZE + A_level) * K + (i * (TENSOR_TILE_SIZE / 2) + A_division * Amem_per_thread + 3)]);
    b[0] = pack(B[(i * (TENSOR_TILE_SIZE / 2) + B_level) * N + (bx * (TENSOR_TILE_SIZE / 2) + B_division * Bmem_per_thread + 0)], B[(i * (TENSOR_TILE_SIZE / 2) + B_level) * N + (bx * (TENSOR_TILE_SIZE / 2) + B_division * Bmem_per_thread + 1)]);
    asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5}, "
      "{%6}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1])
        "r"(b[0])
    );
  }
  for(int i = 0; i < 4; ++i) C[(by * TENSOR_TILE_SIZE + C_level) * N + (bx * (TENSOR_TILE_SIZE / 2) + C_division * Cmem_per_thread + i)] = c[i];

}




__global__ void asm_gemm_3(half* A, half* B, float* C, int M, int N, int K){
  //hu2 a_u;
  //hu1 b_u;
  //uint32_t2 a_u;
  //uint32_t b_u;
  uint2 a_u;
  uint32_t b_u;
  float4 c_f;

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int lane = threadIdx.x % WARP_SIZE;
  int half_tensor_tile_size = TENSOR_TILE_SIZE / 2;

  //Don't need to worry about memory access offset right now because shared memory is not incorporated

  /*
  int Amem_per_thread = (TENSOR_TILE_SIZE * half_tensor_tile_size) / blockDim.x;
  int A_division = lane / TENSOR_TILE_SIZE;
  int A_level = lane & (TENSOR_TILE_SIZE - 1);

  int Bmem_per_thread = (half_tensor_tile_size * half_tensor_tile_size) / blockDim.x;
  int B_division = lane / (half_tensor_tile_size);
  int B_level = lane & (half_tensor_tile_size - 1);

  int Cmem_per_thread = (TENSOR_TILE_SIZE * half_tensor_tile_size) / blockDim.x;
  int C_division = lane / TENSOR_TILE_SIZE;
  int C_level = lane & (TENSOR_TILE_SIZE - 1);
  */

  int Amem_per_thread = (TENSOR_TILE_SIZE * half_tensor_tile_size) / blockDim.x;
  int A_division = lane % (half_tensor_tile_size / Amem_per_thread);
  int A_level = lane / (half_tensor_tile_size / Amem_per_thread);

  int Bmem_per_thread = (half_tensor_tile_size * half_tensor_tile_size) / blockDim.x;
  int B_division = lane % (half_tensor_tile_size / Bmem_per_thread);
  int B_level = lane / (half_tensor_tile_size / Bmem_per_thread);

  int Cmem_per_thread = (TENSOR_TILE_SIZE * half_tensor_tile_size) / blockDim.x;
  int C_division = lane % (half_tensor_tile_size / Cmem_per_thread);
  int C_level = lane / (half_tensor_tile_size / Cmem_per_thread);

  c_f = {0.0f, 0.0f, 0.0f, 0.0f};

  for(int i = 0; i < K / half_tensor_tile_size ; ++i){
    //a_u.h = *reinterpret_cast<half4*>(&A[(by * TENSOR_TILE_SIZE + A_level) * K + (i * half_tensor_tile_size + A_division * Amem_per_thread)]);
    //b_u.h = *reinterpret_cast<half2*>(&B[(i * half_tensor_tile_size + B_level) * N + (bx * half_tensor_tile_size + B_division * Bmem_per_thread)]);
    //a_u = *reinterpret_cast<hu2*>(&A[(by * TENSOR_TILE_SIZE + A_level) * K + (i * half_tensor_tile_size + A_division * Amem_per_thread)]);
    //b_u = *reinterpret_cast<hu1*>(&B[(i * half_tensor_tile_size + B_level) * N + (bx * half_tensor_tile_size + B_division * Bmem_per_thread)]);
    a_u = *reinterpret_cast<uint2*>(&A[(by * TENSOR_TILE_SIZE + A_level) * K + (i * half_tensor_tile_size + A_division * Amem_per_thread)]);
    b_u = *reinterpret_cast<uint32_t*>(&B[(i * half_tensor_tile_size + B_level) * N + (bx * half_tensor_tile_size + B_division * Bmem_per_thread)]);
    asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5}, "
      "{%6}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(c_f.w), "+f"(c_f.x), "+f"(c_f.y), "+f"(c_f.z)
      : "r"(a_u.x), "r"(a_u.y)
        "r"(b_u)
    );
  }
  *reinterpret_cast<float4*>(&C[(by * TENSOR_TILE_SIZE + C_level) * N + (bx * (TENSOR_TILE_SIZE / 2) + C_division * Cmem_per_thread)]) = c_f;
}




__global__ void asm_gemm_4(half* A, half* B, float* C, int M, int N, int K){
  A = (half*)__builtin_assume_aligned(A, 16);
  B = (half*)__builtin_assume_aligned(B, 8);
  C = (float*)__builtin_assume_aligned(C, 16);

  //uint4 a_u;
  //uint2 b_u;
  u4 a_u;
  uint2 a;
  u2 b_u;
  uint32_t b;
  float4 c_f;

  c_f = {0.0f, 0.0f, 0.0f, 0.0f};

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int lane = threadIdx.x % WARP_SIZE;
  int half_tensor_tile_size = TENSOR_TILE_SIZE / 2;

  int Amem_per_thread = (TENSOR_TILE_SIZE * TENSOR_TILE_SIZE) / blockDim.x;
  int A_division = lane % (TENSOR_TILE_SIZE / Amem_per_thread);
  int A_level = lane / (TENSOR_TILE_SIZE / Amem_per_thread);

  int Bmem_per_thread = (TENSOR_TILE_SIZE * half_tensor_tile_size) / blockDim.x;
  int B_division = lane % (half_tensor_tile_size / Bmem_per_thread);
  int B_level = lane / (half_tensor_tile_size / Bmem_per_thread);

  int Cmem_per_thread = (TENSOR_TILE_SIZE * half_tensor_tile_size) / blockDim.x;
  int C_division = lane % (half_tensor_tile_size / Cmem_per_thread);
  int C_level = lane / (half_tensor_tile_size / Cmem_per_thread);

  int a_index_buf = 2 * (lane % 2);
  int inverse_a_index_buf = 2 * (1 - (lane % 2));
  int a_swap_lane = lane ^ 1;
  int b_index_buf = lane / 16;
  int inverse_b_index_buf = 1 - (lane / 16);
  int b_swap_lane = lane ^ 16;

  for(int i = 0; i < K / TENSOR_TILE_SIZE; ++i){
    a_u.temp_u = *reinterpret_cast<uint4*>(&A[(by * TENSOR_TILE_SIZE + A_level) * K + (i * TENSOR_TILE_SIZE + A_division * Amem_per_thread)]);
    b_u.temp_u = *reinterpret_cast<uint2*>(&B[(i * TENSOR_TILE_SIZE + B_level) * N + (bx * half_tensor_tile_size + B_division * Bmem_per_thread)]);
    //a_u.u[a_index_buf] = __shfl_sync(0xFFFFFFFF, a_u.u[inverse_a_index_buf], a_swap_lane);
    //a_u.u[a_index_buf + 1] = __shfl_sync(0xFFFFFFFF, a_u.u[inverse_a_index_buf + 1], a_swap_lane);
    //b_u.u[b_index_buf] = __shfl_sync(0xFFFFFFFF, b_u.u[inverse_b_index_buf], b_swap_lane);
    //a.x = __shfl_sync(0xFFFFFFFF, a_u.u[2], lane ^ 1);

    /*
    uint32_t src_adj  = lane ^ 1;
    asm volatile(
      "ld.global.v4.u32 {%0, %1, %2, %3}, [%5];\n"
      "shfl.sync.idx.b32 %0, %2, %4, 0x1f, 0xFFFFFFFF;\n"
      : "+r"(a_u.u[0]), "+r"(a_u.u[1]), "+r"(a_u.u[2]), "+r"(a_u.u[3]), "+r"(src_adj)
      : "l"(&A[(by * TENSOR_TILE_SIZE + A_level) * K + (i * TENSOR_TILE_SIZE + A_division * Amem_per_thread)])
    );
    */

    /*
    a_u.temp_u = *reinterpret_cast<uint4*>(&A[(by * TENSOR_TILE_SIZE + A_level) * K + (i * TENSOR_TILE_SIZE + A_division * Amem_per_thread)]);
    b_u.temp_u = *reinterpret_cast<uint2*>(&B[(i * TENSOR_TILE_SIZE + B_level) * N + (bx * half_tensor_tile_size + B_division * Bmem_per_thread)]);
    a_u.u[0] = shfl_idx(a_u.u[2], lane ^ 1, 0xffffffffu);
    a_u.u[1] = shfl_idx(a_u.u[3], lane ^ 1, 0xffffffffu);
    //a_u.u[0] = __shfl_sync(0xFFFFFFFFu, a_u.u[2], abs(lane - 1));
    //a_u.u[1] = __shfl_sync(0xFFFFFFFFu, a_u.u[3], abs(lane - 1));
    //b_u.u[0] = __shfl_sync(0xFFFFFFFFu, b_u.u[1], abs(lane - 16));
    */

    /*
    asm volatile(
      "ld.global.v4.u32 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(a_u.x), "=r"(a_u.y), "=r"(a_u.z), "=r"(a_u.w)
      : "l"(&A[(by * TENSOR_TILE_SIZE + A_level) * K + (i * TENSOR_TILE_SIZE + A_division * Amem_per_thread)])
    );
    asm volatile(
      "ld.global.v2.u32 {%0, %1}, [%2];\n"
      : "=r"(b_u.x), "=r"(b_u.y)
      : "l"(&B[(i * TENSOR_TILE_SIZE + B_level) * N + (bx * half_tensor_tile_size + B_division * Bmem_per_thread)])
    );

    int src_adj  = lane ^ 1;
    int src_half = lane ^ 16;

    a_u.x = __shfl_sync(0xFFFFFFFFu, a_u.z, src_adj);
    a_u.y = __shfl_sync(0xFFFFFFFFu, a_u.w, src_adj);
    b_u.y = __shfl_sync(0xFFFFFFFFu, b_u.x, src_half);
    */

    /*
    a_u = *reinterpret_cast<uint4*>(&A[(by * TENSOR_TILE_SIZE + A_level) * K + (i * TENSOR_TILE_SIZE + A_division * Amem_per_thread)]);
    b_u = *reinterpret_cast<uint2*>(&B[(i * TENSOR_TILE_SIZE + B_level) * N + (bx * half_tensor_tile_size + B_division * Bmem_per_thread)]);
    __syncwarp();
    a_u.x = __shfl_sync(1 << lane, a_u.z, abs(lane - 1));
    a_u.y = __shfl_sync(1 << lane, a_u.w, abs(lane - 1));
    b_u.y = __shfl_sync(1 << lane, b_u.x, abs(lane - 16));
    */

    asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5}, "
      "{%6}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(c_f.x), "+f"(c_f.y), "+f"(c_f.z), "+f"(c_f.w)
      : "r"(a_u.u[0]), "r"(a_u.u[1])
        "r"(b_u.u[0])
    );
    asm volatile(
      "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
      "{%0, %1, %2, %3}, "
      "{%4, %5}, "
      "{%6}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(c_f.x), "+f"(c_f.y), "+f"(c_f.z), "+f"(c_f.w)
      : "r"(a_u.u[2]), "r"(a_u.u[3])
        "r"(b_u.u[1])
    );
  }
  *reinterpret_cast<float4*>(&C[(by * TENSOR_TILE_SIZE + C_level) * N + (bx * half_tensor_tile_size + C_division * Cmem_per_thread)]) = c_f;
}




__global__ void asm_gemm_5(half* A, half* B, float* C, int M, int N, int K){
  A = (half*)__builtin_assume_aligned(A, 16);
  B = (half*)__builtin_assume_aligned(B, 16);
  C = (float*)__builtin_assume_aligned(C, 16);

  u4 a[(MACRO_M * MACRO_K) / WARP_SIZE / VEC_SIZE];
  u4 b[(MACRO_K * MACRO_N) / WARP_SIZE / VEC_SIZE];
  float4 c[(MACRO_M * MACRO_N) / WARP_SIZE / (VEC_SIZE / 2)];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int lane = threadIdx.x % WARP_SIZE;

  int m_sections = MACRO_M / MICRO_M;
  int n_sections = MACRO_N / MICRO_N;
  int k_sections = MACRO_K / MICRO_K;

  /*
  int Amem_per_thread = (MACRO_M * MACRO_K) / blockDim.x;
  int A_division = lane % (MACRO_K / Amem_per_thread);
  int A_level = lane / (MACRO_K / Amem_per_thread);
  int A_mem_per_row = Amem_per_thread;
  int A_buf;

  int Bmem_per_thread = (MACRO_K * MACRO_N) / blockDim.x;
  int B_division = lane % (MACRO_N / Bmem_per_thread);
  int B_level = lane / (MACRO_N / Bmem_per_thread);
  int B_mem_per_row = Bmem_per_thread;
  int B_buf;

  int Cmem_per_thread = (MACRO_M * MACRO_N) / blockDim.x;
  int C_division = lane % (MACRO_N / Cmem_per_thread);
  int C_level = lane / (MACRO_N / Cmem_per_thread);
  int C_mem_per_row = Cmem_per_thread;
  int C_buf;
  */

  int Amem_per_thread = (MACRO_M * MACRO_K) / blockDim.x;
  int A_division = lane % ((MACRO_K / Amem_per_thread == 0) ? 1 : MACRO_K / Amem_per_thread);
  int A_level = lane / ((MACRO_K / Amem_per_thread == 0) ? 1 : MACRO_K / Amem_per_thread) * ((Amem_per_thread / MACRO_K == 0) ? 1 : Amem_per_thread / MACRO_K);
  int A_mem_per_row = (Amem_per_thread < MACRO_K) ? MACRO_K : Amem_per_thread;
  int A_buf;

  int Bmem_per_thread = (MACRO_K * MACRO_N) / blockDim.x;
  int B_division = lane % ((MACRO_N / Bmem_per_thread == 0) ? 1 : MACRO_N / Bmem_per_thread);
  int B_level = lane / ((MACRO_N / Bmem_per_thread == 0) ? 1 : MACRO_N / Bmem_per_thread) * ((Bmem_per_thread / MACRO_N == 0) ? 1 : Bmem_per_thread / MACRO_N);
  int B_mem_per_row = (Bmem_per_thread < MACRO_N) ? MACRO_N : Bmem_per_thread;
  int B_buf;

  int Cmem_per_thread = (MACRO_M * MACRO_N) / blockDim.x;
  int C_division = lane % ((MACRO_N / Cmem_per_thread == 0) ? 1 : MACRO_N / Cmem_per_thread);
  int C_level = lane / ((MACRO_N / Cmem_per_thread == 0) ? 1 : MACRO_N / Cmem_per_thread) * ((Cmem_per_thread / MACRO_N == 0) ? 1 : Cmem_per_thread / MACRO_N);
  int C_mem_per_row = (Cmem_per_thread < MACRO_N) ? MACRO_N : Cmem_per_thread;
  int C_buf;

  int volatile rand_var;

  for(int i = 0; i < (MACRO_M * MACRO_N) / blockDim.x / (VEC_SIZE / 2); ++i) c[i] = {0.0f, 0.0f, 0.0f, 0.0f};

  for(int i = 0; i < K / MACRO_K; ++i){
    a[0].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + A_level + (0 * VEC_SIZE / A_mem_per_row)) * K + (i * MACRO_K + A_division * Amem_per_thread + (0 * VEC_SIZE % A_mem_per_row))]);
    a[1].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + A_level + (1 * VEC_SIZE / A_mem_per_row)) * K + (i * MACRO_K + A_division * Amem_per_thread + (1 * VEC_SIZE % A_mem_per_row))]);
    a[2].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + A_level + (2 * VEC_SIZE / A_mem_per_row)) * K + (i * MACRO_K + A_division * Amem_per_thread + (2 * VEC_SIZE % A_mem_per_row))]);
    a[3].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + A_level + (3 * VEC_SIZE / A_mem_per_row)) * K + (i * MACRO_K + A_division * Amem_per_thread + (3 * VEC_SIZE % A_mem_per_row))]);
    a[4].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + A_level + (4 * VEC_SIZE / A_mem_per_row)) * K + (i * MACRO_K + A_division * Amem_per_thread + (4 * VEC_SIZE % A_mem_per_row))]);
    a[5].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + A_level + (5 * VEC_SIZE / A_mem_per_row)) * K + (i * MACRO_K + A_division * Amem_per_thread + (5 * VEC_SIZE % A_mem_per_row))]);
    a[6].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + A_level + (6 * VEC_SIZE / A_mem_per_row)) * K + (i * MACRO_K + A_division * Amem_per_thread + (6 * VEC_SIZE % A_mem_per_row))]);
    a[7].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + A_level + (7 * VEC_SIZE / A_mem_per_row)) * K + (i * MACRO_K + A_division * Amem_per_thread + (7 * VEC_SIZE % A_mem_per_row))]);
    a[8].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + A_level + (8 * VEC_SIZE / A_mem_per_row)) * K + (i * MACRO_K + A_division * Amem_per_thread + (8 * VEC_SIZE % A_mem_per_row))]);
    a[9].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + A_level + (9 * VEC_SIZE / A_mem_per_row)) * K + (i * MACRO_K + A_division * Amem_per_thread + (9 * VEC_SIZE % A_mem_per_row))]);
    a[10].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + A_level + (10 * VEC_SIZE / A_mem_per_row)) * K + (i * MACRO_K + A_division * Amem_per_thread + (10 * VEC_SIZE % A_mem_per_row))]);
    a[11].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + A_level + (11 * VEC_SIZE / A_mem_per_row)) * K + (i * MACRO_K + A_division * Amem_per_thread + (11 * VEC_SIZE % A_mem_per_row))]);
    a[12].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + A_level + (12 * VEC_SIZE / A_mem_per_row)) * K + (i * MACRO_K + A_division * Amem_per_thread + (12 * VEC_SIZE % A_mem_per_row))]);
    a[13].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + A_level + (13 * VEC_SIZE / A_mem_per_row)) * K + (i * MACRO_K + A_division * Amem_per_thread + (13 * VEC_SIZE % A_mem_per_row))]);
    a[14].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + A_level + (14 * VEC_SIZE / A_mem_per_row)) * K + (i * MACRO_K + A_division * Amem_per_thread + (14 * VEC_SIZE % A_mem_per_row))]);
    a[15].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + A_level + (15 * VEC_SIZE / A_mem_per_row)) * K + (i * MACRO_K + A_division * Amem_per_thread + (15 * VEC_SIZE % A_mem_per_row))]);

    b[0].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + B_level + (0 * VEC_SIZE / B_mem_per_row)) * N + (bx * MACRO_N + B_division * Bmem_per_thread + (0 * VEC_SIZE % B_mem_per_row))]);
    b[1].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + B_level + (1 * VEC_SIZE / B_mem_per_row)) * N + (bx * MACRO_N + B_division * Bmem_per_thread + (1 * VEC_SIZE % B_mem_per_row))]);
    b[2].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + B_level + (2 * VEC_SIZE / B_mem_per_row)) * N + (bx * MACRO_N + B_division * Bmem_per_thread + (2 * VEC_SIZE % B_mem_per_row))]);
    b[3].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + B_level + (3 * VEC_SIZE / B_mem_per_row)) * N + (bx * MACRO_N + B_division * Bmem_per_thread + (3 * VEC_SIZE % B_mem_per_row))]);
    b[4].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + B_level + (4 * VEC_SIZE / B_mem_per_row)) * N + (bx * MACRO_N + B_division * Bmem_per_thread + (4 * VEC_SIZE % B_mem_per_row))]);
    b[5].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + B_level + (5 * VEC_SIZE / B_mem_per_row)) * N + (bx * MACRO_N + B_division * Bmem_per_thread + (5 * VEC_SIZE % B_mem_per_row))]);
    b[6].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + B_level + (6 * VEC_SIZE / B_mem_per_row)) * N + (bx * MACRO_N + B_division * Bmem_per_thread + (6 * VEC_SIZE % B_mem_per_row))]);
    b[7].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + B_level + (7 * VEC_SIZE / B_mem_per_row)) * N + (bx * MACRO_N + B_division * Bmem_per_thread + (7 * VEC_SIZE % B_mem_per_row))]);
    b[8].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + B_level + (8 * VEC_SIZE / B_mem_per_row)) * N + (bx * MACRO_N + B_division * Bmem_per_thread + (8 * VEC_SIZE % B_mem_per_row))]);
    b[9].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + B_level + (9 * VEC_SIZE / B_mem_per_row)) * N + (bx * MACRO_N + B_division * Bmem_per_thread + (9 * VEC_SIZE % B_mem_per_row))]);
    b[10].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + B_level + (10 * VEC_SIZE / B_mem_per_row)) * N + (bx * MACRO_N + B_division * Bmem_per_thread + (10 * VEC_SIZE % B_mem_per_row))]);
    b[11].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + B_level + (11 * VEC_SIZE / B_mem_per_row)) * N + (bx * MACRO_N + B_division * Bmem_per_thread + (11 * VEC_SIZE % B_mem_per_row))]);
    b[12].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + B_level + (12 * VEC_SIZE / B_mem_per_row)) * N + (bx * MACRO_N + B_division * Bmem_per_thread + (12 * VEC_SIZE % B_mem_per_row))]);
    b[13].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + B_level + (13 * VEC_SIZE / B_mem_per_row)) * N + (bx * MACRO_N + B_division * Bmem_per_thread + (13 * VEC_SIZE % B_mem_per_row))]);
    b[14].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + B_level + (14 * VEC_SIZE / B_mem_per_row)) * N + (bx * MACRO_N + B_division * Bmem_per_thread + (14 * VEC_SIZE % B_mem_per_row))]);
    b[15].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + B_level + (15 * VEC_SIZE / B_mem_per_row)) * N + (bx * MACRO_N + B_division * Bmem_per_thread + (15 * VEC_SIZE % B_mem_per_row))]);

    //Below is a worst case scenario for shuffle operations (16 shuffle operations in total)

    /*
    a[0].u[0] = __shfl_sync(0xffffffff, a[0].u[3], abs(31 - lane));
    a[0].u[1] = __shfl_sync(0xffffffff, a[0].u[2], abs(31 - lane - 1));
    a[0].u[2] = __shfl_sync(0xffffffff, a[0].u[1], abs(31 - lane - 2));
    a[0].u[3] = __shfl_sync(0xffffffff, a[0].u[0], abs(31 - lane - 3));

    b[0].u[0] = __shfl_sync(0xffffffff, b[0].u[3], abs(31 - lane));
    b[0].u[1] = __shfl_sync(0xffffffff, b[0].u[2], abs(31 - lane - 1));
    b[0].u[2] = __shfl_sync(0xffffffff, b[0].u[1], abs(31 - lane - 2));
    b[0].u[3] = __shfl_sync(0xffffffff, b[0].u[0], abs(31 - lane - 3));

    a[0].u[0] = __shfl_sync(0xffffffff, a[0].u[2], abs(31 - lane));
    a[0].u[1] = __shfl_sync(0xffffffff, a[0].u[3], abs(31 - lane - 4));
    a[0].u[2] = __shfl_sync(0xffffffff, a[0].u[0], abs(31 - lane - 5));
    a[0].u[3] = __shfl_sync(0xffffffff, a[0].u[1], abs(31 - lane - 6));

    b[0].u[0] = __shfl_sync(0xffffffff, b[0].u[2], abs(31 - lane));
    b[0].u[1] = __shfl_sync(0xffffffff, b[0].u[3], abs(31 - lane - 4));
    b[0].u[2] = __shfl_sync(0xffffffff, b[0].u[0], abs(31 - lane - 5));
    b[0].u[3] = __shfl_sync(0xffffffff, b[0].u[1], abs(31 - lane - 6));
    */


    /*
    #pragma unroll 16
    for(int j = 0; j < Amem_per_thread / VEC_SIZE; ++j){
      a[j].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + A_level + (j * VEC_SIZE / A_mem_per_row)) * K + (i * MACRO_K + A_division * Amem_per_thread + (j * VEC_SIZE % A_mem_per_row))]);
    }
    #pragma unroll 16
    for(int j = 0; j < Bmem_per_thread / VEC_SIZE; ++j){
      b[j].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + B_level + (j * VEC_SIZE / B_mem_per_row)) * N + (bx * MACRO_N + B_division * Bmem_per_thread + (j * VEC_SIZE % B_mem_per_row))]);
    }
    */

    //shlf sync instructions would usually go here, do not include them for now

    for(int j = 0; j < MACRO_K / TENSOR_TILE_SIZE; ++j){
      for(int k = 0; k < MACRO_M / TENSOR_TILE_SIZE; ++k){
        for(int l = 0; l < MACRO_N / TENSOR_TILE_SIZE; ++l){
          asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3},"
            "{%8, %9},"
            "{%12},"
            "{%0, %1, %2, %3};\n"
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%13},"
            "{%4, %5, %6, %7};\n"
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3},"
            "{%10, %11},"
            "{%14},"
            "{%0, %1, %2, %3};\n"
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%4, %5, %6, %7},"
            "{%10, %11},"
            "{%15},"
            "{%4, %5, %6, %7};\n"
            : "+f"(c[2 * l].w), "+f"(c[2 * l].x), "+f"(c[2 * l].y), "+f"(c[2 * l].z), "+f"(c[2 * l  + 1].w), "+f"(c[2 * l + 1].x), "+f"(c[2 * l + 1].y), "+f"(c[2 * l + 1].z)
            : "r"(a[j].u[0]), "r"(a[j].u[1]), "r"(a[j].u[2]), "r"(a[j].u[3])
              "r"(b[l].u[0]), "r"(b[l].u[1]), "r"(b[l].u[2]), "r"(b[l].u[3])
          );
        }
      }
    }


  }
  for(int i = 0; i < Cmem_per_thread / (VEC_SIZE / 2); ++i) *reinterpret_cast<float4*>(&C[(by * MACRO_M + C_level + (i * (VEC_SIZE / 2) / C_mem_per_row)) * N + (bx * MACRO_N + C_division * Cmem_per_thread + (i * (VEC_SIZE / 2) % C_mem_per_row))]) = c[i];
  //for(int i = 0; i < Cmem_per_thread / (VEC_SIZE / 2); ++i) *reinterpret_cast<float4*>(&C[4 + i * (VEC_SIZE)]) = c[i];

}




__global__ void asm_gemm_6(half* A, half* B, float* C, int M, int N, int K){
  A = (half*)__builtin_assume_aligned(A, 16);
  B = (half*)__builtin_assume_aligned(B, 16);
  C = (float*)__builtin_assume_aligned(C, 16);

  u4 a[2];
  u4 b[2][MACRO_N / TENSOR_TILE_SIZE];
  float4 c[(MACRO_M * MACRO_N) / WARP_SIZE / (VEC_SIZE / 2)];

  int read_A = 0;
  int write_A = 1;

  int read_B = 0;
  int write_B = 1;

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int lane = threadIdx.x % WARP_SIZE;

  int division = lane % (TENSOR_TILE_SIZE / VEC_SIZE);
  int level = lane / (TENSOR_TILE_SIZE / VEC_SIZE);

  int Cmem_per_thread = (MACRO_M * MACRO_N) / blockDim.x;
  int C_division = lane % ((MACRO_N / Cmem_per_thread == 0) ? 1 : MACRO_N / Cmem_per_thread);
  int C_level = lane / ((MACRO_N / Cmem_per_thread == 0) ? 1 : MACRO_N / Cmem_per_thread) * ((Cmem_per_thread / MACRO_N == 0) ? 1 : Cmem_per_thread / MACRO_N);
  int C_mem_per_row = (Cmem_per_thread < MACRO_N) ? MACRO_N : Cmem_per_thread;

  for(int i = 0; i < (MACRO_M * MACRO_N) / blockDim.x / (VEC_SIZE / 2); ++i) c[i] = {0.0f, 0.0f, 0.0f, 0.0f};

  for(int i = 0; i < K / MACRO_K; ++i){
    b[0][0].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + 0 * TENSOR_TILE_SIZE + level) * N + (bx * MACRO_N + 0 * TENSOR_TILE_SIZE + division * VEC_SIZE)]);
    b[0][1].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + 0 * TENSOR_TILE_SIZE + level) * N + (bx * MACRO_N + 1 * TENSOR_TILE_SIZE + division * VEC_SIZE)]);
    b[0][2].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + 0 * TENSOR_TILE_SIZE + level) * N + (bx * MACRO_N + 2 * TENSOR_TILE_SIZE + division * VEC_SIZE)]);
    b[0][3].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + 0 * TENSOR_TILE_SIZE + level) * N + (bx * MACRO_N + 3 * TENSOR_TILE_SIZE + division * VEC_SIZE)]);
    for(int j = 0; j < MACRO_K / TENSOR_TILE_SIZE - 1; ++j){
      /*
      for(int k = 0; k < MACRO_N / TENSOR_TILE_SIZE - 1; ++k){
        b[k].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + j * TENSOR_TILE_SIZE + level) * N + (bx * MACRO_N + k * TENSOR_TILE_SIZE + division * VEC_SIZE)]);
      }
      */
      b[write_B][0].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + j * TENSOR_TILE_SIZE + level) * N + (bx * MACRO_N + 0 * TENSOR_TILE_SIZE + division * VEC_SIZE)]);
      b[write_B][1].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + j * TENSOR_TILE_SIZE + level) * N + (bx * MACRO_N + 1 * TENSOR_TILE_SIZE + division * VEC_SIZE)]);
      b[write_B][2].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + j * TENSOR_TILE_SIZE + level) * N + (bx * MACRO_N + 2 * TENSOR_TILE_SIZE + division * VEC_SIZE)]);
      b[write_B][3].temp_u = *reinterpret_cast<uint4*>(&B[(i * MACRO_K + j * TENSOR_TILE_SIZE + level) * N + (bx * MACRO_N + 3 * TENSOR_TILE_SIZE + division * VEC_SIZE)]);

      a[0].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + 0 * TENSOR_TILE_SIZE + level) * K + (i * MACRO_K + j * TENSOR_TILE_SIZE + division * VEC_SIZE)]);
      for(int k = 0; k < MACRO_M / TENSOR_TILE_SIZE - 1; ++k){
        a[write_A].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + k * TENSOR_TILE_SIZE + level) * K + (i * MACRO_K + j * TENSOR_TILE_SIZE + division * VEC_SIZE)]);
        for(int l = 0; l < MACRO_N / TENSOR_TILE_SIZE; ++l){
          asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3},"
            "{%8, %9},"
            "{%12},"
            "{%0, %1, %2, %3};\n"
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%13},"
            "{%4, %5, %6, %7};\n"
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3},"
            "{%10, %11},"
            "{%14},"
            "{%0, %1, %2, %3};\n"
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%4, %5, %6, %7},"
            "{%10, %11},"
            "{%15},"
            "{%4, %5, %6, %7};\n"
            : "+f"(c[2 * l].w), "+f"(c[2 * l].x), "+f"(c[2 * l].y), "+f"(c[2 * l].z), "+f"(c[2 * l  + 1].w), "+f"(c[2 * l + 1].x), "+f"(c[2 * l + 1].y), "+f"(c[2 * l + 1].z)
            : "r"(a[read_A].u[0]), "r"(a[read_A].u[1]), "r"(a[read_A].u[2]), "r"(a[read_A].u[3])
              "r"(b[read_B][l].u[0]), "r"(b[read_B][l].u[1]), "r"(b[read_B][l].u[2]), "r"(b[read_B][l].u[3])
          );
        }
        read_A = (read_A + 1) % 2;
        write_A = (write_A + 1) % 2;
      }
      for(int l = 0; l < MACRO_N / TENSOR_TILE_SIZE; ++l){
        asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3},"
          "{%8, %9},"
          "{%12},"
          "{%0, %1, %2, %3};\n"
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%4, %5, %6, %7},"
          "{%8, %9},"
          "{%13},"
          "{%4, %5, %6, %7};\n"
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3},"
          "{%10, %11},"
          "{%14},"
          "{%0, %1, %2, %3};\n"
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%4, %5, %6, %7},"
          "{%10, %11},"
          "{%15},"
          "{%4, %5, %6, %7};\n"
          : "+f"(c[2 * l].w), "+f"(c[2 * l].x), "+f"(c[2 * l].y), "+f"(c[2 * l].z), "+f"(c[2 * l  + 1].w), "+f"(c[2 * l + 1].x), "+f"(c[2 * l + 1].y), "+f"(c[2 * l + 1].z)
          : "r"(a[read_A].u[0]), "r"(a[read_A].u[1]), "r"(a[read_A].u[2]), "r"(a[read_A].u[3])
            "r"(b[read_B][l].u[0]), "r"(b[read_B][l].u[1]), "r"(b[read_B][l].u[2]), "r"(b[read_B][l].u[3])
        );
      }
      read_B = (read_B + 1) % 2;
      write_B = (write_B + 1) % 2;
    }
    a[0].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + 0 * TENSOR_TILE_SIZE + level) * K + (i * MACRO_K + (MACRO_K / TENSOR_TILE_SIZE - 1) * TENSOR_TILE_SIZE + division * VEC_SIZE)]);
      for(int k = 0; k < MACRO_M / TENSOR_TILE_SIZE - 1; ++k){
        a[write_A].temp_u = *reinterpret_cast<uint4*>(&A[(by * MACRO_M + k * TENSOR_TILE_SIZE + level) * K + (i * MACRO_K + (MACRO_K / TENSOR_TILE_SIZE - 1) * TENSOR_TILE_SIZE + division * VEC_SIZE)]);
        for(int l = 0; l < MACRO_N / TENSOR_TILE_SIZE; ++l){
          asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3},"
            "{%8, %9},"
            "{%12},"
            "{%0, %1, %2, %3};\n"
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%4, %5, %6, %7},"
            "{%8, %9},"
            "{%13},"
            "{%4, %5, %6, %7};\n"
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%0, %1, %2, %3},"
            "{%10, %11},"
            "{%14},"
            "{%0, %1, %2, %3};\n"
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
            "{%4, %5, %6, %7},"
            "{%10, %11},"
            "{%15},"
            "{%4, %5, %6, %7};\n"
            : "+f"(c[2 * l].w), "+f"(c[2 * l].x), "+f"(c[2 * l].y), "+f"(c[2 * l].z), "+f"(c[2 * l  + 1].w), "+f"(c[2 * l + 1].x), "+f"(c[2 * l + 1].y), "+f"(c[2 * l + 1].z)
            : "r"(a[read_A].u[0]), "r"(a[read_A].u[1]), "r"(a[read_A].u[2]), "r"(a[read_A].u[3])
              "r"(b[read_B][l].u[0]), "r"(b[read_B][l].u[1]), "r"(b[read_B][l].u[2]), "r"(b[read_B][l].u[3])
          );
        }
      }
      for(int l = 0; l < MACRO_N / TENSOR_TILE_SIZE; ++l){
        asm volatile(
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3},"
          "{%8, %9},"
          "{%12},"
          "{%0, %1, %2, %3};\n"
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%4, %5, %6, %7},"
          "{%8, %9},"
          "{%13},"
          "{%4, %5, %6, %7};\n"
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%0, %1, %2, %3},"
          "{%10, %11},"
          "{%14},"
          "{%0, %1, %2, %3};\n"
          "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32"
          "{%4, %5, %6, %7},"
          "{%10, %11},"
          "{%15},"
          "{%4, %5, %6, %7};\n"
          : "+f"(c[2 * l].w), "+f"(c[2 * l].x), "+f"(c[2 * l].y), "+f"(c[2 * l].z), "+f"(c[2 * l  + 1].w), "+f"(c[2 * l + 1].x), "+f"(c[2 * l + 1].y), "+f"(c[2 * l + 1].z)
          : "r"(a[read_A].u[0]), "r"(a[read_A].u[1]), "r"(a[read_A].u[2]), "r"(a[read_A].u[3])
            "r"(b[read_B][l].u[0]), "r"(b[read_B][l].u[1]), "r"(b[read_B][l].u[2]), "r"(b[read_B][l].u[3])
        );
      }
  }
  for(int i = 0; i < Cmem_per_thread / (VEC_SIZE / 2); ++i) *reinterpret_cast<float4*>(&C[(by * MACRO_M + C_level + (i * (VEC_SIZE / 2) / C_mem_per_row)) * N + (bx * MACRO_N + C_division * Cmem_per_thread + (i * (VEC_SIZE / 2) % C_mem_per_row))]) = c[i];
}




__global__ void asm_gemm_7(half* A, half* B, float* C, int M, int N, int K){
  __shared__ u4 tile_a[TILE_M / MACRO_M][TILE_N / MACRO_N][TENSOR_TILE_SIZE][TENSOR_TILE_SIZE / VEC_SIZE];
  __shared__ u4 tile_b[TILE_N / TENSOR_TILE_SIZE][TENSOR_TILE_SIZE][TENSOR_TILE_SIZE / VEC_SIZE];
  A = (half*)__builtin_assume_aligned(A, 16);
  B = (half*)__builtin_assume_aligned(B, 16);
  C = (float*)__builtin_assume_aligned(C, 16);

  u4 a;
  u4 b;
  float4 c[2];

  int lane = threadIdx.x % WARP_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int tile_division = warp % (TILE_N / MACRO_N);
  int tile_level = warp / (TILE_N / MACRO_N);
  int tile_sub_level = tile_division;

  int thread_division = lane % (TENSOR_TILE_SIZE / VEC_SIZE);
  int thread_level = lane / (TENSOR_TILE_SIZE / VEC_SIZE);

  for(int i = 0; i < K / TILE_K; ++i){
    for(int j = 0; j < TILE_K / MACRO_K; ++j){
      for(int k = 0; k < TILE_M / MACRO_M; ++k){
        for(int l = 0; l < TILE_N / MACRO_N; ++l){

        }
      }
    }




    for(int j = 0; j < MACRO_N / TENSOR_TILE_SIZE; ++j){
      for(int k = 0; k < (MACRO_M / TENSOR_TILE_SIZE) / (TILE_N / MACRO_N); ++k){
        //A[tile_level][tile_sub_level][thread_level][thread_division].temp_u = *reinterpret_cast<uint4*>(&A[(by * TILE_M + tile_level * MACRO_M + tile_sub_level * TENSOR_TILE_SIZE + k * (MACRO_M / (TILE_N / MACRO_N)) * K + (i * TILE_K + j * TENSOR_TILE_SIZE))]);

      }
    }
  }
}




__global__ void naive_experimental_tensor_mat_mul_kernel_11(half* A, half* B, float* C, int M, int N, int K){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, half, row_major> b_frag[TILE_SIZE_6 / TENSOR_TILE_SIZE];
  fragment<accumulator, 16, 16, 16, float> c_frag[TILE_SIZE_6 / TENSOR_TILE_SIZE][TILE_SIZE_6 / TENSOR_TILE_SIZE];
  for(int i = 0; i < TILE_SIZE_6 / TENSOR_TILE_SIZE; ++i){
    for(int j = 0; j < TILE_SIZE_6 / TENSOR_TILE_SIZE; ++j){
      fill_fragment(c_frag[i][j], 0.0f);
    }
  }
  for(int i = 0; i < K / TILE_SIZE_6; ++i){
    for(int j = 0; j < TILE_SIZE_6 / TENSOR_TILE_SIZE; ++j){
      for(int k = 0; k < TILE_SIZE_6 / TENSOR_TILE_SIZE; ++k){
        load_matrix_sync(b_frag[k], &B[(i * TILE_SIZE_6 + j * TENSOR_TILE_SIZE) * N + (bx * TILE_SIZE_6 + k * TENSOR_TILE_SIZE)], N);
      }
      for(int k = 0; k < TILE_SIZE_6 / TENSOR_TILE_SIZE; ++k){
        load_matrix_sync(a_frag, &A[(by * TILE_SIZE_6 + k * TENSOR_TILE_SIZE) * K + (i * TILE_SIZE_6 + j * TENSOR_TILE_SIZE)], K);
        for(int l = 0; l < TILE_SIZE_6 / TENSOR_TILE_SIZE; ++l){
          mma_sync(c_frag[k][l], a_frag, b_frag[l], c_frag[k][l]);
        }
      }
    }
  }
  for(int i = 0; i < TILE_SIZE_6 / TENSOR_TILE_SIZE; ++i){
    for(int j = 0; j < TILE_SIZE_6 / TENSOR_TILE_SIZE; ++j){
      store_matrix_sync(&C[(by * TILE_SIZE_6 + i * TENSOR_TILE_SIZE) * N + (bx * TILE_SIZE_6 + j * TENSOR_TILE_SIZE)], c_frag[i][j], N, mem_row_major);
    }
  }

}




__global__ void figure_out_layout(half* A, half* B, float* C, int M, int N, int K){
  uint32_t a[2];
  uint32_t b[1];
  float c[4];

  int lane = threadIdx.x % WARP_SIZE;

  //Don't need to worry about memory access offset right now because shared memory is not incorporated

  int Amem_per_thread = (TENSOR_TILE_SIZE * TENSOR_TILE_SIZE / 2) / blockDim.x;
  int A_division = lane / TENSOR_TILE_SIZE;
  int A_level = lane & (TENSOR_TILE_SIZE - 1);

  int Bmem_per_thread = (TENSOR_TILE_SIZE / 2 * TENSOR_TILE_SIZE / 2) / blockDim.x;
  int B_division = lane / (TENSOR_TILE_SIZE / 2);
  int B_level = lane & (TENSOR_TILE_SIZE / 2 - 1);

  int Cmem_per_thread = (TENSOR_TILE_SIZE * TENSOR_TILE_SIZE / 2) / blockDim.x;
  int C_division = lane / TENSOR_TILE_SIZE;
  int C_level = lane & (TENSOR_TILE_SIZE - 1);

  a[0] = pack(A[A_level * K + A_division * Amem_per_thread + 0], A[A_level * K + A_division * Amem_per_thread + 1]);
  a[1] = pack(A[A_level * K + A_division * Amem_per_thread + 2], A[A_level * K + A_division * Amem_per_thread + 3]);

  b[0] = pack(B[B_level * N + B_division * Bmem_per_thread + 0], B[B_level * N + B_division * Bmem_per_thread + 1]);

  for(int i = 0; i < 4; ++i) c[i] = 0.0f;

  asm volatile(
    "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
    "{%0, %1, %2, %3}, "
    "{%4, %5}, "
    "{%6}, "
    "{%0, %1, %2, %3};\n"
    : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
    : "r"(a[0]), "r"(a[1]),
      "r"(b[0])
  );

  for(int i = 0; i < 4; ++i) C[C_level * N + C_division * Cmem_per_thread + i] = c[i];

}




float time_avg_us(cudaStream_t s, function<void()> launch, int iters){
  for(int i = 0; i < 1000; ++i){
    launch();
  }
  //cudaStreamSynchronize(s);
  cudaDeviceSynchronize();
  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, s);
  for(int i = 0; i < iters; ++i){
    launch();
  }
  cudaEventRecord(stop, s);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms * 1000.0f / iters;
}




int main() {
  printf("size of u4: %lu\n", sizeof(u4));
  int M = 1024;
  int N = 1024;
  int K = 1024;
  half* h_a = (half*)malloc((M * K) * sizeof(half));
  half* h_b = (half*)malloc((K * N) * sizeof(half));
  float* h_c = (float*)malloc((M * N) * sizeof(float));
  half* d_a; cudaMalloc((void**)&d_a, (M * K) * sizeof(half));
  half* d_b; cudaMalloc((void**)&d_b, (K * N) * sizeof(half));
  float* d_c; cudaMalloc((void**)&d_c, (M * N) * sizeof(float));
  for(int i = 0; i < M * K; ++i) h_a[i] = 1.0;
  for(int i = 0; i < K * N; ++i) h_b[i] = 2.0;
  for(int i = 0; i < M * N; ++i) h_c[i] = 0.0f;
  cudaMemcpy(d_a, h_a, (M * K) * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, (K * N) * sizeof(half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, (M * N) * sizeof(float), cudaMemcpyHostToDevice);
  cudaStream_t s;
  cudaStreamCreate(&s);
  float avg_time;


  //swap_test<<<1, 32>>>();
  //dim3 grid_dim(1), block_dim(32);
  //figure_out_layout<<<1, 32>>>(d_a, d_b, d_c, M, N, K);
  //dim3 grid_dim(N / TENSOR_TILE_SIZE, M / TENSOR_TILE_SIZE), block_dim(32);
  //asm_gemm<<<grid_dim, block_dim>>>(d_a, d_b, d_c, M, N, K);
  //dim3 grid_dim(N / (TENSOR_TILE_SIZE / 2), M / TENSOR_TILE_SIZE), block_dim(32);
  //asm_gemm_2<<<grid_dim, block_dim>>>(d_a, d_b, d_c, M, N, K);
  //dim3 grid_dim2(N / (TENSOR_TILE_SIZE / 2), M / TENSOR_TILE_SIZE), block_dim2(32);
  //asm_gemm_3<<<grid_dim2, block_dim2>>>(d_a, d_b, d_c, M, N, K);

  //dim3 grid_dim3(N / (TENSOR_TILE_SIZE / 2), M / TENSOR_TILE_SIZE), block_dim3(32);
  //asm_gemm_4<<<grid_dim3, block_dim3>>>(d_a, d_b, d_c, M, N, K);
  dim3 grid_dim4(N / MACRO_N, M / MACRO_M), block_dim4(32);
  asm_gemm_5<<<grid_dim4, block_dim4>>>(d_a, d_b, d_c, M, N, K);
  dim3 grid_dim6(N / MACRO_N, M / MACRO_M), block_dim6(32);
  asm_gemm_6<<<grid_dim6, block_dim6>>>(d_a, d_b, d_c, M, N, K);

  //dim3 grid_dim5(N / 16, M / 16), block_dim5(32);
  //naive_tensor_mat_mul_kernel<<<grid_dim5, block_dim5, 0, s>>>(d_a, d_b, d_c, M, N, K);
  //dim3 grid_dim17(N / TILE_SIZE_6, M / TILE_SIZE_6), block_dim17(32);
  //naive_experimental_tensor_mat_mul_kernel_11<<<grid_dim17, block_dim17, 0, s>>>(d_a, d_b, d_c, M, N, K);

  //dim3 grid_dim5(N / 16, M / 16), block_dim5(32);
  //avg_time = time_avg_us(s, [=]{naive_tensor_mat_mul_kernel<<<grid_dim5, block_dim5, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Naive tensor duration: " << avg_time << " us" << endl;

  //dim3 grid_dim(N / (TENSOR_TILE_SIZE / 2), M / TENSOR_TILE_SIZE), block_dim(32);
  //avg_time = time_avg_us(s, [=]{asm_gemm_2<<<grid_dim, block_dim, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Naive 2.0 PTX duration: " << avg_time << " us" << endl;

  //dim3 grid_dim2(N / (TENSOR_TILE_SIZE / 2), M / TENSOR_TILE_SIZE), block_dim2(32);
  //avg_time = time_avg_us(s, [=]{asm_gemm_3<<<grid_dim2, block_dim2, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Naive 3.0 PTX duration: " << avg_time << " us" << endl;

  //dim3 grid_dim3(N / (TENSOR_TILE_SIZE / 2), M / TENSOR_TILE_SIZE), block_dim3(32);
  //avg_time = time_avg_us(s, [=]{asm_gemm_4<<<grid_dim3, block_dim3, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Naive 4.0 PTX duration: " << avg_time << " us" << endl;

  //dim3 grid_dim4(N / MACRO_N, M / MACRO_M), block_dim4(32);
  //avg_time = time_avg_us(s, [=]{asm_gemm_5<<<grid_dim4, block_dim4, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 10000);
  //cout << "Naive 5.0 PTX duration: " << avg_time << " us" << endl;

  //dim3 grid_dim6(N / TILE_N, M / TILE_M), block_dim6((TILE_N / MACRO_N) * (TILE_M / MACRO_M) * WARP_SIZE);
  //avg_time = time_avg_us(s, [=]{asm_gemm_6<<<grid_dim6, block_dim6, 0, s>>>(d_a, d_b, d_c, M, N, K)};, 1000);
  //cout << "Naive 6.0 PTX duration: " << avg_time << " us" << endl;

  //dim3 grid_dim17(N / TILE_SIZE_6, M / TILE_SIZE_6), block_dim17(32);
  //avg_time = time_avg_us(s, [=]{naive_experimental_tensor_mat_mul_kernel_11<<<grid_dim17, block_dim17, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Naive experimental tensor 11.0 duration: " << avg_time << " us" << endl;

  cudaMemcpy(h_c, d_c, (M * N) * sizeof(float), cudaMemcpyDeviceToHost);

  /*
  for(int i = 0; i < M; ++i){
    for(int j = 0; j < N; ++j){
      printf("%f ", h_c[i * N + j]);
    }
    printf("\n");
  }
  */




  //free(h_a);
  //free(h_b);
  //free(h_c);
  //cudaFree(d_a);
  //cudaFree(d_b);
  //cudaFree(d_c);


  /*
  float *d_out;
  float h_out = 0.f;
  cudaMalloc(&d_out, sizeof(float));
  tc_test<<<1, 32>>>(d_out);
  cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  printf("Result = %f\n", h_out);
  cudaFree(d_out);
  */

  return 0;
}

