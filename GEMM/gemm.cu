#include <iostream>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <random>




using namespace std;
using namespace nvcuda::wmma;



#define TILE_SIZE 32
#define TILE_SIZE_2 32
#define TILE_SIZE_3 64
#define TILE_SIZE_4 32
#define TILE_SIZE_5 32
#define TILE_SIZE_6 64
#define TILE_A_LENGTH 64
#define TILE_A_HEIGHT 16
#define TILE_B_LENGTH 16
#define TILE_B_HEIGHT 16
#define TENSOR_TILE_SIZE 16
#define MINI_TILE_SIZE 2
#define MINI_TILE_SIZE_2 2
#define M_DIM 64
#define N_DIM 64
#define K_DIM 64
#define M_DIM_2 64
#define N_DIM_2 64
#define K_DIM_2 64
#define MACRO_DIM_M 32
#define MACRO_DIM_N 64
#define MACRO_DIM_K 32
#define MICRO_DIM_M 32
#define MICRO_DIM_N 32
#define MICRO_DIM_K 32
#define WARP_SIZE 32
#define MICRO_TILE_SIZE 32
#define MACRO_TILE_SIZE 64
#define MICRO_TILE_ROWS 2
#define MICRO_TILE_COLS 4
#define UNROLL_FACTOR 8
#define SM_PAD 1
#define HALF_SIZE 2
#define MEMORY_BANKS 32
#define MEMORY_BANK_SIZE 4
#define PARTIALS 8
#define MULT 1



typedef struct __align__(8) half4{
  __half x, y, z, w;
}half4;




__device__ uint32_t rand_gen(uint32_t& state){
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}




__global__ void test_kernel(half* A, half* B, float* C, int M, int N, int K){
  __shared__ float product_buff[16][16];
  int x = (threadIdx.x % (16 * 16)) % 16;
  int y = (threadIdx.x % (16 * 16)) / 16;
  int global_x = blockIdx.x * 16 + x;
  int global_y = blockIdx.y * 16 + y;
  int k_segment = threadIdx.x / (16 * 16);
  int segment_len = K / (blockDim.x / (16 * 16));
  int start = k_segment * segment_len;
  int end = start + segment_len;
  float acc = 0.0f;
  if(threadIdx.x < 256){
    product_buff[y][x] = 0.0f;
  }
  __syncthreads();
  for(int i = start; i < end; ++i){
    acc += __half2float(A[y * K + i]) * __half2float(B[i * N + x]);
  }
  atomicAdd(&(product_buff[y][x]), acc);
  __syncthreads();
  C[global_y * N + global_x] = product_buff[y][x];

}




__global__ void upgraded_tiled_gemm(const __half* a, const __half* b, float* c, int M, int N, int K){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int read;
  int write;
  float acc = 0.0f;
  __shared__ __half tile_a[2][TILE_SIZE][TILE_SIZE];
  __shared__ __half tile_b[2][TILE_SIZE][TILE_SIZE];
  __pipeline_memcpy_async(&tile_a[0][ty][tx], &a[(by * TILE_SIZE + ty) * K + (0 * TILE_SIZE + tx)], sizeof(__half));
  __pipeline_memcpy_async(&tile_b[0][ty][tx], &b[(0 * TILE_SIZE + ty) * N + (bx * TILE_SIZE + tx)], sizeof(__half));
  __pipeline_commit();
  __pipeline_wait_prior(0);
  __syncthreads();
  for(int i = 0; i < K / TILE_SIZE - 1; ++i){
    read = i % 2;
    write = (i + 1) % 2;

    __pipeline_memcpy_async(&tile_a[write][ty][tx], &a[(by * TILE_SIZE + ty) * K + (i * TILE_SIZE + tx)], sizeof(__half));
    __pipeline_memcpy_async(&tile_b[write][ty][tx], &b[(i * TILE_SIZE + ty) * N + (bx * TILE_SIZE + tx)], sizeof(__half));
    __pipeline_commit();

    for(int j = 0; j < TILE_SIZE; ++j){
      //acc = fmaf(tile_a[read][ty][j], tile_b[read][j][tx], acc);
      acc += __half2float(tile_a[read][ty][j]) * __half2float(tile_b[read][j][tx]);
    }

    __pipeline_wait_prior(0);
    __syncthreads();
  }
  read = (read + 1) % 2;
  for(int j = 0; j < TILE_SIZE; ++j){
    acc += __half2float(tile_a[read][ty][j]) * __half2float(tile_b[read][j][tx]);
  }
  __syncthreads();
  printf("acc: %f\n", acc);
  c[(by * TILE_SIZE + ty) * N + (bx * TILE_SIZE + tx)] = acc;
}




__global__ void warp_shfl_gemm(__half* A, __half* B, float* C, int M, int N, int K){
	int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
	float a_val;
  float b_val;
	float acc = 0.0f;
	for(int i = 0; i < K / WARP_SIZE; ++i){
    a_val = __half2float(A[(by * WARP_SIZE + ty) * K + i * WARP_SIZE + tx]);
    b_val = __half2float(B[(i * WARP_SIZE + ty) * N + bx * WARP_SIZE + tx]);
    for(int j = 0; j < WARP_SIZE; ++j){
      acc = fmaf(__shfl_sync(0xFFFFFFFF, a_val, j), __shfl_sync(0xFFFFFFFF, b_val, j), acc);
    }
  }
  C[(by * WARP_SIZE + ty) * N + bx * WARP_SIZE + tx] = acc;
}




__global__ void warp_tiled_gemm(const __half* a, const __half* b, float* c, int M, int N, int K){
  __shared__ __half tile_a[TILE_SIZE_2 * TILE_SIZE_2];
  __shared__ __half tile_b[TILE_SIZE_2 * TILE_SIZE_2];

  int tile = threadIdx.x / WARP_SIZE;
  int tile_col = tile % (N / TILE_SIZE);
  int tile_row = tile / (M / TILE_SIZE);

  int pos_in_tile = threadIdx.x % WARP_SIZE;

  int col = pos_in_tile / (TILE_SIZE / MICRO_TILE_ROWS);
  int row = pos_in_tile - (col * (TILE_SIZE / MICRO_TILE_ROWS));

  int pos_in_a = tile_row * TILE_SIZE + row * MICRO_TILE_ROWS * N;
  int pos_in_b0 = tile_col * TILE_SIZE + col * MICRO_TILE_COLS;
  int pos_in_c = (tile_row * TILE_SIZE + row * MICRO_TILE_ROWS) * N + (tile_col * TILE_SIZE + col * MICRO_TILE_COLS);

  //printf("thread Idx: %d, tile col: %d, tile_row: %d, col: %d, row: %d\n", threadIdx.x, tile_col, tile_row, col, row);

  float c0, c1, c2, c3, c4, c5, c6, c7;
  c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = 0;
  float a0, a1;
  __half bx, by, bz, bw;

  *reinterpret_cast<half4*>(tile_a + pos_in_c) = *reinterpret_cast<const half4*>(a + pos_in_c);
  *reinterpret_cast<half4*>(tile_a + pos_in_c + K) = *reinterpret_cast<const half4*>(a + pos_in_c + K);
  *reinterpret_cast<half4*>(tile_b + pos_in_c) = *reinterpret_cast<const half4*>(b + pos_in_c);
  *reinterpret_cast<half4*>(tile_b + pos_in_c + N) = *reinterpret_cast<const half4*>(b + pos_in_c + N);
  __syncthreads();

  for(int i = 0; i < K; ++i){
    a0 = __half2float(tile_a[pos_in_a + i]);
    a1 = __half2float(tile_a[pos_in_a + K + i]);

    half4 b0 = *reinterpret_cast<const half4*>(tile_b + i * N + pos_in_b0);
    bx = __half2float(b0.x);
    by = __half2float(b0.y);
    bz = __half2float(b0.z);
    bw = __half2float(b0.w);


    c0 = fmaf(a0, bx, c0);
    c1 = fmaf(a0, by, c1);
    c2 = fmaf(a0, bz, c2);
    c3 = fmaf(a0, bw, c3);
    c4 = fmaf(a1, bx, c4);
    c5 = fmaf(a1, by, c5);
    c6 = fmaf(a1, bz, c6);
    c7 = fmaf(a1, bw, c7);

  }

  *reinterpret_cast<float4*>(c + pos_in_c) = make_float4(c0, c1, c2, c3);
  *reinterpret_cast<float4*>(c + pos_in_c + N) = make_float4(c4, c5, c6, c7);
}




__global__ void warp_gemm(const __half* a, const __half* b, float* c, int M, int N, int K){

  int tile = threadIdx.x / WARP_SIZE;
  int tile_col = tile % (N / TILE_SIZE);
  int tile_row = tile / (M / TILE_SIZE);

  int pos_in_tile = threadIdx.x % WARP_SIZE;

  int col = pos_in_tile / (TILE_SIZE / MICRO_TILE_ROWS);
  int row = pos_in_tile - (col * (TILE_SIZE / MICRO_TILE_ROWS));

  int pos_in_a = tile_row * TILE_SIZE + row * MICRO_TILE_ROWS * N;
  int pos_in_b0 = tile_col * TILE_SIZE + col * MICRO_TILE_COLS;
  int pos_in_c = (tile_row * TILE_SIZE + row * MICRO_TILE_ROWS) * N + (tile_col * TILE_SIZE + col * MICRO_TILE_COLS);

  //printf("thread Idx: %d, tile col: %d, tile_row: %d, col: %d, row: %d\n", threadIdx.x, tile_col, tile_row, col, row);

  float c0, c1, c2, c3, c4, c5, c6, c7;
  c0 = c1 = c2 = c3 = c4 = c5 = c6 = c7 = 0;
  float a0, a1;
  __half bx, by, bz, bw;

  for(int i = 0; i < K; ++i){

    a0 = __half2float(a[pos_in_a + i]);
    a1 = __half2float(a[pos_in_a + K + i]);

    //float4 b0 = *reinterpret_cast<const float4*>(b + i * N + pos_in_b0);
    half4 b0 = *reinterpret_cast<const half4*>(b + i * N + pos_in_b0);
    bx = __half2float(b0.x);
    by = __half2float(b0.y);
    bz = __half2float(b0.z);
    bw = __half2float(b0.w);


    c0 = fmaf(a0, bx, c0);
    c1 = fmaf(a0, by, c1);
    c2 = fmaf(a0, bz, c2);
    c3 = fmaf(a0, bw, c3);
    c4 = fmaf(a1, bx, c4);
    c5 = fmaf(a1, by, c5);
    c6 = fmaf(a1, bz, c6);
    c7 = fmaf(a1, bw, c7);

  }

  *reinterpret_cast<float4*>(c + pos_in_c) = make_float4(c0, c1, c2, c3);
  *reinterpret_cast<float4*>(c + pos_in_c + N) = make_float4(c4, c5, c6, c7);

}



__global__ void tiled_gemm(const __half* a, const __half* b, float* c, int M, int N, int K){
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
  __shared__ float tile_b[TILE_SIZE][TILE_SIZE];
  float acc = 0.0f;
  for(int i = 0; i < K / TILE_SIZE; ++i){
    tile_a[ty][tx] = __half2float(a[(by * TILE_SIZE + ty) * K + (i * TILE_SIZE + tx)]);
    tile_b[ty][tx] = __half2float(b[(i * TILE_SIZE + ty) * N + (bx * TILE_SIZE + tx)]);
    __syncthreads();
    for(int j = 0; j < TILE_SIZE; ++j){
      //acc += __half2float(tile_a[ty][j]) * __half2float(tile_b[j][tx]);
      //acc += tile_a[ty][j] * tile_b[j][tx];
      acc = fmaf(tile_a[ty][j], tile_b[j][tx], acc);
    }
    __syncthreads();
  }
  c[(by * TILE_SIZE + ty) * N + (bx * TILE_SIZE + tx)] = acc;
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




__global__ void naive_experimental_tensor_mat_mul_kernel(half* A, half* B, float* C, int M, int N, int K){
  //IMPORTANT NOTE: Have different waprs calculate different portions of the product matrix
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int warp = threadIdx.x / WARP_SIZE;
  int warps = blockDim.x / WARP_SIZE;
  int tile_tensor_dim = TILE_SIZE / TENSOR_TILE_SIZE;
  int tensors_per_tile = tile_tensor_dim * tile_tensor_dim;
  int division = threadIdx.x / (WARP_SIZE * tensors_per_tile);
  int divisions = blockDim.x / (WARP_SIZE * tensors_per_tile);
  int division_len = (K / TILE_SIZE)  / divisions;
  int ttx = warp % tile_tensor_dim;
  int tty = (warp % tensors_per_tile) / tile_tensor_dim;
  int start = division * division_len;
  int end = start + division_len;
  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
  //2 because 32 / 16 = 2
  __shared__ fragment<accumulator, 16, 16, 16, float> c_frag[2][2];
  __shared__ int ready_flags[2][2];
  fill_fragment(c_frag[tty][ttx], 0.0f);
  ready_flags[tty][ttx] = 0;
  //fill_fragment(c_frag, 0.0f);
  for(int i = start; i < end; ++i){
    for(int j = 0; j < TILE_SIZE / 16; ++j){
      load_matrix_sync(a_frag, &A[(by * TILE_SIZE + tty * 16) * K + (i * TILE_SIZE + ttx * 16)], K);
      load_matrix_sync(b_frag, &B[(i * TILE_SIZE + tty * 16) * K + (bx * TILE_SIZE + ttx * 16)], N);
      while(ready_flags[tty][ttx] == 1) __nanosleep(1);
      atomicAdd(&ready_flags[tty][ttx], 1);
      mma_sync(c_frag[tty][ttx], a_frag, b_frag, c_frag[tty][ttx]);
      atomicSub(&ready_flags[tty][ttx], 1);
    }
  }
  store_matrix_sync(&C[(by * TILE_SIZE + tty * TENSOR_TILE_SIZE) * N + (bx * TILE_SIZE + ttx * TENSOR_TILE_SIZE)], c_frag[tty][ttx], N, mem_row_major);



  /*
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int warp = threadIdx.x / WARP_SIZE;
  int warps = blockDim.x / WARP_SIZE;
  int tensors_per_tile = (TILE_SIZE / TENSOR_TILE_SIZE) * (TILE_SIZE / TENSOR_TILE_SIZE);
  int division = threadIdx.x / (WARP_SIZE * tensors_per_tile);
  int divisions = blockDim.x / (WARP_SIZE * tensors_per_tile);
  int division_len = (K / TILE_SIZE)  / divisions;
  int ttx = warp % (TILE_SIZE / TENSOR_TILE_SIZE);
  int tty = (warp % tensors_per_tile) / (TILE_SIZE / TENSOR_TILE_SIZE);
  int start = division * division_len;
  int end = start + division_len;
  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag;
  fill_fragment(c_frag, 0.0f);
  for(int i = start; i < end; ++i){
    for(int j = 0; j < TILE_SIZE / 16; ++j){
      load_matrix_sync(a_frag, &A[(by * TILE_SIZE + tty * 16) * K + (i * TILE_SIZE + ttx * 16)], K);
      load_matrix_sync(b_frag, &B[(i * TILE_SIZE + tty * 16) * K + (bx * TILE_SIZE + ttx * 16)], N);
      mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
  }
  //store_matrix_sync(C[]);


  __shared__ half tile_a[TILE_SIZE][TILE_SIZE];
  __shared__ half tile_b[TILE_SIZE][TILE_SIZE];
  __shared__ int ready;
  float temp_buf[16][16];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x % TILE_SIZE;
  int ty = threadIdx.x / TILE_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int warps = blockDim.x / WARP_SIZE;
  int tiles_per_warp = (K / TILE_SIZE) / warps;
  int start_tile = warp * tiles_per_warp;
  int end_tile = start_tile + tiles_per_warp;
  ready = 1;
  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag;
  fill_fragment(c_frag, 0.0f);
  for(int i = start_tile; i < end_tile; ++i){
    tile_a[ty][tx] = A[(by * TILE_SIZE + ty) * K + (i * TILE_SIZE + tx)];
    tile_b[ty][tx] = B[(i * TILE_SIZE + ty) * N + (bx * TILE_SIZE + tx)];
    __syncthreads();
    for(int j = 0; j < TILE_SIZE / 16; ++j){
      load_matrix_sync(a_frag, &(tile_a[(ty / 16) * 16][j * 16]), TILE_SIZE);
      load_matrix_sync(b_frag, &(tile_b[j * 16][tx / 16]), TILE_SIZE);
      mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    __syncthreads();
  }
  //__syncthreads();
  while(ready == 0) __nanosleep(1000);
  atomicAdd(&ready, 1);
  atomicSub(&ready, 1);
  */
}




__global__ void naive_experimental_tensor_mat_mul_kernel_2(half* A, half* B, float* C, int M, int N, int K){
  //IMPORTANT NOTE: Have different waprs calculate different portions of the product matrix
  __shared__ half tile_a[TILE_SIZE][TILE_SIZE];
  __shared__ half tile_b[TILE_SIZE][TILE_SIZE];
  __shared__ float product_tile[16][16];
  float temp_buff[16][16];
  int mem_size = (TILE_SIZE * TILE_SIZE) / blockDim.x;
  int tile_pos = threadIdx.x * mem_size;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = tile_pos % TILE_SIZE;
  int ty = tile_pos / TILE_SIZE;
  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag;
  fill_fragment(c_frag, 0.0f);
  for(int i = 0; i < K / TILE_SIZE; ++i){
    for(int j = 0; j < mem_size; ++j){
      tile_a[ty][tx + j] = A[(by * TILE_SIZE + ty) * K + (i * TILE_SIZE + tx + j)];
      tile_b[ty][tx + j] = B[(i * TILE_SIZE + ty) * N + (bx * TILE_SIZE + tx + j)];
      //printf("%f\n", __half2float(tile_a[ty][tx + j]));
      //printf("%f\n", __half2float(tile_b[ty][tx + j]));
    }
    __syncthreads();
    for(int j = 0; j < TILE_SIZE / 16; ++j){
      load_matrix_sync(a_frag, &(tile_a[(ty / 16) * 16][j * 16]), TILE_SIZE);
      load_matrix_sync(b_frag, &(tile_b[j * 16][(tx / 16) * 16]), TILE_SIZE);
      mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    __syncthreads();
  }
  __syncthreads();
  store_matrix_sync(&C[(by * TILE_SIZE + (ty / 16) * 16) * N + (bx * TILE_SIZE + (tx / 16) * 16)], c_frag, N, mem_row_major);
  __syncthreads();

  /*
  __syncthreads();
  if(threadIdx.x < 32){
    store_matrix_sync(&(product_tile[0][0]), c_frag, 16, mem_row_major);
  }
  __syncthreads();
  */

  /*
  if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0){
    printf("Hello from kernel, thread 0\n");
    for(int i = 0; i < 16; ++i){
      for(int j = 0; j < 16; ++j){
        printf("%f, ", C[(by * TILE_SIZE + (ty / 16) * 16 + i) * N + (bx * TILE_SIZE + (tx / 16) * 16 + j)]);
        //printf("%f, ", product_tile[i][j]);
      }
      printf("\n");
    }
  }
  */
}




__global__ void naive_experimental_tensor_mat_mul_kernel_3(half* A, half* B, float* C, int M, int N, int K){
  //IMPORTANT NOTE: Have different waprs calculate different portions of the product matrix
  //IMPORTANT NOTE: Take note of where the number "8" is located
  //IMPORTANT NOTE: USE SHARED FRAGMENTS, for matrix a and matrix b, reduce the number of loads.
  //IMPROTANT NOTE: Shared memory is not actually necessary, can just add values straight from fragment into global memory representing C matrix
  //transformer guided innovation with a GNN
  __shared__ float partials[PARTIALS][TILE_SIZE][TILE_SIZE];
  //__shared__ float acc[TILE_SIZE][TILE_SIZE];
  //__shared__ float add[TILE_SIZE][TILE_SIZE];
  //__shared__ int ready_flags[TILE_TENSOR_DIM][TILE_TENSOR_DIM];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x % TILE_SIZE;
  int ty = threadIdx.x / TILE_SIZE;

  int tensor_tile_pos_x = (threadIdx.x % WARP_SIZE) / 16;
  int tensor_tile_pos_y = (threadIdx.x % WARP_SIZE) & 15;

  int warp = threadIdx.x / WARP_SIZE;
  int warps = blockDim.x / WARP_SIZE;
  int tile_tensor_dim = TILE_SIZE / TENSOR_TILE_SIZE;
  int tensors_per_tile = tile_tensor_dim * tile_tensor_dim;
  int division = threadIdx.x / (WARP_SIZE * tensors_per_tile);
  int divisions = blockDim.x / (WARP_SIZE * tensors_per_tile);
  int division_len = (K / TILE_SIZE)  / divisions;
  //int ttx = warp % TILE_TENSOR_DIM;
  //int tty = (warp % tensors_per_tile) / TILE_TENSOR_DIM;
  int ttx = warp % 2;
  int tty = (warp % tensors_per_tile) / 2;
  int start = division * division_len;
  int end = start + division_len;
  float acc = 0.0;
  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag;
  fill_fragment(c_frag, 0.0f);
  for(int i = start; i < end; ++i){
    for(int j = 0; j < TILE_SIZE / 16; ++j){
      load_matrix_sync(a_frag, &A[(by * TILE_SIZE + tty * 16) * K + (i * TILE_SIZE + j * 16)], K);
      load_matrix_sync(b_frag, &B[(i * TILE_SIZE + j * 16) * K + (bx * TILE_SIZE + ttx * 16)], N);
      mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
  }

  for(int i = 0; i < PARTIALS; ++i){

  }

  /*
  store_matrix_sync(&(partials[division][tty * TENSOR_TILE_SIZE][ttx * TENSOR_TILE_SIZE]), c_frag, TILE_SIZE, mem_row_major);
  __syncthreads();
  for(int i = 0; i < PARTIALS; ++i){
    acc += partials[i][ty][tx];
  }
  C[(by * TILE_SIZE + ty) * N + (bx * TILE_SIZE + tx)] = acc;
  */
}




__global__ void naive_experimental_tensor_mat_mul_kernel_4(half* A, half* B, float* C, int M, int N, int K){
  __shared__ float partials[1][TILE_SIZE][TILE_SIZE];
  //__shared__ float acc[TILE_SIZE][TILE_SIZE];
  //__shared__ float add[TILE_SIZE][TILE_SIZE];
  //__shared__ int ready_flags[TILE_TENSOR_DIM][TILE_TENSOR_DIM];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x % TILE_SIZE;
  int ty = threadIdx.x / TILE_SIZE;
  int warp = threadIdx.x / WARP_SIZE;
  int warps = blockDim.x / WARP_SIZE;
  int tile_tensor_dim = TILE_SIZE / TENSOR_TILE_SIZE;
  int tensors_per_tile = tile_tensor_dim * tile_tensor_dim;
  int division = threadIdx.x / (WARP_SIZE * tensors_per_tile);
  int divisions = blockDim.x / (WARP_SIZE * tensors_per_tile);
  int division_len = (K / TILE_SIZE)  / divisions;
  //int ttx = warp % TILE_TENSOR_DIM;
  //int tty = (warp % tensors_per_tile) / TILE_TENSOR_DIM;
  int ttx = warp % 2;
  int tty = (warp % tensors_per_tile) / 2;
  int start = division * division_len;
  int end = start + division_len;
  float acc = 0.0;
  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag;
  fill_fragment(c_frag, 0.0f);
  for(int i = start; i < end; ++i){
    for(int j = 0; j < TILE_SIZE / 16; ++j){
      load_matrix_sync(a_frag, &A[(by * TILE_SIZE + tty * 16) * K + (i * TILE_SIZE + j * 16)], K);
      load_matrix_sync(b_frag, &B[(i * TILE_SIZE + j * 16) * N + (bx * TILE_SIZE + ttx * 16)], N);
      mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
  }
  __syncthreads();
  for(int i = 0; i < TENSOR_TILE_SIZE * TENSOR_TILE_SIZE; ++i){
    //C[(by * TILE_SIZE + ty) * N + (bx * TILE_SIZE + tx)] +=
  }

  store_matrix_sync(&(partials[division][tty * TENSOR_TILE_SIZE][ttx * TENSOR_TILE_SIZE]), c_frag, TILE_SIZE, mem_row_major);
  for(int i = 0; i < 8; ++i){
    acc += (partials[i][ty][tx]);
  }
  C[(by * TILE_SIZE + ty) * N + (bx * TILE_SIZE + tx)] = acc;

  /*

  //store_matrix_sync(&(partials[division][tty * TENSOR_TILE_SIZE][ttx * TENSOR_TILE_SIZE]), c_frag, TILE_SIZE, mem_row_major);
  for(int i = 0; i < TENSOR_TILE_SIZE * TENSOR_TILE_SIZE; ++i){
    C[(by * TILE_SIZE + ty) * N + (bx * TILE_SIZE + tx)]
    //partials[division][i / TENSOR_TILE_SIZE][i % TENSOR_TILE_SIZE] = c_frag.x[i];
    //*reinterpret_cast<float4*>(&(partials[division][i / TENSOR_TILE_SIZE][i % TENSOR_TILE_SIZE])) = *reinterpret_cast<float4*>(&(c_frag.x[i]));
  }

  __syncthreads();

  for(int i = 0; i < 8; ++i){
    acc += (partials[i][ty][tx]);
  }
  C[(by * TILE_SIZE + ty) * N + (bx * TILE_SIZE + tx)] = acc;
  */
}

__global__ void naive_experimental_tensor_mat_mul_kernel_5(half* A, half* B, float* C, int M, int N, int K){
  __shared__ float partials[1][TILE_SIZE_2][TILE_SIZE_2];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x % TILE_SIZE_2;
  int ty = threadIdx.x / TILE_SIZE_2;
  int tensor_tile_pos_x = (threadIdx.x % WARP_SIZE) / 16;
  int tensor_tile_pos_y = (threadIdx.x % WARP_SIZE) & 15;
  int warp = threadIdx.x / WARP_SIZE;
  int warps = blockDim.x / WARP_SIZE;
  int tile_tensor_dim = TILE_SIZE_2 / TENSOR_TILE_SIZE;

  int tensors_per_tile = tile_tensor_dim * tile_tensor_dim;

  int division = threadIdx.x / (WARP_SIZE * tensors_per_tile);
  int divisions = blockDim.x / (WARP_SIZE * tensors_per_tile);
  int division_len = (K / TILE_SIZE_2)  / divisions;
  int ttx = warp % tile_tensor_dim;
  int tty = (warp % tensors_per_tile) / tile_tensor_dim;
  int start = division * division_len;
  int end = start + division_len;
  float acc = 0.0;
  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag;
  fill_fragment(c_frag, 0.0f);
  for(int i = start; i < end; ++i){
    for(int j = 0; j < TILE_SIZE_2 / 16; ++j){
      load_matrix_sync(a_frag, &A[(by * TILE_SIZE_2 + tty * 16) * K + (i * TILE_SIZE_2 + j * 16)], K);
      load_matrix_sync(b_frag, &B[(i * TILE_SIZE_2 + j * 16) * N + (bx * TILE_SIZE_2 + ttx * 16)], N);
      mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
  }
  store_matrix_sync(&C[(by * TILE_SIZE_2 + tty * TENSOR_TILE_SIZE) * N + (bx * TILE_SIZE_2 + ttx * TENSOR_TILE_SIZE)], c_frag, N, mem_row_major);



  /*
  store_matrix_sync(&(partials[division][tty * TENSOR_TILE_SIZE][ttx * TENSOR_TILE_SIZE]), c_frag, TILE_SIZE_2, mem_row_major);
  __syncthreads();
  for(int i = 0; i < 1; ++i){
    acc += partials[i][ty][tx];
  }
  */

}




__global__ void naive_experimental_tensor_mat_mul_kernel_6(half* A, half* B, float* C, int M, int N, int K){
  //__shared__ float partials[2][TILE_SIZE_2][TILE_SIZE_2];
  //int indices_per_thread = (TILE_SIZE_2 * TILE_SIZE_2) / blockDim.x;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x % TILE_SIZE_2;
  int ty = threadIdx.x / TILE_SIZE_3;
  int warp = threadIdx.x / WARP_SIZE;
  int warps = blockDim.x / WARP_SIZE;
  int tile_tensor_dim = TILE_SIZE_2 / TENSOR_TILE_SIZE;

  //int tensors_per_tile = TILE_TENSOR_DIM * TILE_TENSOR_DIM;

  int tensors_per_tile = tile_tensor_dim * tile_tensor_dim;

  int division = threadIdx.x / (WARP_SIZE * tensors_per_tile);
  int divisions = blockDim.x / (WARP_SIZE * tensors_per_tile);
  int division_len = (K / TILE_SIZE_2)  / divisions;
  int ttx = warp % tile_tensor_dim;
  int tty = (warp % tensors_per_tile) / tile_tensor_dim;
  int start = division * division_len;
  int end = start + division_len;
  float acc = 0.0;
  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag;
  fill_fragment(c_frag, 0.0f);
  for(int i = start; i < end; ++i){
    for(int j = 0; j < TILE_SIZE_2 / 16; ++j){
      load_matrix_sync(a_frag, &A[(by * TILE_SIZE_2 + tty * 16) * K + (i * TILE_SIZE_2 + ttx * 16)], K);
      load_matrix_sync(b_frag, &B[(i * TILE_SIZE_2 + tty * 16) * N + (bx * TILE_SIZE_2 + ttx * 16)], N);
      mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
  }
  store_matrix_sync(&C[(by * TILE_SIZE_2 + tty * TENSOR_TILE_SIZE) * N + (bx * TILE_SIZE_2 + ttx * TENSOR_TILE_SIZE)], c_frag, N, mem_row_major);

  /*
  store_matrix_sync(&(partials[division][tty * TENSOR_TILE_SIZE][ttx * TENSOR_TILE_SIZE]), c_frag, TILE_SIZE_2, mem_row_major);
  for(int i = 0; i < indices_per_thread; ++i){
    for(int j = 0; j < 2; ++j){
      acc += partials[j][ty][tx + i];
    }
    C[(by * TILE_SIZE_2 + ty) * N + (bx * TILE_SIZE_2 + tx + i)] = acc;
    acc = 0.0f;
  }
  */
}




__global__ void naive_experimental_tensor_mat_mul_kernel_7(half* A, half* B, float* C, int M, int N, int K){
  //NOTE: If one group of threads in one warp is trying to access one section of memory, but then the scheduler switches to another warp that tries to access the same
  //section of memory there may be some kind of delay, warps waiting in some kind of queue to access same section of memory, thus create some kind of offset, for example
  //lets say we have a sequence of 4 memory sections: [0, 1, 2, 3] and 4 warps: [0, 1, 2, 3]. All 4 warps might try to access section 0, 1, 2, 3 in that order. Thus it might
  //be a good idea to set up a system where the following warps access memory in the following respective order: 0: [0, 1, 2, 3], 1: [1, 2, 3, 0], 2: [2, 3, 0, 1], 3: [3, 0, 1, 2]
  //printf("Hello from kernel 7.0\n");
  __shared__ __half tile_a[TILE_SIZE_2][TILE_SIZE_2];
  __shared__ __half tile_b[TILE_SIZE_2][TILE_SIZE_2];
  __shared__ int flag;
  int mem_per_thread = 8;
  int mem_per_warp = mem_per_thread * WARP_SIZE;
  int unit_size = 2;
  int units_per_bank = MEMORY_BANK_SIZE / unit_size;
  int warp = threadIdx.x / WARP_SIZE;
  int tensor_tile_size = TILE_SIZE_2 / TENSOR_TILE_SIZE;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int ttx = warp % tensor_tile_size;
  int tty = warp / tensor_tile_size;
  int sub_tile_length = (MEMORY_BANKS >= TILE_SIZE_2 / units_per_bank) ? TILE_SIZE_2 : MEMORY_BANKS;
  int sub_tile_height = (TENSOR_TILE_SIZE * TENSOR_TILE_SIZE) / sub_tile_length;
  int wx = warp % (TILE_SIZE_2 / sub_tile_length);
  int wy = warp / (TILE_SIZE_2 / sub_tile_length);

  int lane = threadIdx.x % WARP_SIZE;
  int level = lane & (sub_tile_height - 1);
  int division = lane / sub_tile_height;
  mem_per_thread = (TENSOR_TILE_SIZE * TENSOR_TILE_SIZE) / WARP_SIZE;
  //Add double cast for variable values that might not neatly divide
  int boost = (mem_per_thread / sub_tile_height > 0) ? mem_per_thread / sub_tile_height : 1;
  int start_pos = (level * boost) % mem_per_thread;

  fragment<matrix_a, 16, 16, 16, __half, row_major> frag_a;
  fragment<matrix_b, 16, 16, 16, __half, row_major> frag_b;
  fragment<accumulator, 16, 16, 16, float> frag_c;
  fill_fragment(frag_c, 0.0f);
  flag = 0;

  for(int i = 0; i < K / TILE_SIZE_2; ++i){
    for(int j = start_pos; j < start_pos + mem_per_thread; ++j){
      tile_a[wy * sub_tile_height + level][wx * sub_tile_length + division * mem_per_thread + (j % mem_per_thread)] = A[(by * TILE_SIZE_2 + wy * sub_tile_height + level) * K + i * TILE_SIZE_2 + wx * sub_tile_length + division * mem_per_thread + (j % mem_per_thread)];
      tile_b[wy * sub_tile_height + level][wx * sub_tile_length + division * mem_per_thread + (j % mem_per_thread)] = B[(i * TILE_SIZE_2 + wy * sub_tile_height + level) * N + bx * TILE_SIZE_2 + wx * sub_tile_length + division * mem_per_thread + (j % mem_per_thread)];
      int memory_bank = (((wy * sub_tile_height + level) * TILE_SIZE_2 + wx * sub_tile_length + division * mem_per_thread + (j % mem_per_thread)) * 2 / 4) % 32;
    }
    __syncthreads();
    for(int j = 0; j < TILE_SIZE_2 / TENSOR_TILE_SIZE; ++j){
      load_matrix_sync(frag_a, &(tile_a[tty * TENSOR_TILE_SIZE][j * TENSOR_TILE_SIZE]), TILE_SIZE_2);
      load_matrix_sync(frag_b, &(tile_b[j * TENSOR_TILE_SIZE][ttx * TENSOR_TILE_SIZE]), TILE_SIZE_2);
      mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    __syncthreads();
  }
  store_matrix_sync(&C[(by * TILE_SIZE_2 + tty * TENSOR_TILE_SIZE) * N + bx * TILE_SIZE_2 + ttx * TENSOR_TILE_SIZE], frag_c, N, mem_row_major);
}





__global__ void naive_experimental_tensor_mat_mul_kernel_8(half* A, half* B, float* C, int M, int N, int K){
  __shared__ __half tile_a[2][TILE_SIZE][TILE_SIZE];
  __shared__ __half tile_b[2][TILE_SIZE][TILE_SIZE];
  __shared__ float producer_flag;
  __shared__ float consumer_flag;
  producer_flag = 0.0f;
  consumer_flag = 0.0f;
  __syncthreads();
  int unit_size = 2;
  int units_per_bank = MEMORY_BANK_SIZE / unit_size;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tensor_tile_size = TILE_SIZE / TENSOR_TILE_SIZE;
  int tensor_per_tile = tensor_tile_size * tensor_tile_size;
  int warp = (threadIdx.x % (WARP_SIZE * tensor_per_tile)) / WARP_SIZE;
  int ttx = warp % tensor_tile_size;
  int tty = warp / tensor_tile_size;

  int sub_tile_length = (MEMORY_BANKS >= TILE_SIZE / units_per_bank) ? TILE_SIZE : MEMORY_BANKS;
  int sub_tile_height = (TENSOR_TILE_SIZE * TENSOR_TILE_SIZE) / sub_tile_length;
  int wx = warp % (TILE_SIZE_2 / sub_tile_length);
  int wy = warp / (TILE_SIZE_2 / sub_tile_length);

  int lane = threadIdx.x % WARP_SIZE;
  int level = lane & (sub_tile_height - 1);
  int division = lane / sub_tile_height;
  int mem_per_thread = (TENSOR_TILE_SIZE * TENSOR_TILE_SIZE) / WARP_SIZE;
  int boost = (mem_per_thread / sub_tile_height > 0) ? mem_per_thread / sub_tile_height : 1;
  int start_pos = (level * boost) % mem_per_thread;

  int producer_exch_num;
  int consumer_exch_num;

  if(threadIdx.x < WARP_SIZE * tensor_per_tile){
    for(int j = start_pos; j < mem_per_thread + start_pos; ++j){
      tile_a[0][wy * sub_tile_height + level][wx * sub_tile_length + division * mem_per_thread + (j % mem_per_thread)] = A[(by * TILE_SIZE + wy * sub_tile_height + level) * K + (0 * TILE_SIZE + wx * sub_tile_length + division * mem_per_thread + (j % mem_per_thread))];
      tile_b[0][wy * sub_tile_height + level][wx * sub_tile_length + division * mem_per_thread + (j % mem_per_thread)] = B[(0 * TILE_SIZE + wy * sub_tile_height + level) * N + (bx * TILE_SIZE + wx * sub_tile_length + division * mem_per_thread + (j % mem_per_thread))];
    }
    if(threadIdx.x < 1){
      atomicExch(&producer_flag, 1.0f);
    }
    for(int i = 1; i < K / TILE_SIZE; ++i){
      while(producer_flag == consumer_flag){
        __nanosleep(100);
      }
      for(int j = start_pos; j < mem_per_thread + start_pos; ++j){
        tile_a[i % 2][wy * sub_tile_height + level][wx * sub_tile_length + division * mem_per_thread + (j % mem_per_thread)] = A[(by * TILE_SIZE + wy * sub_tile_height + level) * K + (i * TILE_SIZE + wx * sub_tile_length + division * mem_per_thread + (j % mem_per_thread))];
        tile_b[i % 2][wy * sub_tile_height + level][wx * sub_tile_length + division * mem_per_thread + (j % mem_per_thread)] = B[(i * TILE_SIZE + wy * sub_tile_height + level) * N + (bx * TILE_SIZE + wx * sub_tile_length + division * mem_per_thread + (j % mem_per_thread))];
      }
      producer_exch_num = (i + 1) % 2;
      if(threadIdx.x < 1){
        atomicExch(&producer_flag, producer_exch_num);
      }
    }
  }
  else{
    fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag;
    fragment<matrix_b, 16, 16, 16, __half, row_major> b_frag;
    fragment<accumulator, 16, 16, 16, float> c_frag;
    fill_fragment(c_frag, 0.0f);
    for(int i = 0; i < K / TILE_SIZE; ++i){
      while(consumer_flag == producer_flag){
        __nanosleep(100);
      }
      for(int j = 0; j < TILE_SIZE / TENSOR_TILE_SIZE; ++j){
        load_matrix_sync(a_frag, &(tile_a[i % 2][tty * TENSOR_TILE_SIZE][j * TENSOR_TILE_SIZE]), TILE_SIZE);
        load_matrix_sync(b_frag, &(tile_b[i % 2][j * TENSOR_TILE_SIZE][ttx * TENSOR_TILE_SIZE]), TILE_SIZE);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      consumer_exch_num = (i + 1) % 2;
      if(threadIdx.x < WARP_SIZE * tensor_per_tile + 1){
        atomicAdd(&consumer_flag, consumer_exch_num);
      }
    }
    store_matrix_sync(&C[(by * TILE_SIZE + tty * TENSOR_TILE_SIZE) * N + (bx * TILE_SIZE + ttx * TENSOR_TILE_SIZE)], c_frag, N, mem_row_major);
  }

}




__global__ void naive_experimental_tensor_mat_mul_kernel_9(half* A, half* B, float* C, int M, int N, int K){
  int warp = threadIdx.x / WARP_SIZE;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag[TILE_SIZE_3 / TENSOR_TILE_SIZE][TILE_SIZE_3 / TENSOR_TILE_SIZE];
  fragment<matrix_b, 16, 16, 16, half, row_major> b_frag[TILE_SIZE_3 / TENSOR_TILE_SIZE][TILE_SIZE_3 / TENSOR_TILE_SIZE];
  fragment<accumulator, 16, 16, 16, float> c_frag[TILE_SIZE_3 / TENSOR_TILE_SIZE][TILE_SIZE_3 / TENSOR_TILE_SIZE];
  for(int i = 0; i < TILE_SIZE_3 / TENSOR_TILE_SIZE; ++i){
    for(int j = 0; j < TILE_SIZE_3 / TENSOR_TILE_SIZE; ++j){
      fill_fragment(c_frag[i][j], 0.0f);
    }
  }
  for(int i = 0; i < K / TILE_SIZE_3; ++i){
    for(int j = 0; j < TILE_SIZE_3 / TENSOR_TILE_SIZE; ++j){
      for(int k = 0; k < TILE_SIZE_3 / TENSOR_TILE_SIZE; ++k){
        load_matrix_sync(a_frag[j][k], &A[(by * TILE_SIZE_3 + j * TENSOR_TILE_SIZE) * K + (i * TILE_SIZE_3 + k * TENSOR_TILE_SIZE)], K);
        load_matrix_sync(b_frag[j][k], &B[(i * TILE_SIZE_3 + j * TENSOR_TILE_SIZE) * N + (bx * TILE_SIZE_3 + k * TENSOR_TILE_SIZE)], N);
      }
    }
    for(int j = 0; j < TILE_SIZE_3 / TENSOR_TILE_SIZE; ++j){
      for(int k = 0; k < TILE_SIZE_3 / TENSOR_TILE_SIZE; ++k){
        for(int l = 0; l < TILE_SIZE_3 / TENSOR_TILE_SIZE; ++l){
          mma_sync(c_frag[j][k], a_frag[j][l], b_frag[l][k], c_frag[j][k]);
        }
      }
    }
  }
  for(int i = 0; i < TILE_SIZE_3 / TENSOR_TILE_SIZE; ++i){
    for(int j = 0; j < TILE_SIZE_3 / TENSOR_TILE_SIZE; ++j){
      store_matrix_sync(&C[(by * TILE_SIZE_3 + i * TENSOR_TILE_SIZE) * N + (bx * TILE_SIZE_3 + j * TENSOR_TILE_SIZE)], c_frag[i][j], N, mem_row_major);
    }
  }
}




__global__ void naive_experimental_tensor_mat_mul_kernel_10(half* A, half* B, float *C, int M, int N, int K){
  __shared__ half tile_a[2][TILE_SIZE_4][TILE_SIZE_4];
  __shared__ half tile_b[2][TILE_SIZE_4][TILE_SIZE_4];
  int warp = threadIdx.x / WARP_SIZE;
  int tensor_tile_size = TILE_SIZE_4 / TENSOR_TILE_SIZE;
  int mem_per_thread = (TILE_SIZE_4 * TILE_SIZE_4) / blockDim.x;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = (threadIdx.x * mem_per_thread) % TILE_SIZE_4;
  int ty = (threadIdx.x * mem_per_thread) / TILE_SIZE_4;
  int ttx = warp % tensor_tile_size;
  int tty = warp / tensor_tile_size;

  int lane = threadIdx.x % WARP_SIZE;
  int threads_per_level = TILE_SIZE_4 / mem_per_thread;
  int division = lane % threads_per_level;
  int level = lane & (threads_per_level - 1);
  int levels = (WARP_SIZE / threads_per_level > 0) ? WARP_SIZE / threads_per_level : 1;
  int offset_unit = mem_per_thread / levels;
  int offset = level * offset_unit;

  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag;

  for(int i = offset; i < mem_per_thread + offset; ++i){
    tile_a[0][ty][tx + (i % mem_per_thread)] = A[(by * TILE_SIZE_4 + ty) * K + (0 * TILE_SIZE_4 + tx + (i % mem_per_thread))];
    tile_b[0][ty][tx + (i % mem_per_thread)] = B[(0 * TILE_SIZE_4 + ty) * N + (bx * TILE_SIZE_4 + tx + (i % mem_per_thread))];
  }
  for(int i = offset; i < mem_per_thread + offset; ++i){
    tile_a[1][ty][tx + (i % mem_per_thread)] = A[(by * TILE_SIZE_4 + ty) * K + (1 * TILE_SIZE_4 + tx + (i % mem_per_thread))];
    tile_b[1][ty][tx + (i % mem_per_thread)] = B[(1 * TILE_SIZE_4 + ty) * N + (bx * TILE_SIZE_4 + tx + (i % mem_per_thread))];
  }
  __syncthreads();

  for(int i = 0; i < K / TILE_SIZE_4 - 1; ++i){
    if(threadIdx.x < 128){
      for(int j = 0; j < TILE_SIZE_4 / TENSOR_TILE_SIZE; ++j){
        load_matrix_sync(a_frag, &(tile_a[i % 2][tty * TENSOR_TILE_SIZE][j * TENSOR_TILE_SIZE]), TILE_SIZE_4);
        load_matrix_sync(b_frag, &(tile_b[i % 2][j * TENSOR_TILE_SIZE][ttx * TENSOR_TILE_SIZE]), TILE_SIZE_4);
        mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
    }
    for(int j = offset; j < mem_per_thread + offset; ++j){
      tile_a[(i + 1) % 2][ty][tx + (j % mem_per_thread)] = A[(by * TILE_SIZE_4 + ty) * K + (i * TILE_SIZE_4 + tx + (j % mem_per_thread))];
      tile_b[(i + 1) % 2][ty][tx + (j % mem_per_thread)] = B[(i * TILE_SIZE_4 + ty) * N + (bx * TILE_SIZE_4 + tx + (j % mem_per_thread))];
    }
    __syncthreads();
  }

  if(threadIdx.x < 128){
    for(int i = 0; i < TILE_SIZE_4 / TENSOR_TILE_SIZE; ++i){
      load_matrix_sync(a_frag, &(tile_a[(K / TILE_SIZE_4 - 1) % 2][tty * TENSOR_TILE_SIZE][i * TENSOR_TILE_SIZE]), TILE_SIZE_4);
      load_matrix_sync(b_frag, &(tile_b[(K / TILE_SIZE_4 - 1) % 2][i * TENSOR_TILE_SIZE][ttx * TENSOR_TILE_SIZE]), TILE_SIZE_4);
      mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    store_matrix_sync(&C[(by * TILE_SIZE_4 + tty * TENSOR_TILE_SIZE) * N + (bx * TILE_SIZE_4 + ttx * TENSOR_TILE_SIZE)], c_frag, N, mem_row_major);
  }
}




__global__ void naive_experimental_tensor_mat_mul_kernel_11(half* A, half* B, float* C, int M, int N, int K){
  //create 32 x 128 a tile in shared memory and create 32 x 32 B tile and
  //Create longer tiles, more perimoiter for same amount of memory loaded, more perimeter means more reuse.
  //instead of using 32x32 fragment tiles, use 16x64 fragment tiles, see what happens and then perhaps try 8x128 tiles using 8x32x16 tensor configuration
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


  /*
  int tile_a_length_by_tensor = TILE_A_LENGTH / TENSOR_TILE_SIZE;
  int tile_a_height_by_tensor = TILE_A_HEIGHT / TENSOR_TILE_SIZE;
  int tile_b_length_by_tensor = TILE_B_LENGTH / TENSOR_TILE_SIZE;
  int tile_b_height_by_tensor = TILE_B_HEIGHT / TENSOR_TILE_SIZE;
  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, half, row_major> b_frag;
  fragment<accumulator, 16, 16, 16, float> c_frag[TILE_A_HEIGHT / TENSOR_TILE_SIZE][TILE_A_LENGTH / TENSOR_TILE_SIZE];
  for(int i = 0; i < TILE_A_HEIGHT / TENSOR_TILE_SIZE; ++i){
    for(int j = 0; j < TILE_A_LENGTH / TENSOR_TILE_SIZE; ++j){
      fill_fragment(c_frag[i][j], 0.0f);
    }
  }
  int warp = threadIdx.x / WARP_SIZE;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  for(int i = 0; i < K / TILE_A_LENGTH; ++i){
    for(int j = 0; j < TILE_A_HEIGHT / TENSOR_TILE_SIZE; ++j){
      for(int k = 0; k < TILE_A_LENGTH / TENSOR_TILE_SIZE; ++k){
        load_matrix_sync(a_frag, &(A[(by * TILE_A_HEIGHT + j * TENSOR_TILE_SIZE) * K + (i * TILE_A_LENGTH + k * TENSOR_TILE_SIZE)]), K);
        for(int l = 0; l < TILE_A_LENGTH / TENSOR_TILE_SIZE; ++l){
          load_matrix_sync(b_frag, &(B[(i * TILE_A_LENGTH + k * TENSOR_TILE_SIZE) * N + (bx * TILE_A_LENGTH + l * TENSOR_TILE_SIZE)]), N);
          mma_sync(c_frag[j][k], a_frag, b_frag, c_frag[j][k]);
        }
      }
    }
  }
  for(int i = 0; i < TILE_A_HEIGHT / TENSOR_TILE_SIZE; ++i){
    for(int j = 0; j < TILE_A_LENGTH / TENSOR_TILE_SIZE; ++j){
      store_matrix_sync(&(C[(by * TILE_A_HEIGHT + i * TENSOR_TILE_SIZE) * N + (bx * TILE_A_LENGTH + j * TENSOR_TILE_SIZE)]), c_frag[i][j], N, mem_row_major);
    }
  }
  */

}




__global__ void naive_experimental_tensor_mat_mul_kernel_12(half* A, half* B, float* C, int M, int N, int K){
  //create 32 x 128 a tile in shared memory and create 32 x 32 B tile and
  //Create longer tiles, more perimoiter for same amount of memory loaded, more perimeter means more reuse.
  //instead of using 32x32 fragment tiles, use 16x64 fragment tiles, see what happens and then perhaps try 8x128 tiles using 8x32x16 tensor configuration
  int bx = blockIdx.x;
  int by = blockIdx.y;
  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, half, row_major> b_frag[N_DIM / TENSOR_TILE_SIZE];
  fragment<accumulator, 16, 16, 16, float> c_frag[M_DIM / TENSOR_TILE_SIZE][N_DIM / TENSOR_TILE_SIZE];
  for(int i = 0; i < M_DIM / TENSOR_TILE_SIZE; ++i){
    for(int j = 0; j < N_DIM / TENSOR_TILE_SIZE; ++j){
      fill_fragment(c_frag[i][j], 0.0f);
    }
  }
  for(int i = 0; i < K / K_DIM; ++i){
    for(int j = 0; j < K_DIM / TENSOR_TILE_SIZE; ++j){
      for(int k = 0; k < N_DIM / TENSOR_TILE_SIZE; ++k){
        load_matrix_sync(b_frag[k], &B[(i * K_DIM + j * TENSOR_TILE_SIZE) * N + (bx * N_DIM + k * TENSOR_TILE_SIZE)], N);
      }
      for(int k = 0; k < M_DIM / TENSOR_TILE_SIZE; ++k){
        load_matrix_sync(a_frag, &A[(by * M_DIM + k * TENSOR_TILE_SIZE) * K + (i * K_DIM + j * TENSOR_TILE_SIZE)], K);
        for(int l = 0; l < N_DIM / TENSOR_TILE_SIZE; ++l){
          mma_sync(c_frag[k][l], a_frag, b_frag[l], c_frag[k][l]);
        }
      }
    }
  }
  for(int i = 0; i < M_DIM / TENSOR_TILE_SIZE; ++i){
    for(int j = 0; j < N_DIM / TENSOR_TILE_SIZE; ++j){
      store_matrix_sync(&C[(by * M_DIM + i * TENSOR_TILE_SIZE) * N + (bx * N_DIM + j * TENSOR_TILE_SIZE)], c_frag[i][j], N, mem_row_major);
    }
  }
}




__global__ void naive_experimental_tensor_mat_mul_kernel_13(half* A, half* B, float* C, int M, int N, int K){
  //create 32 x 128 a tile in shared memory and create 32 x 32 B tile and
  //Create longer tiles, more perimoiter for same amount of memory loaded, more perimeter means more reuse.
  //instead of using 32x32 fragment tiles, use 16x64 fragment tiles, see what happens and then perhaps try 8x128 tiles using 8x32x16 tensor configuration
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int warp = threadIdx.x / WARP_SIZE;
  int ttx = warp % (MULT * N_DIM_2);
  int tty = warp / (MULT * N_DIM_2);
  fragment<matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment<matrix_b, 16, 16, 16, half, row_major> b_frag[N_DIM_2 / TENSOR_TILE_SIZE];
  fragment<accumulator, 16, 16, 16, float> c_frag[M_DIM_2 / TENSOR_TILE_SIZE][N_DIM_2 / TENSOR_TILE_SIZE];
  for(int i = 0; i < M_DIM_2 / TENSOR_TILE_SIZE; ++i){
    for(int j = 0; j < N_DIM_2 / TENSOR_TILE_SIZE; ++j){
      fill_fragment(c_frag[i][j], 0.0f);
    }
  }
  for(int i = 0; i < K / K_DIM_2; ++i){
    for(int j = 0; j < K_DIM_2 / TENSOR_TILE_SIZE; ++j){
      for(int k = 0; k < N_DIM_2 / TENSOR_TILE_SIZE; ++k){
        load_matrix_sync(b_frag[k], &B[(i * K_DIM_2 + j * TENSOR_TILE_SIZE) * N + (bx * (MULT * N_DIM_2) + ttx * N_DIM_2 + k * TENSOR_TILE_SIZE)], N);
      }
      for(int k = 0; k < M_DIM_2 / TENSOR_TILE_SIZE; ++k){
        load_matrix_sync(a_frag, &A[(by * (MULT * M_DIM_2) + tty * M_DIM_2 + k * TENSOR_TILE_SIZE) * K + (i * K_DIM_2 + j * TENSOR_TILE_SIZE)], K);
        for(int l = 0; l < N_DIM_2 / TENSOR_TILE_SIZE; ++l){
          mma_sync(c_frag[k][l], a_frag, b_frag[l], c_frag[k][l]);
        }
      }
    }
  }
  for(int i = 0; i < M_DIM_2 / TENSOR_TILE_SIZE; ++i){
    for(int j = 0; j < N_DIM_2 / TENSOR_TILE_SIZE; ++j){
      store_matrix_sync(&C[(by * (MULT * M_DIM_2) + tty * M_DIM_2 + i * TENSOR_TILE_SIZE) * N + (bx * (MULT * N_DIM_2) + ttx * N_DIM_2 + j * TENSOR_TILE_SIZE)], c_frag[i][j], N, mem_row_major);
    }
  }
}




__global__ void naive_experimental_tensor_mat_mul_kernel_14(half* A, half* B, float* C, int M, int N, int K){
  __shared__ half tile_a[(MACRO_DIM_M / MICRO_DIM_M) * TENSOR_TILE_SIZE];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int warp = threadIdx.x / WARP_SIZE;
  int ttx = warp % (MACRO_DIM_N / MICRO_DIM_N);
  int tty = warp / (MACRO_DIM_N / MICRO_DIM_N);

  //there might be a situation where the number of threads outstips the number of spaces to fill in tile_a, forget about that scenario for now.
  int lane = threadIdx.x % WARP_SIZE;
  int mem_per_thread = (MACRO_DIM_M / MICRO_DIM_M) * (TENSOR_TILE_SIZE * TENSOR_TILE_SIZE) / blockDim.x;
  int banks_per_thread = mem_per_thread * HALF_SIZE / MEMORY_BANK_SIZE;
  int threads_per_row = TENSOR_TILE_SIZE / ((banks_per_thread > 0) ? mem_per_thread : MEMORY_BANK_SIZE / HALF_SIZE);
  int threads_per_col = WARP_SIZE / threads_per_row;
  int banks_per_row = TENSOR_TILE_SIZE * HALF_SIZE / MEMORY_BANK_SIZE;
  int rows_per_full_bank_set = MEMORY_BANKS / banks_per_row;
  int threads_per_full_bank_set = threads_per_row * rows_per_full_bank_set;
  int wtx = lane % threads_per_row;
  int wty = lane / threads_per_row;
  int warps_per_section = (banks_per_thread > 0) ? 1 : MEMORY_BANK_SIZE / (mem_per_thread * HALF_SIZE);
  int section = ttx / warps_per_section;
  int start_offset = (lane / threads_per_full_bank_set) * (MEMORY_BANK_SIZE / HALF_SIZE) + (ttx & (warps_per_section - 1)) * mem_per_thread * HALF_SIZE;

  fragment <matrix_a, 16, 16, 16, half, row_major> a_frag;
  fragment <matrix_b, 16, 16, 16, half, row_major> b_frag[MICRO_DIM_N];
  fragment <accumulator, 16, 16, 16, float> c_frag[MICRO_DIM_M][MICRO_DIM_N];

  for(int i = 0; i < MICRO_DIM_M; ++i){
    for(int j = 0; j < MICRO_DIM_N; ++j){
      fill_fragment(c_frag[i][j], 0.0f);
    }
  }

  for(int i = 0; i < K / MACRO_DIM_K; ++i){
    for(int j = 0; j < MACRO_DIM_K / MICRO_DIM_K; ++j){
      for(int k = 0; k < MACRO_DIM_M / MICRO_DIM_M; ++k){
        for(int l = start_offset; l < start_offset + mem_per_thread; ++l){
          //tile_a[];
        }
      }
    }
  }

}




__global__ void reuse_gemm(const half* A, const half* B, float* C, int M, int N, int K){
  half a;
  half b[MINI_TILE_SIZE];
  float c[MINI_TILE_SIZE][MINI_TILE_SIZE];
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  for(int i = 0; i < K / MINI_TILE_SIZE; ++i){
    for(int j = 0; j < MINI_TILE_SIZE; ++j){
      for(int k = 0; k < MINI_TILE_SIZE; ++k){
        b[k] = B[(i * MINI_TILE_SIZE + j) * N + (col * MINI_TILE_SIZE + k)];
      }
      for(int k = 0; k < MINI_TILE_SIZE; ++k){
        a = A[(row * MINI_TILE_SIZE + k) * K + (i * MINI_TILE_SIZE + j)];
        for(int l = 0; l < MINI_TILE_SIZE; ++l){
          c[k][l] += __half2float(a) * __half2float(b[l]);
        }
      }
    }
  }
  for(int i = 0; i < MINI_TILE_SIZE; ++i){
    for(int j = 0; j < MINI_TILE_SIZE; ++j){
      C[(row * MINI_TILE_SIZE + i) * N + (col * MINI_TILE_SIZE + j)] = c[i][j];
    }
  }
}




__global__ void reuse_gemm_2(const half* A, const half* B, float* C, int M, int N, int K){
  half a[MINI_TILE_SIZE_2][MINI_TILE_SIZE_2];
  half b[MINI_TILE_SIZE_2][MINI_TILE_SIZE_2];
  float c[MINI_TILE_SIZE_2][MINI_TILE_SIZE_2];
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int thread_num = threadIdx.y * 16 + threadIdx.x;
  int lane = thread_num % WARP_SIZE;
  int warp_col = lane % 16;
  int warp_row = lane / 16;
  int mini_warp_size = 2;
  int mini_warp_x_pos = warp_col / mini_warp_size;
  int mini_warp_y_pos = warp_row / mini_warp_size;
  int reference_thread = (mini_warp_y_pos * mini_warp_size) * 16 + (mini_warp_x_pos * mini_warp_size);
  int x_pos_in_mini_warp = warp_col % mini_warp_size;
  int y_pos_in_mini_warp = warp_row % mini_warp_size;

  for(int i = 0; i < K / (mini_warp_size * MINI_TILE_SIZE_2); ++i){
    for(int j = 0; j < MINI_TILE_SIZE_2; ++j){
      for(int k = 0; k < MINI_TILE_SIZE_2; ++k){
        a[j][k] = A[(row * MINI_TILE_SIZE_2 + j) * K + (i * MINI_TILE_SIZE_2 + k)];
        b[j][k] = B[(i * MINI_TILE_SIZE_2 + j) * N + (col * MINI_TILE_SIZE_2 + k)];
      }
    }
    for(int j = 0; j < MINI_TILE_SIZE_2; ++j){
      for(int k = 0; k < MINI_TILE_SIZE_2; ++k){
        for(int l = 0; l < MINI_TILE_SIZE_2; ++l){
          c[j][k] += __half2float(a[j][l]) * __half2float(b[l][k]);
        }
      }
    }
    for(int j = 1; j < mini_warp_size; ++j){
      for(int k = 0; k < MINI_TILE_SIZE_2; ++k){
        for(int l = 0; l < MINI_TILE_SIZE_2; ++l){
          a[k][l] = __shfl_sync(1 << thread_num, a[k][l], reference_thread + 16 * y_pos_in_mini_warp + j);
          b[k][l] = __shfl_sync(1 << thread_num, b[k][l], reference_thread + 16 * j + x_pos_in_mini_warp);
        }
      }
      for(int k = 0; k < MINI_TILE_SIZE_2; ++k){
        for(int l = 0; l < MINI_TILE_SIZE_2; ++l){
          for(int m = 0; m < MINI_TILE_SIZE_2; ++m){
            c[k][l] += __half2float(a[k][m]) * __half2float(b[m][l]);
          }
        }
      }
    }
  }
  for(int i = 0; i < MINI_TILE_SIZE_2; ++i){
    for(int j = 0; j < MINI_TILE_SIZE_2; ++j){
      C[(row * MINI_TILE_SIZE_2 + i) * N + (col * MINI_TILE_SIZE_2 + j)] = c[i][j];
    }
  }

}




__global__ void naive_gemm(const __half* a, const __half* b, float* c, int M, int N, int K){
  //printf("hello from naive_gemm()\n");
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  float acc = 0.0f;
  for(int i = 0; i < K; ++i){
    float av = __half2float(a[row * K + i]);
    float bv = __half2float(b[i * N + col]);
    acc += av * bv;
  }
  c[row * N + col] = acc;
}





__global__ void technique_test(half* A, half* B, float *C, int M, int N, int K){
  __shared__ __half tile_a[TILE_SIZE_2][TILE_SIZE_2];
  __shared__ __half tile_b[TILE_SIZE_2][TILE_SIZE_2];
  int mem_per_thread = 8;
  int mem_per_warp = mem_per_thread * WARP_SIZE;
  int unit_size = 2;
  int units_per_bank = MEMORY_BANK_SIZE / unit_size;
  int warp = threadIdx.x / WARP_SIZE;
  int tensor_tile_size = TILE_SIZE_2 / TENSOR_TILE_SIZE;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int ttx = warp % tensor_tile_size;
  int tty = warp / tensor_tile_size;
  int sub_tile_length = (MEMORY_BANKS >= TILE_SIZE_2 / units_per_bank) ? TILE_SIZE_2 : MEMORY_BANKS;
  int sub_tile_height = (TENSOR_TILE_SIZE * TENSOR_TILE_SIZE) / sub_tile_length;
  int wx = warp % (TILE_SIZE_2 / sub_tile_length);
  int wy = warp / (TILE_SIZE_2 / sub_tile_length);

  int lane = threadIdx.x % WARP_SIZE;
  int level = lane & (sub_tile_height - 1);
  int division = lane / sub_tile_height;
  mem_per_thread = (TENSOR_TILE_SIZE * TENSOR_TILE_SIZE) / WARP_SIZE;
  //Add double cast for variable values that might not neatly divide
  int boost = (mem_per_thread / sub_tile_height > 0) ? mem_per_thread / sub_tile_height : 1;
  int start_pos = (level * boost) % mem_per_thread;

  fragment<matrix_a, 16, 16, 16, __half, row_major> frag_a;
  fragment<matrix_b, 16, 16, 16, __half, row_major> frag_b;
  fragment<accumulator, 16, 16, 16, float> frag_c;
  fill_fragment(frag_c, 0.0f);

  int x_map = wx * sub_tile_length + division * mem_per_thread;
  int y_map = wy * sub_tile_height + level;
  int A_x_map_boost = wx * sub_tile_length + division * mem_per_thread;
  int A_y_map_boost = (by * TILE_SIZE_2 + wy * sub_tile_height + level) * K;
  int A_map;
  int B_x_map_boost = bx * TILE_SIZE_2 + wx * sub_tile_length + division * mem_per_thread;
  int B_y_map_boost = (wy * sub_tile_height + level) * N;
  int B_map;

  int A_test_boost;
  int B_test_boost;

  uint32_t state = blockIdx.x * blockDim.x + threadIdx.x;

  //NOTES:
  //With full tiling activated - Latency: 1.65 ms
  //With no tiling activated - Latency: 0.73866 ms
  //With back half tiling activated - Latency: 1.19 ms


  for(int i = 0; i < K / TILE_SIZE_2 - 1; ++i){
    //A_test_boost = rand_gen(state) % 1017;
    //B_test_boost = rand_gen(state) % 1017;
    A_map = A_y_map_boost + i * TILE_SIZE_2 + A_x_map_boost;
    B_map = i * TILE_SIZE_2 * N + B_y_map_boost + B_x_map_boost;
    for(int j = start_pos + 4; j < start_pos + mem_per_thread; ++j){
      tile_a[y_map][x_map + (j % mem_per_thread)] = A[A_map + (j % mem_per_thread)];
      tile_b[y_map][x_map + (j % mem_per_thread)] = B[B_map + (j % mem_per_thread)];
    }
    __syncthreads();
    for(int j = 0; j < TILE_SIZE_2 / TENSOR_TILE_SIZE; ++j){
      load_matrix_sync(frag_a, &(tile_a[tty * TENSOR_TILE_SIZE][j * TENSOR_TILE_SIZE]), TILE_SIZE_2);
      load_matrix_sync(frag_b, &(tile_b[j * TENSOR_TILE_SIZE][ttx * TENSOR_TILE_SIZE]), TILE_SIZE_2);
      mma_sync(frag_c, frag_a, frag_b, frag_c);
    }
    __syncthreads();
  }
  store_matrix_sync(&C[(by * TILE_SIZE_2 + tty * TENSOR_TILE_SIZE) * N + bx * TILE_SIZE_2 + ttx * TENSOR_TILE_SIZE], frag_c, N, mem_row_major);

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




int main(){
  //int M = 1024;
  //int N = 1024;
  //int K = 1024;
  int M = 4096;
  int N = 4096;
  int K = 4096;
  __half* h_a = (__half*)malloc(M * K * sizeof(__half));
  __half* h_b = (__half*)malloc(K * N * sizeof(__half));
  float* h_c = (float*)malloc(M * N * sizeof(float));
  __half* d_a;
  __half* d_b;
  float* d_c;
  //memset(h_a, 1, M * K * sizeof(__half));
  //memset(h_b, 2, K * N * sizeof(__half));
  //memset(h_c, 0.0f, M * N * sizeof(float));
  for (int i = 0; i < M*K; ++i) h_a[i] = __float2half(1.0f);
  for (int i = 0; i < K*N; ++i) h_b[i] = __float2half(2.0f);
  for (int i = 0; i < M*N; ++i) h_c[i] = 0.0f;
  cudaMalloc((void**)&d_a, M * K * sizeof(__half));
  cudaMalloc((void**)&d_b, K * N * sizeof(__half));
  cudaMalloc((void**)&d_c, M * N * sizeof(float));
  cudaMemcpy(d_a, h_a, M * K * sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, K * N * sizeof(__half), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, M * N * sizeof(float), cudaMemcpyHostToDevice);
  cudaStream_t s;
  cudaStreamCreate(&s);
  float avg_time;


  //dim3 grid_dim(N / 16, M / 16), block_dim(16,16);
  //avg_time = time_avg_us(s, [=]{naive_gemm<<<grid_dim, block_dim, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 10000);
  //cout << "Naive duration: " << avg_time << " us" << endl;

  //dim3 grid_dim1(N / (MINI_TILE_SIZE * 16), M / (MINI_TILE_SIZE * 16)), block_dim1(16,16);
  //avg_time = time_avg_us(s, [=]{reuse_gemm<<<grid_dim1, block_dim1, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 10000);
  //cout << "Reuse duration: " << avg_time << " us" << endl;

  //dim3 grid_dim02(N / (MINI_TILE_SIZE_2 * 16), M / (MINI_TILE_SIZE_2 * 16)), block_dim02(16,16);
  //avg_time = time_avg_us(s, [=]{reuse_gemm_2<<<grid_dim02, block_dim02, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 10000);
  //cout << "Reuse duration 2: " << avg_time << " us" << endl;

  //dim3 grid_dim1(N / TILE_SIZE, M / TILE_SIZE), block_dim1(TILE_SIZE, TILE_SIZE);
  //avg_time = time_avg_us(s, [=]{tiled_gemm<<<grid_dim1, block_dim1, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Tiled duration: " << avg_time << " us" << endl;

  //dim3 grid_dim6(N / TILE_SIZE, M / TILE_SIZE), block_dim6(TILE_SIZE, TILE_SIZE);
  //avg_time = time_avg_us(s, [=]{upgraded_tiled_gemm<<<grid_dim6, block_dim6, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Upgraded tiled duration: " << avg_time << " us" << endl;

  /*
  dim3 grid_dim2(1), block_dim2(128);
  avg_time = time_avg_us(s, [=]{warp_gemm<<<grid_dim2, block_dim2, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  cout << "Warp level duration: " << avg_time << " us" << endl;

  dim3 grid_dim3(1), block_dim3(128);
  avg_time = time_avg_us(s, [=]{warp_tiled_gemm<<<grid_dim3, block_dim3, 0,s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  cout << "Warp tile duration: " << avg_time << " us" << endl;

  dim3 grid_dim4(N / WARP_SIZE, M / WARP_SIZE), block_dim4(WARP_SIZE, WARP_SIZE);
  avg_time = time_avg_us(s, [=]{warp_shfl_gemm<<<grid_dim4, block_dim4>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  cout << "Warp shfl duration: " << avg_time << " us" << endl;
  */


  //dim3 grid_dim5(N / 16, M / 16), block_dim5(32);
  //avg_time = time_avg_us(s, [=]{naive_tensor_mat_mul_kernel<<<grid_dim5, block_dim5, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Naive tensor duration: " << avg_time << " us" << endl;


  //dim3 grid_dim6(N / 32, M / 32), block_dim6(TILE_SIZE * TILE_SIZE);
  //avg_time = time_avg_us(s, [=]{naive_experimental_tensor_mat_mul_kernel<<<grid_dim6, block_dim6, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Naive experimental tensor duration: " << avg_time << " us" << endl;


  //dim3 grid_dim7(N / 16, M / 16), block_dim7(1024);
  //avg_time = time_avg_us(s, [=]{test_kernel<<<grid_dim7, block_dim7, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Test kernel duration: " << avg_time << " us" << endl;


  //dim3 grid_dim8(N / 32, M / 32), block_dim8(128);
  //avg_time = time_avg_us(s, [=]{naive_experimental_tensor_mat_mul_kernel_2<<<grid_dim8, block_dim8, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Naive experimental tensor duration: " << avg_time << " us" << endl;


  //dim3 grid_dim9(N / TILE_SIZE, M / TILE_SIZE), block_dim9(1024);
  //avg_time = time_avg_us(s, [=]{naive_experimental_tensor_mat_mul_kernel_3<<<grid_dim9, block_dim9, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Naive experimental tensor 3.0 duration: " << avg_time << " us" << endl;


  //dim3 grid_dim10(N / 32, M / 32), block_dim10(TILE_SIZE * TILE_SIZE);
  //avg_time = time_avg_us(s, [=]{naive_experimental_tensor_mat_mul_kernel_4<<<grid_dim10, block_dim10, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Naive experimental tensor 4.0 duration: " << avg_time << " us" << endl;


  //dim3 grid_dim11(N / TILE_SIZE_2, M / TILE_SIZE_2), block_dim11(512);
  //avg_time = time_avg_us(s, [=]{naive_experimental_tensor_mat_mul_kernel_5<<<grid_dim11, block_dim11, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Naive experimental tensor 5.0 duration: " << avg_time << " us" << endl;


  //dim3 grid_dim12(N / TILE_SIZE_2, M / TILE_SIZE_2), block_dim12(1024);
  //avg_time = time_avg_us(s, [=]{naive_experimental_tensor_mat_mul_kernel_6<<<grid_dim12, block_dim12, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Naive experimental tensor 6.0 duration: " << avg_time << " us" << endl;


  //dim3 grid_dim13(N / TILE_SIZE, M / TILE_SIZE), block_dim13(256);
  //avg_time = time_avg_us(s, [=]{naive_experimental_tensor_mat_mul_kernel_8<<<grid_dim13, block_dim13, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Naive experimental tensor 8.0 duration: " << avg_time << " us" << endl;

  //dim3 grid_dim15(N / TILE_SIZE_3, M / TILE_SIZE_3), block_dim15(32);
  //avg_time = time_avg_us(s, [=]{naive_experimental_tensor_mat_mul_kernel_9<<<grid_dim15, block_dim15, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Naive experimental tensor 9.0 duration: " << avg_time << " us" << endl;

  //dim3 grid_dim17(N / TILE_SIZE_6, M / TILE_SIZE_6), block_dim17(32);
  //avg_time = time_avg_us(s, [=]{naive_experimental_tensor_mat_mul_kernel_11<<<grid_dim17, block_dim17, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Naive experimental tensor 11.0 duration: " << avg_time << " us" << endl;

  //dim3 grid_dim19(N / (MULT * N_DIM_2), M / (MULT * M_DIM_2)), block_dim19(MULT * MULT * WARP_SIZE);
  //avg_time = time_avg_us(s, [=]{naive_experimental_tensor_mat_mul_kernel_13<<<grid_dim19, block_dim19, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Naive experimental tensor 13.0 duration: " << avg_time << " us" << endl;

  //dim3 grid_dim20(N / MACRO_DIM_N, M / MACRO_DIM_M), block_dim20((MACRO_DIM_N / MICRO_DIM_N) * (MACRO_DIM_M / MICRO_DIM_M) * WARP_SIZE);
  //avg_time = time_avg_us(s, [=]{naive_experimental_tensor_mat_mul_kernel_14<<<grid_dim20, block_dim20, 0, s>>>(d_a, d_b, d_c, M, N, K);}, 1000);
  //cout << "Naive experimental tensor 14.0 duration: " << avg_time << " us" << endl;


  dim3 grid_dim(N / 16, M / 16), block_dim(16,16);
  naive_gemm<<<grid_dim, block_dim>>>(d_a, d_b, d_c, M, N, K);
  /*
  //tiled_gemm<<<grid_dim1, block_dim1>>>(d_a, d_b, d_c, M, N, K);
  //warp_gemm<<<grid_dim2, block_dim2>>>(d_a, d_b, d_c, M, N, K);
  //dim3 grid_dim4(1), block_dim4(N, M);
  //warp_shfl_gemm<<<grid_dim4, block_dim4>>>(d_a, d_b, d_c, M, N, K);
  //dim3 grid_dim6(N / TILE_SIZE, M / TILE_SIZE), block_dim6(TILE_SIZE, TILE_SIZE);
  //upgraded_tiled_gemm<<<grid_dim6, block_dim6, 0, s>>>(d_a, d_b, d_c, M, N, K);
  //dim3 grid_dim(N / 16, M / 16), block_dim(16,16);
  //naive_gemm<<<grid_dim, block_dim>>>(d_a, d_b, d_c, M, N, K);
  //dim3 grid_dim7(N / 16, M / 16), block_dim7(1024);
  //test_kernel<<<grid_dim7, block_dim7>>>(d_a, d_b, d_c, M, N, K);
  */

  //dim3 grid_dim5(N / TILE_SIZE, M / TILE_SIZE), block_dim5(512);
  //dim3 grid_dim5(1, 1), block_dim5(512);
  //naive_experimental_tensor_mat_mul_kernel_8<<<grid_dim5, block_dim5, 0, s>>>(d_a, d_b, d_c, M, N, K);

  //dim3 grid_dim8(N / TENSOR_TILE_SIZE, M / TENSOR_TILE_SIZE), block_dim8(32);
  //dim3 grid_dim8(N / TILE_SIZE, M / TILE_SIZE), block_dim8(128);
  //dim3 grid_dim13(N / TILE_SIZE_2, M / TILE_SIZE_2), block_dim13(128);
  //dim3 grid_dim14(N / TILE_SIZE, M / TILE_SIZE), block_dim14(256);

  dim3 grid_dim15(N / TILE_SIZE_3, M / TILE_SIZE_3), block_dim15(32);

  //dim3 gridDim(N / TILE_SIZE_2, M / TILE_SIZE_2), blockDim(128);
  //dim3 grid_dim16(N / TILE_SIZE_4, M / TILE_SIZE_4), block_dim16(1024);
  //dim3 grid_dim17(N / TILE_A_LENGTH, M / TILE_A_HEIGHT), block_dim17(32);

  dim3 grid_dim17(N / TILE_SIZE_6, M / TILE_SIZE_6), block_dim17(32);

  //dim3 grid_dim18(N / N_DIM, M / M_DIM), block_dim18(32);
  //dim3 grid_dim19(N / (MULT * N_DIM_2), M / (MULT * M_DIM_2)), block_dim19(MULT * MULT * WARP_SIZE);
  //dim3 grid_dim20(N / MACRO_DIM_N, M / MACRO_DIM_M), block_dim20((MACRO_DIM_N / MICRO_DIM_N) * (MACRO_DIM_M / MICRO_DIM_M) * WARP_SIZE);
  //naive_tensor_mat_mul_kernel<<<grid_dim8, block_dim8, 0, s>>>(d_a, d_b, d_c, M, N, K);
  //naive_experimental_tensor_mat_mul_kernel_2<<<grid_dim8, block_dim8, 0, s>>>(d_a, d_b, d_c, M, N, K);
  //naive_experimental_tensor_mat_mul_kernel_4<<<grid_dim10, block_dim10, 0, s>>>(d_a, d_b, d_c, M, N, K);
  //naive_experimental_tensor_mat_mul_kernel_3<<<grid_dim9, block_dim9, 0, s>>>(d_a, d_b, d_c, M, N, K);
  //naive_experimental_tensor_mat_mul_kernel_7<<<grid_dim13, block_dim13, 0, s>>>(d_a, d_b, d_c, M, N, K);
  //naive_experimental_tensor_mat_mul_kernel_8<<<grid_dim14, block_dim14, 0, s>>>(d_a, d_b, d_c, M, N, K);

  naive_experimental_tensor_mat_mul_kernel_9<<<grid_dim15, block_dim15, 0, s>>>(d_a, d_b, d_c, M, N, K);

  //naive_experimental_tensor_mat_mul_kernel_10<<<grid_dim16, block_dim16, 0, s>>>(d_a, d_b, d_c, M, N, K);

  naive_experimental_tensor_mat_mul_kernel_11<<<grid_dim17, block_dim17, 0, s>>>(d_a, d_b, d_c, M, N, K);

  //naive_experimental_tensor_mat_mul_kernel_12<<<grid_dim18, block_dim18, 0, s>>>(d_a, d_b, d_c, M, N, K);
  //naive_experimental_tensor_mat_mul_kernel_13<<<grid_dim19, block_dim19, 0, s>>>(d_a, d_b, d_c, M, N, K);

  //naive_experimental_tensor_mat_mul_kernel_14<<<grid_dim20, block_dim20, 0, s>>>(d_a, d_b, d_c, M, N, K);

  //technique_test<<<gridDim, blockDim, 0, s>>>(d_a, d_b, d_c, M, N, K);
  //tiled_gemm<<<grid_dim1, block_dim1, 0, s>>>(d_a, d_b, d_c, M, N, K);
  //naive_tensor_mat_mul_kernel<<<grid_dim8, block_dim8, 0, s>>>(d_a, d_b, d_c, M, N, K);
  cudaMemcpy(h_c, d_c, M * N * sizeof(float), cudaMemcpyDeviceToHost);
  //printf("Hello from main()\n");
  //printf("Hello from main() again\n");

  /*
  for(int i = 0; i < 16; ++i){
    for(int j = 0; j < 64; ++j){
      cout << h_c[i * N + j] << ", ";
    }
    cout << endl;
  }
  */

  /*
  for(int i = 0; i < M; ++i){
    for(int j = 0; j < N; ++j){
      cout << h_c[i * N + j] << ", ";
    }
    cout << endl;
  }
  */
  return 0;
}
