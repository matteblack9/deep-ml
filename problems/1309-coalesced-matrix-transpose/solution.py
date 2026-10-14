#include <cuda_runtime.h>

// Be careful BANK CONFLICT !!!

#define TILE_DIM 32

__global__ void coalesced_matrix_transpose(float* __restrict__ A, float* __restrict__ B, int M, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    int x_in = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_in = blockIdx.y * TILE_DIM + threadIdx.y;

    if (y_in < M && x_in < N) {
        tile[threadIdx.y][threadIdx.x] = A[y_in * N + x_in];
    }

    __syncthreads();

    int x_out =  blockIdx.y * TILE_DIM + threadIdx.x;
    int y_out = blockIdx.x * TILE_DIM + threadIdx.y;

    if (y_out < N && x_out < M)
        B[y_out * M + x_out] = tile[threadIdx.x][threadIdx.y];
}

void solve(const float* A, float* B, int M, int N) {
    float* d_A, *d_B;
    size_t size = M * N * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(N + (TILE_DIM - 1) / TILE_DIM, M + (TILE_DIM - 1) / TILE_DIM);

    coalesced_matrix_transpose<<<grid, block>>>(d_A, d_B, M, N);

    cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
}
