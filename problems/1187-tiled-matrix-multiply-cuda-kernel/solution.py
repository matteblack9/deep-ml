#include <cuda_runtime.h>

#define TILE_DIM 32

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    int t_row = threadIdx.y;
    int t_col = threadIdx.x;

    float sum = 0.0f;

    int num_tiles = (K + (TILE_DIM - 1)) / TILE_DIM;

    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * TILE_DIM + t_col;
        int a_row = row;

        if (a_col < K && a_row < M) {
            s_A[t_row][t_col] = A[a_row * K + a_col];
        } else {
            s_A[t_row][t_col] = 0.0f;
        }

        int b_col = col;
        int b_row = t * TILE_DIM + t_row;

        if (b_col < N && b_row < K) {
            s_B[t_row][t_col] = B[b_row * N + b_col];
        } else {
            s_B[t_row][t_col] = 0.0;
        }

        __syncthreads();

        for (int k = 0; k < TILE_DIM; k++) {
            sum += s_A[t_row][k] * s_B[k][t_col];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: allocate device memory, copy in, launch kernel, copy out, free

    float* d_A, *d_B, *d_C;

    size_t size_A = sizeof(float) * M * K;
    size_t size_B = sizeof(float) * K * N;
    size_t size_C = sizeof(float) * M * N;

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (M + (TILE_DIM - 1))/ TILE_DIM);

    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}