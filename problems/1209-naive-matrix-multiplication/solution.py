#include <cuda_runtime.h>
#include <vector>

#define TILE_DIM 32

using namespace std;

__global__ void matmul_kernel(const float* A, const float* B, float* C, int M, int K, int N) {
    // thread (row, col): if row<M && col<N, C[row*N+col] = sum_k A[row*K+k]*B[k*N+col]
    int x_out = blockIdx.x * TILE_DIM + threadIdx.x;
    int y_out = blockIdx.y * TILE_DIM + threadIdx.y;


    if (x_out < N && y_out < M) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[y_out * K + k] * B[k * N + x_out];
        }

        C[y_out * N + x_out]  += sum;
    }
}

std::vector<float> matmul(const std::vector<std::vector<float>>& A,
                          const std::vector<std::vector<float>>& B) {
    int M = A.size(), K = A[0].size(), N = B[0].size();

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> C(M * N); // 결과 담을 공간(M * N) 미리 할당!

    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            h_A[i * K + j] = A[i][j];
        }
    }

    for (int i = 0; i < K; i++) {
        for (int j = 0; j < N; j++) {
            h_B[i * N + j] = B[i][j];
        }
    }

    float *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid((N + (TILE_DIM - 1)) / TILE_DIM, (M + (TILE_DIM - 1)) / TILE_DIM);

    matmul_kernel<<<grid, block>>>(d_A, d_B, d_C, M, K, N);

    cudaMemcpy(C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return C;
}
