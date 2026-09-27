#include <cuda_runtime.h>

__global__ void kernel(const int* input, int* hist, int n) {
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    size_t bin = input[idx];
    atomicAdd(hist + bin, 1);
}

void solve(const int* input, int* hist, int n, int num_bins) {
    // TODO

    int* dinput;
    int* dhist;
    
    cudaMalloc(&dinput, n * sizeof(int));
    cudaMemcpy(dinput, input, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&dhist, n * sizeof(int));
    cudaMemset(dhist, 0, n * sizeof(int));

    // malloc(&hist, num_bins * sizeof(int));
    memset(hist, 0, num_bins * sizeof(int));

    int threads = num_bins;
    int blockSize = (n + (threads - 1)) / threads;

    kernel<<<blockSize, threads, num_bins * sizeof(int)>>>(dinput, dhist, n);

    cudaMemcpy(hist, dhist, num_bins * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dinput);
    cudaFree(dhist);
}
