#include <cuda_runtime.h>

__device__ float shuffleReduceSum(float value, int warpSize) {
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }

    return value;
}

__global__ void sumKernel(const float* x, float* out, int n) {
    extern __shared__ float shared[]; // thread 256 / 32 = 8

    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int warpSize = 32;
    int value = idx < n ? x[idx] : 0.0f;

    int warpIdx = tid / warpSize;
    int lane = tid % warpSize;

    value = shuffleReduceSum(value, warpSize);

    if (lane == 0)
        shared[warpIdx] = value;
        
    __syncthreads();

    if (warpIdx == 0) {
        int numWarps = blockDim.x / 32;
        value = (lane < numWarps) ? shared[lane] : 0.0f;
        value = shuffleReduceSum(value, warpSize);
        
        if (lane == 0)
            atomicAdd(out, value);
    }
}

void solve(const float* x, float* out, int n) {
    float* dx;
    cudaMalloc(&dx, n * sizeof(float));
    cudaMemcpy(dx, x, n * sizeof(float), cudaMemcpyHostToDevice);

    float* dout;
    cudaMalloc(&dout, sizeof(float));
    cudaMemset(dout, 0, sizeof(float));

    int threads = 256;
    int blockSize = (n + (threads - 1)) / threads;

    sumKernel<<<blockSize, threads, (threads / 32) * sizeof(float)>>>(dx, dout, n);

    cudaMemcpy(out, dout, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dx);
    cudaFree(dout);
}
