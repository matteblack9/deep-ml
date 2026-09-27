#include <cuda_runtime.h>
#include <vector>

__global__ void sum_kernel(const float* x, float* out, int n) {
    // 1. load x[tid] (or 0) into __shared__ memory, then __syncthreads()
    // 2. tree-reduce: for (s = blockDim.x/2; s > 0; s >>= 1) add sdata[tid+s]
    // 3. thread 0 writes out[0]

    extern __shared__ float shared[];

    int tid = threadIdx.x;
    shared[tid] = x[tid];
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (stride > tid)
            shared[tid] += shared[tid + stride];

        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(out, shared[0]);
}

float array_sum(const std::vector<float>& x) {
    float* din;
    float* dout;

    int N = x.size();

    cudaMalloc(&din, N * sizeof(float));
    cudaMemcpy(
        din,
        x.data(),
        N * sizeof(float),
        cudaMemcpyHostToDevice
    );

    cudaMalloc(&dout, sizeof(float));
    cudaMemset(dout, 0, sizeof(float));

    int threads = 256;

    sum_kernel<<<1, threads, threads * sizeof(float)>>>(din, dout, N);

    float out = 0;

    cudaMemcpy(
        &out,
        dout,
        sizeof(float),
        cudaMemcpyDeviceToHost
    );

    return out;
}
