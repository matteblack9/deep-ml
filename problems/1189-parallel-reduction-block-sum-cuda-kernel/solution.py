#include <cuda_runtime.h>

__global__ void kernel(const float* in, float* out, int N) {
    // TODO: block-wise reduction using shared memory
    extern __shared__ float shared[];

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    if (idx < N)
        shared[tid] = in[idx];
    else
        shared[tid] = 0.0f;

    for (int stride = blockDim.x >> 1; stride > 0; stride = stride >> 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];

        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(out, shared[0]);
}

void solve(const float* input, float* output, int N) {
    // TODO: allocate device memory, copy in, launch kernel, copy out, free
    float* dInput;
    float* dOutput;

    size_t allocSize = N * sizeof(float);
    
    cudaMalloc(&dInput, allocSize);
    cudaMalloc(&dOutput, sizeof(float));

    cudaMemcpy(
        dInput, 
        input, 
        allocSize, 
        cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + (threads - 1)) / threads;

    kernel<<<blocks, threads, threads * sizeof(float)>>>(dInput, dOutput, N);

    cudaMemcpy(
        output, 
        dOutput, 
        sizeof(float), 
        cudaMemcpyDeviceToHost
    );

    cudaFree(dInput);
    cudaFree(dOutput);
}