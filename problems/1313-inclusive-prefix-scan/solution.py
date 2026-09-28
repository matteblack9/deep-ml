#include <cuda_runtime.h>

__global__ void blellochScan(const float* input, float* output, int n) { 
    extern __shared__ float shared[];

    int tid = threadIdx.x;

    if (tid < n) {
        shared[tid] = input[tid];
    } else {
        shared[tid] = 0.0f;
    }

    __syncthreads();

    for(int offset = 1; offset < n; offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1;

        if (idx < n) {
            shared[idx] += shared[idx - offset];
        }

        __syncthreads();
    }

    int root = shared[n - 1];
    if (tid == 0) {
        shared[n - 1] = 0.0f;
    }

    __syncthreads();

    for (int offset = n / 2; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * offset * 2 - 1;

        if (idx < n) {
            float temp = shared[idx - offset];
            shared[idx - offset] = shared[idx];
            shared[idx] += temp;
        }

        __syncthreads();
    }

    output[tid] = input[tid] + shared[tid];    
}

void solve(const float* input, float* output, int n) {
    float* dinput;
    float* doutput;

    cudaMalloc(&dinput, n * sizeof(float));
    cudaMemcpy(dinput, input, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&doutput, n * sizeof(float));
    cudaMemset(doutput, 0, n * sizeof(float));

    int threads = 1024;
    int blocks = (n + (threads - 1)) / threads;

    int powerOfn = 1;
    while (powerOfn < blocks) powerOfn <<= 1;
    blocks = powerOfn;

    size_t smemSize = threads * sizeof(float);

    blellochScan<<<blocks, threads, smemSize>>>(dinput, doutput, n);

    cudaMemcpy(output, doutput, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dinput);
    cudaFree(doutput);
}
