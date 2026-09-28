#include <cuda_runtime.h>

__global__ void compaction(float* input, float* output, int* count, int n) {
    extern __shared__ int shared[];
    
    int tid = threadIdx.x;
    
    int flag = input[tid] == 0 ? 0 : 1;
    shared[tid] = flag;
    __syncthreads();

    for (int offset = 1; offset < n; offset <<= 1) {
        int idx = (tid + 1) * offset * 2 - 1;

        if (idx < n) {
            shared[idx] += shared[idx - offset];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *count = shared[n - 1];
        shared[n - 1] = 0.0f;
    }
    __syncthreads();

    for (int offset = n >> 1; offset > 0; offset >>= 1) {
        int idx = (tid + 1) * offset * 2 - 1;

        if (idx < n) {
            int temp = shared[idx - offset];
            shared[idx - offset] = shared[idx];
            shared[idx] += temp;
        }
        __syncthreads();
    }

    if (flag) {
        int outputIdx = shared[tid];
        output[outputIdx] = input[tid];
    }

}

void solve(const float* input, float* output, int* count, int n) {
    float* dinput;
    float* doutput;
    int* dcount;

    cudaMalloc(&dinput, n * sizeof(float));
    cudaMemcpy(dinput, input, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&doutput, n * sizeof(float));
    cudaMemcpy(doutput, output, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&dcount, sizeof(int));

    int threads = n;
    int blockSize = 1;

    compaction<<<1, threads, threads * sizeof(float)>>>(dinput, doutput, dcount, n);

    cudaMemcpy(output, doutput, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(count, dcount, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dinput);
    cudaFree(doutput);
    cudaFree(dcount);
}
