#include <cuda_runtime.h>

__global__ void segmentedSum(const float* values, const int* flags, float* output, int n) {
    extern __shared__ char shared[];
    float* valueSums = reinterpret_cast<float*>(shared);
    int* flagSums = reinterpret_cast<int*>(valueSums + n);

    int tid = threadIdx.x;
    flagSums[tid] = flags[tid];

    __syncthreads();

    for (int offset = 1; offset < n; offset <<= 1) {
        int add = 0;

        if (tid >= offset) 
            add = flagSums[tid - offset];

        __syncthreads();

        if (tid >= offset)
            flagSums[tid] += add;

        __syncthreads();
    }

    int groupNum = flagSums[tid];

    atomicAdd(&output[groupNum - 1], values[tid]);

// [1, 0, 0, 1, 0, 1]
// [0, 1, 1, 1, 2, 2]
// [1, 1, 1, 2, 2, 3]
}

void solve(const float* values, const int* flags, float* output, int n) {
    float* dvalues;
    int* dflags;
    float* doutput;

    cudaMalloc(&dvalues, n * sizeof(float));
    cudaMemcpy(dvalues, values, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&dflags, n * sizeof(int));
    cudaMemcpy(dflags, flags, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMalloc(&doutput, n * sizeof(float));

    int threads = n;
    int blockSize = 1;
    size_t smemValuesSize = n * sizeof(float);
    size_t smemFlagSize = n * sizeof(int);
    size_t smemBoundarySize = n * sizeof(int);

    segmentedSum<<<blockSize, threads, smemValuesSize + smemFlagSize + smemBoundarySize>>>(dvalues, dflags, doutput, n);

    cudaMemcpy(output, doutput, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dvalues);
    cudaFree(dflags);
    cudaFree(doutput);
}
