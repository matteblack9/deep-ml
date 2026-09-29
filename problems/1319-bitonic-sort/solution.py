#include <cuda_runtime.h>

__global__ void bitonicSort(const float* input, float* output, int n) {
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    shared[tid] = input[tid];

    __syncthreads();

    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            int partner = tid ^ j;
            
            if (tid < partner) {
                bool ascending = ((k & tid) == 0);
                if ((ascending && shared[tid] > shared[partner]) || 
                !ascending && shared[tid] < shared[partner]) {
                    float temp = shared[tid];
                    shared[tid] = shared[partner];
                    shared[partner] = temp;
                }
            }
            __syncthreads();
        }
    }

    output[tid] = shared[tid];
}

void solve(const float* input, float* output, int n) {
    // TODO
    float* dinput;
    float* doutput;

    cudaMalloc(&dinput, n * sizeof(float));
    cudaMemcpy(dinput, input, n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&doutput, n * sizeof(float));

    int threads = n;
    int blockSize = 1;

    bitonicSort<<<blockSize, threads, n * sizeof(float)>>>(dinput, doutput, n);

    cudaMemcpy(output, doutput, n * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dinput);
    cudaFree(doutput);
}
