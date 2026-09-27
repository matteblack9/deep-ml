#include <cuda_runtime.h>
#include <vector>

__global__ void dot_kernel(const float* a, const float* b, float* out, int n) {
    // load a[tid]*b[tid] (or 0) into shared memory, then tree-reduce and write out[0]
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    shared[tid] = tid < n ? a[tid] * b[tid] : 0.0f;

    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tid < stride)
            shared[tid] += shared[tid + stride];
            __syncthreads();
    }

    if (tid == 0)
        atomicAdd(out, shared[0]);
}

float dot_product(const std::vector<float>& a, const std::vector<float>& b) {
    float* da;
    float* db;
    float* dout;

    int n = a.size();

    cudaMalloc(&da, n * sizeof(float));
    cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&db, n * sizeof(float));
    cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&dout, sizeof(float));
    cudaMemset(dout, 0, sizeof(float));

    int threads = 256;
    int blockSize = (n + threads - 1) / threads;
    dot_kernel<<<blockSize, threads, threads * sizeof(float)>>>(da, db, dout, n);

    float out;

    cudaMemcpy(&out, dout, sizeof(float), cudaMemcpyDeviceToHost);

    return out;
}
