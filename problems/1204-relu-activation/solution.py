#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

using namespace std;

__global__ void relu_kernel(const float* x, float* out, int n) {
    // out[i] = max(0, x[i])
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) out[idx] = 0 > x[idx] ? 0 : x[idx];
}

std::vector<float> relu(const std::vector<float>& x) {
    int n = x.size();
    vector<float> result(n);

    float* xDevice;
    cudaMalloc(&xDevice, n * sizeof(float));
    cudaMemcpy(
        xDevice, 
        x.data(), 
        n * sizeof(n), 
        cudaMemcpyHostToDevice
    );

    float* outDevice;
    cudaMalloc(&outDevice, n * sizeof(float));

    int threadPerBlock = 256;
    int blockSize = (n + (threadPerBlock - 1)) / threadPerBlock;

    relu_kernel<<<blockSize, threadPerBlock>>>(xDevice, outDevice, n);

    cudaMemcpy(
        result.data(),
        outDevice,
        n * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    return result;
}
