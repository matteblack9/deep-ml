#include <cuda_runtime.h>
#include <vector>

using namespace std;

__global__ void scale_kernel(const float* x, float a, float* y, int n) {
    // y[i] = a * x[i]
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) y[idx] = x[idx] * a;
}

std::vector<float> scalar_multiply(const std::vector<float>& x, float a) {
    int n = x.size();
    vector<float> result(n);

    float* xDevice;
    cudaMalloc(&xDevice, n * sizeof(float));

    cudaMemcpy(
        xDevice,
        x.data(),
        n * sizeof(float),
        cudaMemcpyHostToDevice
    );

    float* yDevice;
    cudaMalloc(&yDevice, n * sizeof(float));

    int threadPerBlock = 256;
    int blockSize = (n + (threadPerBlock - 1)) / threadPerBlock;

    scale_kernel<<<blockSize, threadPerBlock>>>(xDevice, a, yDevice, n);

    cudaMemcpy(
        result.data(),
        yDevice,
        n * sizeof(float),
        cudaMemcpyDeviceToHost
    );

    return result;
}
