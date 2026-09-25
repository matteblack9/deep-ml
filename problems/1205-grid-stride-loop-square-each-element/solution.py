#include <cuda_runtime.h>
#include <vector>

using namespace std;

__global__ void square_kernel(const float* x, float* out, int n) {
    // Use a grid-stride loop:
    //   for (int i = start; i < n; i += stride) out[i] = x[i] * x[i];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = blockDim.x;

    for (int i = idx; i < n; i += stride)
        out[i] = x[i] * x[i];
}

std::vector<float> square(const std::vector<float>& x) {
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

    float* outDevice;
    cudaMalloc(&outDevice, n * sizeof(float));

    int threadNum = 8;
    int blockSize = 1;

    square_kernel<<<blockSize, threadNum>>>(xDevice, outDevice, n);

    cudaMemcpy(
        result.data(), 
        outDevice, 
        n * sizeof(float), 
        cudaMemcpyDeviceToHost
    );

    cudaFree(xDevice);
    cudaFree(outDevice);

    return result;
}
