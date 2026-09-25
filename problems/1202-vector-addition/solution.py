#include <cuda_runtime.h>
#include <vector>

using namespace std;

__global__ void add_kernel(const float* a, const float* b, float* c, int n) {
    // c[i] = a[i] + b[i], guarded by i < n
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

std::vector<float> vector_add(const std::vector<float>& a, const std::vector<float>& b) {
    // allocate a, b, c on the device; copy a and b over (Host -> Device);
    // launch the kernel; copy c back; free memory

    int n = a.size();
    vector<float> result(n);
    
    float* aDevice;
    float* bDevice;

    cudaMalloc(&aDevice, n * sizeof(float));
    cudaMalloc(&bDevice, n * sizeof(float));

    cudaMemcpy(aDevice, a.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bDevice, b.data(), n * sizeof(float), cudaMemcpyHostToDevice);

    float* cDevice;
    cudaMalloc(&cDevice, n * sizeof(float));

    int threadPerBlock = 256;
    int blockSize = (n + (threadPerBlock - 1)) / threadPerBlock;

    add_kernel<<<blockSize, threadPerBlock>>>(aDevice, bDevice, cDevice, n);

    cudaMemcpy(result.data(), cDevice, n * sizeof(float), cudaMemcpyDeviceToHost);

    return result;
}
