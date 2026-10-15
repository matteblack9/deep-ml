#include <cuda_runtime.h>

__global__ void conv_1d(const float* input, const float* kernel, float* output, const int n, const int radius) {
    const size_t s_kernel_size = radius * 2 + 1;

    __shared__ float s_input[1024];
    __shared__ float s_kernel[8];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (0 <= idx && idx < n) {
        s_input[idx] = input[idx];
    }

    if (0 <= idx && idx < s_kernel_size) {
        s_kernel[idx] = kernel[idx];
    }

    __syncthreads();

    float sum = 0.0f;
    for (int i = 0; i < s_kernel_size; i++) {
        int convIdx = idx - radius + i;

        if (convIdx >= 0 && convIdx < n)
            sum += s_kernel[i] * s_input[convIdx];
    }

    output[idx] = sum;
}

void solve(const float* input, const float* kernel, float* output, int n, int radius) {
    float* d_input, *d_kernel, *d_output;
    size_t input_size = sizeof(float) * n;
    size_t kernel_size = sizeof(float) * (2 * radius + 1);
    
    cudaMalloc(&d_input, input_size);
    cudaMalloc(&d_kernel, kernel_size);
    cudaMalloc(&d_output, input_size);

    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, kernel, kernel_size, cudaMemcpyHostToDevice);
    
    size_t threads = 1024;
    size_t blocks = (max(n, 2 * radius + 1) + threads - 1) / threads;

    conv_1d<<<blocks, threads>>>(d_input, d_kernel, d_output, n, radius);

    cudaMemcpy(output, d_output, input_size, cudaMemcpyDeviceToHost);
    
    cudaFree(d_input);
    cudaFree(d_kernel);
    cudaFree(d_output);
}
