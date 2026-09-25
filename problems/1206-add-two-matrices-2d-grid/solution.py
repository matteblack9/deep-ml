#include <cuda_runtime.h>
#include <vector>

__global__ void matadd_kernel(const float* A, const float* B, float* C, int rows, int cols) {
    // Compute (row, col) from the 2D thread index; guard row<rows && col<cols;
    // C[row*cols + col] = A[row*cols + col] + B[row*cols + col];
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        C[idx] = A[idx] + B[idx];
    }
}

std::vector<float> matrix_add(const std::vector<std::vector<float>>& A,
                              const std::vector<std::vector<float>>& B) {
    // flatten A and B to row-major 1D, run the kernel, return C flattened
    int rows = A.size(), cols = A[0].size();
    int n = rows * cols;

    std::vector<float> tempA(n);
    std::vector<float> tempB(n);
    std::vector<float> result(n);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            tempA[i * cols + j] = A[i][j];
            tempB[i * cols + j] = B[i][j];
        }
    }

    float* ADevice;
    cudaMalloc(&ADevice, n * sizeof(float));
    cudaMemcpy(
        ADevice,
        tempA.data(),
        n * sizeof(float),
        cudaMemcpyHostToDevice
    );
    
    float* BDevice;
    cudaMalloc(&BDevice, n * sizeof(float));
    cudaMemcpy(
        BDevice,
        tempB.data(),
        n * sizeof(float),
        cudaMemcpyHostToDevice
    );

    float* CDevice;
    cudaMalloc(&CDevice, n * sizeof(float));

    dim3 block(16, 16);
    dim3 grid(
        (cols + block.x - 1) / block.x,
        (rows + block.y - 1) / block.y
    );

    matadd_kernel<<<grid, block>>>(ADevice, BDevice, CDevice, rows, cols);

    cudaMemcpy(
        result.data(), 
        CDevice, 
        n * sizeof(float), 
        cudaMemcpyDeviceToHost
    );

    cudaFree(ADevice);
    cudaFree(BDevice);
    cudaFree(CDevice);

    return result;
}
