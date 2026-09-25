#include <cuda_runtime.h>
#include <vector>

using namespace std;

__global__ void index_kernel(int* out, int n) {
    // Compute this thread's global index and, if it is < n, write it to out.

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) out[idx] = idx;
}

std::vector<int> global_thread_indices(int n) {
    // 1. allocate device memory for n ints
    // 2. launch the kernel with enough threads to cover n
    // 3. copy the result back to the host and return it

    vector<int> indexes(n);

    int *d_out;
    cudaMalloc(&d_out, n * sizeof(int));

    int threadsPerBlock = 128;
    int numBlocks = (n + (threadsPerBlock - 1)) / threadsPerBlock;

    index_kernel<<<numBlocks, threadsPerBlock>>>(d_out, n);
    cudaMemcpy(
        indexes.data(), 
        d_out, 
        n * sizeof(int), 
        cudaMemcpyDeviceToHost);

    cudaFree(d_out);

    return indexes;
}
