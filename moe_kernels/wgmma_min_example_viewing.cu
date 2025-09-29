#include <cuda.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <cassert>
#include <cuda/barrier>
#include <cuda/ptx>

// Type alias for FP8 E4M3
using fp8 = __nv_fp8_e4m3;

// WGMMA synchronization functions
__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait() {
    static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// // Create simple shared memory descriptor for WGMMA (simplified approach)
// __device__ uint64_t make_smem_desc(const void* ptr) {
//     uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
//     // Simple descriptor - just the encoded address
//     return (addr >> 4) & 0x3FFFF;
// }

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

__device__ uint64_t make_smem_desc(fp8* ptr, int leading_dim_bytes, int stride_dim_bytes) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)leading_dim_bytes) << 16;
    desc |= matrix_descriptor_encode((uint64_t)stride_dim_bytes) << 32;
    // desc |= 1llu << 62; // 128B swizzle
    return desc;
  }

// WGMMA function for M64N16K32 with FP8 E4M3 inputs and FP32 output
template<int ScaleD, int ScaleA, int ScaleB>
__device__ void wgmmaM64N16K32(float d[2][2][2], fp8* sA, fp8* sB) {
    // Create simple matrix descriptors
    // uint64_t desc_a = make_smem_desc(sA, 32, 256);
    // uint64_t desc_b = make_smem_desc(sB, 32, 256);

    uint64_t desc_a = make_smem_desc(sA, 1024, 128);
    uint64_t desc_b = make_smem_desc(sB, 256, 128);

    // uint64_t desc_a = make_smem_desc(sA, 128, 256);
    // uint64_t desc_b = make_smem_desc(sB, 128, 256);

    if (threadIdx.x == 0) {
        printf("desc_a: %llu\n", desc_a);
        printf("desc_b: %llu\n", desc_b);
    }
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "%8, %9, "
        "%10, %11, %12;\n"
        : "+f"(d[0][0][0]), "+f"(d[0][0][1]), "+f"(d[1][0][0]), "+f"(d[1][0][1]), 
          "+f"(d[0][1][0]), "+f"(d[0][1][1]), "+f"(d[1][1][0]), "+f"(d[1][1][1])
        : "l"(desc_a), "l"(desc_b), 
          "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)));
}

// Minimal kernel demonstrating WGMMA usage
__global__ void wgmma_minimal_kernel(
    const fp8* __restrict__ A_global,  // 64x32 matrix A
    const fp8* __restrict__ B_global,  // 16x32 matrix B  
    float* __restrict__ C_global       // 64x16 output matrix C
) {
    // Shared memory for matrices (aligned to 128 bytes)
    __shared__ alignas(128) fp8 sA[64][32];  // 64x32 matrix A
    __shared__ alignas(128) fp8 sB[16][32];  // 16x32 matrix B
    
    const int lane_id = threadIdx.x % 32;
    const int warp_id = threadIdx.x / 32;
    
    // Load matrix A into shared memory (64x32)
    // Use coalesced loads - each thread loads one element per iteration
    for (int i = threadIdx.x; i < 64 * 32; i += blockDim.x) {
        int row = i / 32;
        int col = i % 32;

        int m0 = row % 8;
        int m1 = row / 8;
        int k0 = col % 16;
        int k1 = col / 16;
        int ofs = m0 * 16 + m1 * 128 + k0 + k1 * 1024;

        int sw_row = ofs / 32;
        int sw_col = ofs % 32;

        sA[sw_row][sw_col] = A_global[i];
    }
    // // copy directly into shared memory
    // for (int i = threadIdx.x; i < 64 * 32; i += blockDim.x) {
    //     int row = i / 32;
    //     int col = i % 32;
    //     sA[row][col] = A_global[i];
    // }
    
    // Load matrix B into shared memory (16x32)
    for (int i = threadIdx.x; i < 16 * 32; i += blockDim.x) {
        int row = i / 32;
        int col = i % 32;

        int m0 = row % 8;
        int m1 = row / 8;
        int k0 = col % 16;
        int k1 = col / 16;
        int ofs = m0 * 16 + m1 * 128 + k0 + k1 * 256;

        int sw_row = ofs / 32;
        int sw_col = ofs % 32;

        sB[sw_row][sw_col] = B_global[i];
    }


    
    __syncthreads();
    
    // Initialize accumulator - only for threads in warp group 0
    float acc[2][2][2] = {0.0f};
    
    // WGMMA operation - only executed by warp group 0 (first 4 warps)
    if (warp_id < 4) {
        warpgroup_arrive();
        wgmmaM64N16K32<1, 1, 1>(acc, &sA[0][0], &sB[0][0]);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
    }
    
    __syncthreads();

    if (threadIdx.x == 0) {
        printf("Thread 0\n");
        printf("acc[0][0][0]: %f\n", acc[0][0][0]);
        printf("acc[0][0][1]: %f\n", acc[0][0][1]);
        printf("acc[1][0][0]: %f\n", acc[1][0][0]);
        printf("acc[1][0][1]: %f\n", acc[1][0][1]);
        printf("acc[0][1][0]: %f\n", acc[0][1][0]);
        printf("acc[0][1][1]: %f\n", acc[0][1][1]);
        printf("acc[1][1][0]: %f\n", acc[1][1][0]);
        printf("acc[1][1][1]: %f\n", acc[1][1][1]);
    }
    if (threadIdx.x == 1) {
        printf("Thread 1\n");
        printf("acc[0][0][0]: %f\n", acc[0][0][0]);
        printf("acc[0][0][1]: %f\n", acc[0][0][1]);
        printf("acc[1][0][0]: %f\n", acc[1][0][0]);
        printf("acc[1][0][1]: %f\n", acc[1][0][1]);
        printf("acc[0][1][0]: %f\n", acc[0][1][0]);
        printf("acc[0][1][1]: %f\n", acc[0][1][1]);
        printf("acc[1][1][0]: %f\n", acc[1][1][0]);
        printf("acc[1][1][1]: %f\n", acc[1][1][1]);
    }
    if (threadIdx.x == 2) {
        printf("Thread 2\n");
        printf("acc[0][0][0]: %f\n", acc[0][0][0]);
        printf("acc[0][0][1]: %f\n", acc[0][0][1]);
        printf("acc[1][0][0]: %f\n", acc[1][0][0]);
        printf("acc[1][0][1]: %f\n", acc[1][0][1]);
        printf("acc[0][1][0]: %f\n", acc[0][1][0]);
        printf("acc[0][1][1]: %f\n", acc[0][1][1]);
        printf("acc[1][1][0]: %f\n", acc[1][1][0]);
        printf("acc[1][1][1]: %f\n", acc[1][1][1]);
    }
    if (threadIdx.x == 3) {
        printf("Thread 3\n");
        printf("acc[0][0][0]: %f\n", acc[0][0][0]);
        printf("acc[0][0][1]: %f\n", acc[0][0][1]);
        printf("acc[1][0][0]: %f\n", acc[1][0][0]);
        printf("acc[1][0][1]: %f\n", acc[1][0][1]);
        printf("acc[0][1][0]: %f\n", acc[0][1][0]);
        printf("acc[0][1][1]: %f\n", acc[0][1][1]);
        printf("acc[1][1][0]: %f\n", acc[1][1][0]);
        printf("acc[1][1][1]: %f\n", acc[1][1][1]);
    }
    if (threadIdx.x == 4) {
        printf("Thread 4\n");
        printf("acc[0][0][0]: %f\n", acc[0][0][0]);
        printf("acc[0][0][1]: %f\n", acc[0][0][1]);
        printf("acc[1][0][0]: %f\n", acc[1][0][0]);
        printf("acc[1][0][1]: %f\n", acc[1][0][1]);
        printf("acc[0][1][0]: %f\n", acc[0][1][0]);
        printf("acc[0][1][1]: %f\n", acc[0][1][1]);
        printf("acc[1][1][0]: %f\n", acc[1][1][0]);
        printf("acc[1][1][1]: %f\n", acc[1][1][1]);
    }
    
    // Store results back to global memory
    // Only threads that participated in WGMMA write results
    if (warp_id < 4) {
        // Simple mapping: each thread writes one result element
        int row_length = 16;
        int thread_in_warpgroup = warp_id * 32 + lane_id;
        if (thread_in_warpgroup < 64 * 16 / 8) {  // 8 results per thread
            int warp_group_off = warp_id * 16 * 16;
            int thread_in_warp_idx = threadIdx.x % 32;

            int row0 = thread_in_warp_idx / 4 * row_length;
            int row1 = 8*row_length + row0;

            int col0 = (thread_in_warp_idx % 4) * 2;
            int col1 = 8 + (thread_in_warp_idx % 4) * 2;
            
            // // Write the 8 accumulator values
            C_global[warp_group_off + row0 + col0] = acc[0][0][0];
            C_global[warp_group_off + row0 + col0 + 1] = acc[0][0][1];
            C_global[warp_group_off + row1 + col0] = acc[1][0][0];
            C_global[warp_group_off + row1 + col0 + 1] = acc[1][0][1];

            C_global[warp_group_off + row0 + col1] = acc[0][1][0];
            C_global[warp_group_off + row0 + col1 + 1] = acc[0][1][1];
            C_global[warp_group_off + row1 + col1] = acc[1][1][0];
            C_global[warp_group_off + row1 + col1 + 1] = acc[1][1][1];
        }
    }
}

// Host function to launch the kernel
void run_wgmma_minimal_example() {
    const int M = 64, K = 32, N = 16;
    
    // Allocate host memory
    fp8* h_A = new fp8[M * K];
    fp8* h_B = new fp8[N * K]; 
    float* h_C = new float[M * N];
    
    // Initialize matrices with simple values
    for (int i = 0; i < M * K; i++) {
        // if (i%K < 4 && i<2*K){
        //     h_A[i] = fp8(i%K);
        //     // h_A[i] = fp8(1);
        // } else {
        //     // h_A[i] = fp8(1);
        //     h_A[i] = fp8(0);
        // }
        h_A[i] = fp8(float(i)/10.0f);
    }
    for (int i = 0; i < N * K; i++) {
        // if (i%K < 4 && i<2*K){
        //     // h_B[i] = fp8(i%K);
        //     h_B[i] = fp8(1);
        // } else {
        //     h_B[i] = fp8(1);
        //     // h_B[i] = fp8(0);
        // }
        // Make an identity matrix
        if (i%K == i/K) {
            h_B[i] = fp8(1);
        } else {
            h_B[i] = fp8(0);
        }
    }


    printf("A:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            printf("%f ", float(h_A[i * K + j]));
        }
        printf("\n");
    }
    printf("B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < K; j++) {
            printf("%f ", float(h_B[i * K + j]));
        }
        printf("\n");
    }
    
    // Allocate device memory
    fp8* d_A;
    fp8* d_B;
    float* d_C;
    
    cudaMalloc(&d_A, M * K * sizeof(fp8));
    cudaMalloc(&d_B, N * K * sizeof(fp8));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, h_A, M * K * sizeof(fp8), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(fp8), cudaMemcpyHostToDevice);
    
    // Launch kernel with 128 threads (warp group size) per block
    dim3 blockDim(128);
    dim3 gridDim(1);
    
    printf("Launching WGMMA minimal example kernel...\n");
    wgmma_minimal_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C);
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
        goto cleanup;
    }
    
    // Copy result back to host
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Print a few results for verification
    // printf("Results (first 4x4 submatrix):\n");
    // for (int i = 0; i < 4; i++) {
    //     for (int j = 0; j < 4; j++) {
    //         printf("%8.4f ", h_C[i * N + j]);
    //     }
    //     printf("\n");
    // }
    printf("C:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%8.4f ", h_C[i * N + j]);
        }
        printf("\n");
    }
    
    printf("Expected value: %f (0.5 * 0.25 * 32 = 4.0)\n", 0.5f * 0.25f * 32);
    
cleanup:
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}

// Main function for testing
int main() {
    printf("WGMMA Minimal Example\n");
    printf("Matrix A: 64x32 (FP8 E4M3)\n");
    printf("Matrix B: 16x32 (FP8 E4M3)\n");
    printf("Matrix C: 64x16 (FP32)\n");
    printf("Operation: C = A * B^T\n\n");
    
    run_wgmma_minimal_example();
    
    return 0;
}