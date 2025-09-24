#include <cuda.h>
#include <cuda_fp8.h>
#include <stdio.h>

using fp8 = __nv_fp8_e4m3;

__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory");
}

// Simplified WGMMA call using direct shared memory pointers
__device__ void simple_wgmma(float acc[8], fp8* sA, fp8* sB) {
    uint32_t sA_addr = __cvta_generic_to_shared(sA);
    uint32_t sB_addr = __cvta_generic_to_shared(sB);
    
    // Create simple descriptors - just the address encoded
    uint64_t desc_a = (sA_addr >> 4) & 0x3FFFF;
    uint64_t desc_b = (sB_addr >> 4) & 0x3FFFF;
    
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7}, "
        "%8, %9, "
        "1, 1, 1;\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3]),
          "+f"(acc[4]), "+f"(acc[5]), "+f"(acc[6]), "+f"(acc[7])
        : "l"(desc_a), "l"(desc_b)
    );
}

__global__ void simple_wgmma_kernel(
    const fp8* A_global, 
    const fp8* B_global, 
    float* C_global
) {
    __shared__ alignas(128) fp8 sA[64 * 32];
    __shared__ alignas(128) fp8 sB[16 * 32];
    
    // Load data cooperatively
    for (int i = threadIdx.x; i < 64 * 32; i += blockDim.x) {
        sA[i] = A_global[i];
    }
    for (int i = threadIdx.x; i < 16 * 32; i += blockDim.x) {
        sB[i] = B_global[i];
    }
    
    __syncthreads();
    
    // Only first warp group (128 threads) participates
    if (threadIdx.x < 128) {
        float acc[8] = {0.0f};
        
        warpgroup_arrive();
        simple_wgmma(acc, sA, sB);
        warpgroup_commit_batch();
        warpgroup_wait<0>();
        
        // Simple output: each thread writes its 8 results
        int base = threadIdx.x * 8;
        for (int i = 0; i < 8 && base + i < 64 * 16; i++) {
            C_global[base + i] = acc[i];
        }
    }
}

int main() {
    const int M = 64, K = 32, N = 16;
    
    fp8* h_A = new fp8[M * K];
    fp8* h_B = new fp8[N * K];
    float* h_C = new float[M * N];
    
    // Initialize with simple values
    for (int i = 0; i < M * K; i++) h_A[i] = fp8(0.5f);
    for (int i = 0; i < N * K; i++) h_B[i] = fp8(0.25f);
    
    fp8* d_A; fp8* d_B; float* d_C;
    cudaMalloc(&d_A, M * K * sizeof(fp8));
    cudaMalloc(&d_B, N * K * sizeof(fp8));
    cudaMalloc(&d_C, M * N * sizeof(float));
    
    cudaMemcpy(d_A, h_A, M * K * sizeof(fp8), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * K * sizeof(fp8), cudaMemcpyHostToDevice);
    
    printf("Launching simple WGMMA kernel...\n");
    simple_wgmma_kernel<<<1, 128>>>(d_A, d_B, d_C);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    printf("First few results:\n");
    for (int i = 0; i < 16; i++) {
        printf("%.2f ", h_C[i]);
        if ((i + 1) % 8 == 0) printf("\n");
    }
    
    delete[] h_A; delete[] h_B; delete[] h_C;
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
