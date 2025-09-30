#include <cuda.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <cassert>
#include <cuda/barrier>
#include <cuda/ptx>

#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap

// Not gonna type all that
using fp8 = __nv_fp8_e4m3;

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

template<int ScaleD, int ScaleA, int ScaleB>
__device__ void wgmmaM64N16K32(float d[2][2][2], fp8* sA, fp8* sB) {
    uint64_t desc_a = make_smem_desc(sA, 1024, 128);
    uint64_t desc_b = make_smem_desc(sB, 256, 128);
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

template <int BM, int BK, int BN>
__global__ void fused_moe_w8a8_wgmma_naive_kernel_v2(
        const fp8* __restrict__ x,
        const float* __restrict__ x_scale,
        const fp8* __restrict__ w,
        const float* __restrict__ w_scale,
        __nv_bfloat16* __restrict__ out,
        const int* __restrict__ sorted_token_ids,
        const int* __restrict__ expert_ids,
        const int* __restrict__ num_tokens_post_padded,
        const int top_k,
        int M,
        int K,
        int N
        )
{
    const int32_t warp_idx = threadIdx.x / 32;
    const int32_t lane_idx = threadIdx.x % 32;
    const int32_t warp_group_N = (blockIdx.x*blockDim.x+threadIdx.x) / 32; // warps 0-3 for each warp group
    const int32_t warp_group_M = blockIdx.y; // y is only 1D

    if(warp_group_M * BM >= num_tokens_post_padded[0])
        return;
    
    //TODO should not be hardcoded
    constexpr int block_shape[2] = {128, 128};
    
    const int exp_idx = expert_ids[warp_group_M];
    const fp8* exp_w = w + exp_idx * K * N;
    const int w_row = warp_group_N * BN + (lane_idx>>2);

    int token_dest[2];
    int idx = warp_group_M*BM + warp_idx * 16 + (lane_idx>>2);
    token_dest[0] = sorted_token_ids[warp_group_M*BM + warp_idx * 16 + (lane_idx>>2)];
    token_dest[1] = sorted_token_ids[warp_group_M*BM + warp_idx * 16 + (lane_idx>>2) + 8];
    // if (idx < num_tokens_post_padded[0]) {
    //     token_dest[0] = sorted_token_ids[warp_group_M*BM + warp_idx * 16 + (lane_idx>>2)];
    // } else {
    //     token_dest[0] = M;
    // }
    // if (idx + 8 < num_tokens_post_padded[0]) {
    //     token_dest[1] = sorted_token_ids[warp_group_M*BM + warp_idx * 16 + (lane_idx>>2) + 8];
    // } else {
    //     token_dest[1] = M;
    // }

    if (warp_group_N == 15 && lane_idx == 0) {
        printf("warp_group_N: %d, lane_idx: %d, ThreadIdx.x: %d, BlockIdx.x: %d, BlockIdx.y: %d, token_dest[0]: %d, token_dest[1]: %d\n", warp_group_N, lane_idx, threadIdx.x, blockIdx.x, blockIdx.y, token_dest[0], token_dest[1]);
    }

    if (token_dest[0] == 1 || token_dest[1] == 1) {
        printf("token_dest[0]: %d, token_dest[1]: %d, threadIdx.x: %d, blockIdx.x: %d, blockIdx.y: %d, threadIdx.y: %d, block: %d, k: %d\n",
                token_dest[0], token_dest[1], threadIdx.x, blockIdx.x, blockIdx.y, threadIdx.y);
    }

    int token_src[2];
    token_src[0] = sorted_token_ids[warp_group_M*BM + warp_idx * 16 + (lane_idx>>2)] / top_k;
    token_src[1] = sorted_token_ids[warp_group_M*BM + warp_idx * 16 + (lane_idx>>2) + 8] / top_k;
    // if (idx < num_tokens_post_padded[0]) {
    //     token_src[0] = token_dest[0] / top_k;
    // } else {
    //     token_src[0] = M;
    // }
    // if (idx + 8 < num_tokens_post_padded[0]) {
    //     token_src[1] = token_dest[1] / top_k;
    // } else {
    //     token_src[1] = M;
    // }

    float f_acc[2][2][2] = {0.f};

    for (int block=0; block < K/block_shape[0]; block += 1)
    {
        const int scale_cols_x = K/block_shape[1];
        const int scale_rows_w = N/block_shape[1];
        const int scale_cols_w = K/block_shape[0];

        float scale_x[2];
        if (token_src[0] < M)
        {
            scale_x[0] = x_scale[(token_src[0])*scale_cols_x + block];
        }
        if (token_src[1] < M)
        {
            scale_x[1] = x_scale[(token_src[1])*scale_cols_x + block];
        }

        float scale_w = w_scale[exp_idx * scale_rows_w * scale_cols_w + (w_row/block_shape[1])*scale_cols_w + block];

        int b_off = block * block_shape[0];
        float acc[2][2][2] = {0.f};
        __shared__ alignas(128) fp8 s_x[BM][BK];
        __shared__ alignas(128) fp8 s_wT[BN][BK];
        for(int k = 0; k < block_shape[0]; k += BK)
        {   
            /// Load x to shared memory with proper WGMMA striding
            for (int i = threadIdx.x; i < BM * BK; i += blockDim.x) {
                int row = i / BK;  // Which row in the BM x BK tile
                int col = i % BK;  // Which column in the BM x BK tile

                // Apply WGMMA striding pattern
                int m0 = row % 8;
                int m1 = row / 8;
                int k0 = col % 16;
                int k1 = col / 16;
                int ofs = m0 * 16 + m1 * 128 + k0 + k1 * 1024;
                
                int sw_row = ofs / BK;
                int sw_col = ofs % BK;

                // if (warp_group_M*BM + row >= num_tokens_post_padded[0]) {
                //     s_x[sw_row][sw_col] = fp8(0);
                // }
                // else {
                // Get the token source for this row
                int token_src_local = sorted_token_ids[warp_group_M*BM + row] / top_k;

                if (token_src_local < M) {
                    s_x[sw_row][sw_col] = x[token_src_local*K + k + b_off + col];
                }
                else {
                    s_x[sw_row][sw_col] = fp8(0);
                }
                // }
            }
            __syncthreads();
            // Print s_x
            if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.y == 0 && block == 0 && k == 0) {
                for (int i = 0; i < BM; i++) {
                    for (int j = 0; j < BK; j++) {
                        printf("s_x[%d][%d]: %f", i, j, float(s_x[i][j]));
                    }
                    printf("\n");
                }
            }
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
        
                s_wT[sw_row][sw_col] = exp_w[row*K + k + b_off + col];
            }
            __syncthreads();
            
            float acc_local[2][2][2] = {0.f}; // Check if this is necessary

            warpgroup_arrive();
            wgmmaM64N16K32<1, 1, 1>(acc_local, &s_x[0][0], &s_wT[0][0]);
            warpgroup_commit_batch();
            warpgroup_wait<0>();
            __syncthreads();
            if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 1 && threadIdx.y == 0 && block == 0 && k == 0) {
                printf("acc_local: %f, %f, %f, %f\n", acc_local[0][0][0], acc_local[0][0][1], acc_local[0][1][0], acc_local[0][1][1]);
                printf("acc_local: %f, %f, %f, %f\n", acc_local[1][0][0], acc_local[1][0][1], acc_local[1][1][0], acc_local[1][1][1]);
            }

            // Check if this is necessary
            acc[0][0][0] += acc_local[0][0][0];
            acc[0][0][1] += acc_local[0][0][1];
            acc[0][1][0] += acc_local[0][1][0];
            acc[0][1][1] += acc_local[0][1][1];
            acc[1][0][0] += acc_local[1][0][0];
            acc[1][0][1] += acc_local[1][0][1];
            acc[1][1][0] += acc_local[1][1][0];
            acc[1][1][1] += acc_local[1][1][1];
        }
        __syncthreads();

        if (token_src[0] < M)
        {
            f_acc[0][0][0] += scale_x[0] * scale_w * acc[0][0][0];
            f_acc[0][0][1] += scale_x[0] * scale_w * acc[0][0][1];
            f_acc[0][1][0] += scale_x[0] * scale_w * acc[0][1][0];
            f_acc[0][1][1] += scale_x[0] * scale_w * acc[0][1][1];
        }
        if (token_src[1] < M)
        {
            f_acc[1][0][0] += scale_x[1] * scale_w * acc[1][0][0];
            f_acc[1][0][1] += scale_x[1] * scale_w * acc[1][0][1];
            f_acc[1][1][0] += scale_x[1] * scale_w * acc[1][1][0];
            f_acc[1][1][1] += scale_x[1] * scale_w * acc[1][1][1];
        }
    }
    if (token_src[0] < M)
    {
        if (token_dest[0] == 0){
            printf("warp_group_N: %d, lane_idx: %d, ThreadIdx.x: %d, BlockIdx.x: %d, BlockIdx.y: %d\n", warp_group_N, lane_idx, threadIdx.x, blockIdx.x, blockIdx.y);
        }
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[0]*N + warp_group_N * BN + (lane_idx%4)*2) = __nv_bfloat162(f_acc[0][0][0], f_acc[0][0][1]);;
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[0]*N + warp_group_N * BN + (lane_idx%4)*2 + 8) = __nv_bfloat162(f_acc[0][1][0], f_acc[0][1][1]);;
    }
    if (token_src[1] < M)
    {
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[1]*N + warp_group_N * BN + (lane_idx%4)*2) = __nv_bfloat162(f_acc[1][0][0], f_acc[1][0][1]);;
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[1]*N + warp_group_N * BN + (lane_idx%4)*2 + 8) = __nv_bfloat162(f_acc[1][1][0], f_acc[1][1][1]);;
    }
}

void fused_moe_w8a8_wgmma_naive_v2(
        const fp8* x,
        const float* x_scale,
        const fp8* w, const float* w_scale,
        __nv_bfloat16* out,
        const int* sorted_token_ids,
        const int* expert_ids,
        const int* num_tokens_post_padded,
        const int top_k,
        int M,
        int K,
        int N,
        int sorted_num
        )
{
    // 1st row of 1st tile is correct
    // 2nd row of 1st tile is wrong. -> Correct loading of x?
    // Tile to the left is all 0. -> Is a tile running there?
    constexpr int BM = 64;
    constexpr int BK = 32;
    constexpr int BN = 16;
    dim3 dimBlock(128, 1, 1); // One warp group per block
    dim3 dimGrid(std::ceil((float)N/(BN)), std::ceil((float)sorted_num/(BM)),1);

    fused_moe_w8a8_wgmma_naive_kernel_v2<BM, BK, BN><<<dimGrid, dimBlock>>>(
            x,
            x_scale,
            w,
            w_scale,
            out,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k,
            M,
            K,
            N
            );

     // Check for kernel launch errors
     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) {
         fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
     }
     
     // Wait for kernel to complete
     err = cudaDeviceSynchronize();
     if (err != cudaSuccess) {
         fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
     }
}
