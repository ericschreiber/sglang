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
    uint64_t desc_b = make_smem_desc(sB, 128, 128);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
        "{%0, %1, %2, %3, %4, %5, %6, %7},"
        " %8, %9,"
        " %10, %11, %12;\n"
        "}\n"
        : "+f"(d[0][0][0]), "+f"(d[0][0][1]), "+f"(d[1][0][0]), "+f"(d[1][0][1]), "+f"(d[0][1][0]), "+f"(d[0][1][1]), "+f"(d[1][1][0]), "+f"(d[1][1][1])
        : "l"(desc_a), "l"(desc_b), 
        "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)), "n"(int32_t(ScaleB)));
        }

template <int BM, int BK, int BN>
__global__ void fused_moe_w8a8_wgmma_naive_kernel(
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
    const int32_t warpM = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int32_t warpN = blockIdx.y*blockDim.y+threadIdx.y;
    //TODO should not be hardcoded
    constexpr int block_shape[2] = {128, 128};

    const int exp_idx = expert_ids[warpM];
    const fp8* exp_w = w + exp_idx * K * N;
    const int lane_id = threadIdx.x%32;
    const int w_row = warpN * BN + (lane_id>>2);

    if(warpM * BM >= num_tokens_post_padded[0])
        return;

    // if(exp_idx < 0 || exp_idx >= 257)
    //     printf("INVALID IDX %d, %d, %d\n",blockIdx.y, exp_idx, num_tokens_post_padded[0]);


    int token_dest[2];
    token_dest[0] = sorted_token_ids[warpM*BM + (lane_id>>2)];
    token_dest[1] = sorted_token_ids[warpM*BM + (lane_id>>2) + 8];
    int token_src[2];
    token_src[0] = sorted_token_ids[warpM*BM + (lane_id>>2)] / top_k;
    token_src[1] = sorted_token_ids[warpM*BM + (lane_id>>2) + 8] / top_k;

    float f_acc[2][2][2] = {0.f};
    // bool p = blockIdx.x == 1 && blockIdx.y == 5 && threadIdx.x == 0;

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
            // Load x to shared memory with proper WGMMA striding
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
                
                // Get the token source for this row
                int token_src = sorted_token_ids[warpM*BM + row] / top_k;
                
                if (token_src < M) {
                    s_x[sw_row][sw_col] = reinterpret_cast<const fp8*>(x + token_src*K + k + b_off + col)[0];
                } else {
                    s_x[sw_row][sw_col] = fp8(0);
                }
            }
            // Load w to shared memory
            // if (warp_idx == 0){
            //     for (int i = 0; i < BN; i++){
            //         int wT_row = warpN * BN + i;
            //         int wT_col = k + b_off;
            //         s_wT[i][lane_id] = reinterpret_cast<const fp8*>(exp_w + wT_row*K + wT_col)[lane_id];
            //     }
            // }
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
        
                s_wT[sw_row][sw_col] = reinterpret_cast<const fp8*>(exp_w + w_row*K + k + b_off + col)[lane_id];
            }

            __syncthreads();

            // const int w_col = (lane_id%4)*4 + k + b_off;
            // tile_w[0] = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + w_col);
            // tile_w[1] = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + w_col + 16);
            float acc_local[2][2][2] = {0.f}; // Check if this is necessary

            warpgroup_arrive();
            wgmmaM64N16K32<1, 1, 1>(acc_local, &s_x[0][0], &s_wT[0][0]);
            warpgroup_commit_batch();
            warpgroup_wait<0>();

            acc[0][0][0] += acc_local[0][0][0];
            acc[0][0][1] += acc_local[0][0][1];
            acc[0][1][0] += acc_local[0][1][0];
            acc[0][1][1] += acc_local[0][1][1];
            acc[1][0][0] += acc_local[1][0][0];
            acc[1][0][1] += acc_local[1][0][1];
            acc[1][1][0] += acc_local[1][1][0];
            acc[1][1][1] += acc_local[1][1][1];
        }


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
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[0]*N + warpN * BN + (lane_id%4)*2) = __nv_bfloat162(f_acc[0][0][0], f_acc[0][0][1]);;
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[0]*N + warpN * BN + (lane_id%4)*2 + 8) = __nv_bfloat162(f_acc[0][1][0], f_acc[0][1][1]);;
    }
    if (token_src[1] < M)
    {
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[1]*N + warpN * BN + (lane_id%4)*2) = __nv_bfloat162(f_acc[1][0][0], f_acc[1][0][1]);;
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[1]*N + warpN * BN + (lane_id%4)*2 + 8) = __nv_bfloat162(f_acc[1][1][0], f_acc[1][1][1]);;
    }
}

void fused_moe_w8a8_wgmma_naive(
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
    constexpr int BM = 64;
    constexpr int BK = 32;
    constexpr int BN = 16;
    // constexpr int num_warps_x_WGMMA = 4;
    // constexpr int num_warps_y = 1;
    // dim3 dimBlock(32*num_warps_x_WGMMA, num_warps_y, 1);
    dim3 dimBlock(128, 1, 1); // One warp group per block
    // dim3 dimGrid(std::ceil((float)sorted_num/(BM*num_warps_x_WGMMA)), std::ceil((float)N/(BN*num_warps_y)),1);
    dim3 dimGrid(std::ceil((float)sorted_num/BM), std::ceil((float)N/BN), 1);
    fused_moe_w8a8_wgmma_naive_kernel<BM, BK, BN><<<dimGrid, dimBlock>>>(
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
}
