#include <cuda.h>
#include <cuda_fp8.h>
#include <stdio.h>

// Not gonna type all that
using fp8 = __nv_fp8_e4m3;

template <int BM, int BK, int BN>
__global__ void fused_moe_w8a8_m16n8k128_kernel(
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
    const int32_t warpN = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int32_t warpM = blockIdx.y*blockDim.y+threadIdx.y;

    //TODO should not be hardcoded
    constexpr int block_shape[2] = {128, 128};

    const int exp_idx = expert_ids[warpM];
    const fp8* exp_w = w + exp_idx * K * N;
    const int lane_id = threadIdx.x%32;
    const int w_row = warpN * BN + (lane_id>>2);

    if(warpM * BM >= num_tokens_post_padded[0])
        return;


    int token_dest[2];
    token_dest[0] = sorted_token_ids[warpM*BM + (lane_id>>2)];
    token_dest[1] = sorted_token_ids[warpM*BM + (lane_id>>2) + 8];
    int token_src[2];
    token_src[0] = sorted_token_ids[warpM*BM + (lane_id>>2)] / top_k;
    token_src[1] = sorted_token_ids[warpM*BM + (lane_id>>2) + 8] / top_k;

    // For m16n8k128, we need to load 128 FP8 elements (32 uint32_t values) for each tensor core operation
    uint32_t tile_x[16]; // 16 * 4 bytes = 64 bytes = 128 FP8 elements for 2 rows
    uint32_t tile_w[8];  // 8 * 4 bytes = 32 bytes = 64 FP8 elements for weight
    float f_acc[4] = {0.f};
    // bool p = blockIdx.x == 1 && blockIdx.y == 5 && threadIdx.x == 0;

    const int scale_cols_x = K/block_shape[1];
    const int scale_rows_w = N/block_shape[1];
    const int scale_cols_w = K/block_shape[0];

    for (int block=0; block < K/block_shape[0]; block += 1)
    {
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
        float acc[4] = {0.f};
        
        // For m16n8k128, we process the entire block_shape[0] (128) elements in one tensor core operation
        // Load 128 FP8 elements for activation (64 elements per row, 2 rows)
        if (token_src[0] < M)
        {
            // Load 64 FP8 elements (16 uint32_t) for first row
            for (int i = 0; i < 8; i++) {
                tile_x[i] = reinterpret_cast<const uint32_t*>(x + token_src[0]*K + b_off)[lane_id%4 + i*4];
            }
        }
        if (token_src[1] < M)
        {
            // Load 64 FP8 elements (16 uint32_t) for second row
            for (int i = 0; i < 8; i++) {
                tile_x[8 + i] = reinterpret_cast<const uint32_t*>(x + token_src[1]*K + b_off)[lane_id%4 + i*4];
            }
        }

        // Load 64 FP8 elements (8 uint32_t) for weight
        for (int i = 0; i < 8; i++) {
            const int w_col = (lane_id%4)*4 + i*16 + b_off;
            tile_w[i] = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + w_col);
        }

        // Perform tensor core operation with 128 elements in K dimension
        asm volatile("mma.sync.aligned.m16n8k128.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19}, {%20, %21, %22, %23, %24, %25, %26, %27}, {%0, %1, %2, %3};"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(tile_x[0]), "r"(tile_x[1]), "r"(tile_x[2]), "r"(tile_x[3]), 
                  "r"(tile_x[4]), "r"(tile_x[5]), "r"(tile_x[6]), "r"(tile_x[7]),
                  "r"(tile_x[8]), "r"(tile_x[9]), "r"(tile_x[10]), "r"(tile_x[11]),
                  "r"(tile_x[12]), "r"(tile_x[13]), "r"(tile_x[14]), "r"(tile_x[15]),
                  "r"(tile_w[0]), "r"(tile_w[1]), "r"(tile_w[2]), "r"(tile_w[3]),
                  "r"(tile_w[4]), "r"(tile_w[5]), "r"(tile_w[6]), "r"(tile_w[7]));

        // Apply scaling and accumulate
        if (token_src[0] < M)
        {
            f_acc[0] += scale_x[0] * scale_w * acc[0];
            f_acc[1] += scale_x[0] * scale_w * acc[1];
        }
        if (token_src[1] < M)
        {
            f_acc[2] += scale_x[1] * scale_w * acc[2];
            f_acc[3] += scale_x[1] * scale_w * acc[3];
        }
    }
    if (token_src[0] < M)
    {
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[0]*N + warpN * BN + (lane_id%4)*2) = __nv_bfloat162(f_acc[0], f_acc[1]);
    }
    if (token_src[1] < M)
    {
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[1]*N + warpN * BN + (lane_id%4)*2) = __nv_bfloat162(f_acc[2], f_acc[3]);
    }
}

void fused_moe_w8a8_m16n8k128(
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
    constexpr int BM = 16;
    constexpr int BK = 128;
    constexpr int BN = 8;
    constexpr int num_warps_x = 4;
    constexpr int num_warps_y = 2;
    dim3 dimBlock(32*num_warps_x, num_warps_y, 1);
    dim3 dimGrid(std::ceil((float)N/(BN*num_warps_x)), std::ceil((float)sorted_num/(BM*num_warps_y)), 1);
    fused_moe_w8a8_m16n8k128_kernel<BM, BK, BN><<<dimGrid, dimBlock>>>(
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
