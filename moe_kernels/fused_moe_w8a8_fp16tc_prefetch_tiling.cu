#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <stdio.h>

// Not gonna type all that
using fp8 = __nv_fp8_e4m3;

template <int BM, int BK, int BN, int PF, int tile_factor_K>
__global__ void fused_moe_w8a8_fp16tc_prefetch_tiling_kernel(
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

    float scale_x[PF][2];
    float scale_w[PF];
    const int scale_cols_x = K/block_shape[1];
    const int scale_rows_w = N/block_shape[1];
    const int scale_cols_w = K/block_shape[0];

    uint32_t tile_x[PF][tile_factor_K][4];  // Prefetched x tiles (fp16 format)
    uint32_t tile_w[PF][tile_factor_K][2];  // Prefetched w tiles (fp16 format)
    float f_acc[4] = {0.f};
    int tc_stage = 0;
    int scale_stage = 0;

    auto load_tiles = [&](int off, int stage)
    {
        // Load tile_factor_K tiles for K dimension
        for (int k_tile = 0; k_tile < tile_factor_K; k_tile++)
        {
            int k_offset = off + k_tile * BK;
            
            // Load x data and convert fp8 to fp16
            uint16_t load_x[4] = {0};
            if (token_src[0] < M && k_offset < K)
            {
                load_x[0] = reinterpret_cast<const uint16_t*>(x + token_src[0]*K + k_offset)[lane_id%4];
                load_x[2] = reinterpret_cast<const uint16_t*>(x + token_src[0]*K + k_offset + 8)[lane_id%4];
            }
            if (token_src[1] < M && k_offset < K)
            {
                load_x[1] = reinterpret_cast<const uint16_t*>(x + token_src[1]*K + k_offset)[lane_id%4];
                load_x[3] = reinterpret_cast<const uint16_t*>(x + token_src[1]*K + k_offset + 8)[lane_id%4];
            }
            
            // Convert each fp8 to fp16
            __half load_xfp16[8];
            for (int i = 0; i < 8; i++)
            {
                load_xfp16[i] = __half(reinterpret_cast<const fp8*>(&load_x)[i]);
            }
            
            // Store as uint32_t tiles
            tile_x[stage][k_tile][0] = reinterpret_cast<uint32_t*>(&load_xfp16)[0];
            tile_x[stage][k_tile][1] = reinterpret_cast<uint32_t*>(&load_xfp16)[1];
            tile_x[stage][k_tile][2] = reinterpret_cast<uint32_t*>(&load_xfp16)[2];
            tile_x[stage][k_tile][3] = reinterpret_cast<uint32_t*>(&load_xfp16)[3];

            // Load w data and convert fp8 to fp16
            const int w_col = (lane_id%4)*2 + k_offset;
            
            if (k_offset < K)
            {
                // Load first contiguous pair (fp8 → fp16 → pack into uint32_t)
                fp8 fp8_pair1[2];
                *reinterpret_cast<uint16_t*>(fp8_pair1) = *reinterpret_cast<const uint16_t*>(&exp_w[w_row*K + w_col]);
                half2 h2_pair1 = make_half2(__half(fp8_pair1[0]), __half(fp8_pair1[1]));
                tile_w[stage][k_tile][0] = *reinterpret_cast<uint32_t*>(&h2_pair1);
                
                // Load second contiguous pair (fp8 → fp16 → pack into uint32_t)
                fp8 fp8_pair2[2];
                *reinterpret_cast<uint16_t*>(fp8_pair2) = *reinterpret_cast<const uint16_t*>(&exp_w[w_row*K + w_col + 8]);
                half2 h2_pair2 = make_half2(__half(fp8_pair2[0]), __half(fp8_pair2[1]));
                tile_w[stage][k_tile][1] = *reinterpret_cast<uint32_t*>(&h2_pair2);
            }
            else
            {
                // Zero out tiles beyond K boundary
                tile_w[stage][k_tile][0] = 0;
                tile_w[stage][k_tile][1] = 0;
            }
        }
    };

    auto load_scales = [&](int off, int stage)
    {
        if (token_src[0] < M)
        {
            scale_x[stage][0] = x_scale[(token_src[0])*scale_cols_x + off];
        }
        if (token_src[1] < M)
        {
            scale_x[stage][1] = x_scale[(token_src[1])*scale_cols_x + off];
        }
        scale_w[stage] = w_scale[exp_idx * scale_rows_w * scale_cols_w + (w_row/block_shape[1])*scale_cols_w + off];
    };

    // Prefetch initial tiles
    for(int stage = 0; stage < PF && stage*BK*tile_factor_K < K; stage++)
    {
        load_tiles(stage*BK*tile_factor_K, stage);
    }

    // Prefetch initial scales
    for(int stage = 0; stage < PF && stage < K/block_shape[0]; stage++)
    {
        load_scales(stage, stage);
    }

    for (int block = 0; block < K/block_shape[0]; block += 1)
    {
        int b_off = block * block_shape[0];
        float acc[4] = {0.f};
        
        for(int k = 0; k < block_shape[0]; k += BK*tile_factor_K)
        {
            // Compute using tensor cores for each K tile
            for (int k_tile = 0; k_tile < tile_factor_K && k + k_tile*BK < block_shape[0]; k_tile++)
            {
                // Use fp16 tensor core instruction
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(tile_x[tc_stage][k_tile][0]), "r"(tile_x[tc_stage][k_tile][1]), "r"(tile_x[tc_stage][k_tile][2]), "r"(tile_x[tc_stage][k_tile][3]), 
                          "r"(tile_w[tc_stage][k_tile][0]), "r"(tile_w[tc_stage][k_tile][1]));
            }
            
            // Prefetch next tiles if available
            if(b_off + k + PF*BK*tile_factor_K < K)
                load_tiles(b_off + k + PF*BK*tile_factor_K, tc_stage);
            
            tc_stage = (tc_stage + 1) % PF;
        }
        
        // Apply scaling and accumulate
        if (token_src[0] < M)
        {
            f_acc[0] += scale_x[scale_stage][0] * scale_w[scale_stage] * acc[0];
            f_acc[1] += scale_x[scale_stage][0] * scale_w[scale_stage] * acc[1];
        }
        if (token_src[1] < M)
        {
            f_acc[2] += scale_x[scale_stage][1] * scale_w[scale_stage] * acc[2];
            f_acc[3] += scale_x[scale_stage][1] * scale_w[scale_stage] * acc[3];
        }
        
        // Prefetch next scales if available
        if(block + PF < K/block_shape[0])
        {
            load_scales(block + PF, scale_stage);
        }
        scale_stage = (scale_stage + 1) % PF;
    }
    
    // Write results
    if (token_src[0] < M)
    {
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[0]*N + warpN * BN + (lane_id%4)*2) = __nv_bfloat162(f_acc[0], f_acc[1]);
    }
    if (token_src[1] < M)
    {
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[1]*N + warpN * BN + (lane_id%4)*2) = __nv_bfloat162(f_acc[2], f_acc[3]);
    }
}

void fused_moe_w8a8_fp16tc_prefetch_tiling(
        const fp8* x,
        const float* x_scale,
        const fp8* w, 
        const float* w_scale,
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
    constexpr int BK = 16;
    constexpr int BN = 8;
    constexpr int PF = 1;
    constexpr int tile_factor_K = 4;
    constexpr int num_warps_x = 4;
    constexpr int num_warps_y = 2;
    dim3 dimBlock(32*num_warps_x, num_warps_y, 1);
    dim3 dimGrid(std::ceil((float)N/(BN*num_warps_x)), std::ceil((float)sorted_num/(BM*num_warps_y)), 1);
    
    // TODO get some JIT mechanism instead of hard coding
    if (top_k == 1)
    {
        fused_moe_w8a8_fp16tc_prefetch_tiling_kernel<BM, BK, BN, PF, tile_factor_K><<<dimGrid, dimBlock>>>(
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
    else
    {
        fused_moe_w8a8_fp16tc_prefetch_tiling_kernel<BM, BK, BN, PF, tile_factor_K><<<dimGrid, dimBlock>>>(
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
}