#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <stdio.h>

// Not gonna type all that
using fp8 = __nv_fp8_e4m3;

template <int BM, int BK, int BN>
__global__ void fused_moe_w8a8_fp16tc_kernel(
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
    //TODO should not be hardcoded
    constexpr int block_shape[2] = {128, 128};

    const int exp_idx = expert_ids[blockIdx.y];
    const fp8* exp_w = w + exp_idx * K * N;
    const int lane_id = threadIdx.x%32;
    const int w_row = blockIdx.x * BN + (lane_id>>2);

    if(blockIdx.y * BM >= num_tokens_post_padded[0])
        return;

    int token_dest[2];
    token_dest[0] = sorted_token_ids[blockIdx.y*BM + (lane_id>>2)];
    token_dest[1] = sorted_token_ids[blockIdx.y*BM + (lane_id>>2) + 8];
    int token_src[2];
    token_src[0] = sorted_token_ids[blockIdx.y*BM + (lane_id>>2)] / top_k;
    token_src[1] = sorted_token_ids[blockIdx.y*BM + (lane_id>>2) + 8] / top_k;

    uint32_t tile_w[2];
    float f_acc[4] = {0.f};
    bool p = blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0;

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
        float acc[4] = {0.f};
        for(int k = 0; k < block_shape[0]; k += BK)
        {   
            uint16_t load_x[4];
            if (token_src[0] < M)
            {
                load_x[0] = reinterpret_cast<const uint16_t*>(x + token_src[0]*K + k + b_off)[lane_id%4];
                load_x[2] = reinterpret_cast<const uint16_t*>(x + token_src[0]*K + k + b_off + 8)[lane_id%4];
            }
            if (token_src[1] < M)
            {
                load_x[1] = reinterpret_cast<const uint16_t*>(x + token_src[1]*K + k + b_off)[lane_id%4];
                load_x[3] = reinterpret_cast<const uint16_t*>(x + token_src[1]*K + k + b_off + 8)[lane_id%4];
            }
            
            // convert each fp8 to fp16
            __half load_xfp16[8];
            for (int i = 0; i<8; i++)
            {
                load_xfp16[i] = __half(reinterpret_cast<const fp8*>(&load_x)[i]);
            }

            uint32_t* tile_x = reinterpret_cast<uint32_t*>(&load_xfp16);
            

            fp8 tmp[2];
            for (int i = 0; i<2; i++)
            {
                const int w_col = (lane_id%4)*2 + i + k + b_off;
                tmp[i] = exp_w[w_row*K + w_col];
            }
            __half tmpfp16[2];
            tmpfp16[0] = __half(tmp[0]);
            tmpfp16[1] = __half(tmp[1]);
            tile_w[0] = *reinterpret_cast<uint32_t*>(&tmpfp16);
            fp8 tmp2[2];
            for (int i = 0; i<2; i++)
            {
                const int w_col = (lane_id%4)*2 + i + k + b_off + 8;
                tmp2[i] = exp_w[w_row*K + w_col];
            }
            __half tmpfp16_2[2];
            tmpfp16_2[0] = __half(tmp2[0]);
            tmpfp16_2[1] = __half(tmp2[1]);
            tile_w[1] = *reinterpret_cast<uint32_t*>(&tmpfp16_2);
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                    : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                    : "r"(tile_x[0]), "r"(tile_x[1]), "r"(tile_x[2]), "r"(tile_x[3]), "r"(tile_w[0]), "r"(tile_w[1]));
        }
        f_acc[0] += scale_x[0] * scale_w * acc[0];
        f_acc[1] += scale_x[0] * scale_w * acc[1];
        f_acc[2] += scale_x[1] * scale_w * acc[2];
        f_acc[3] += scale_x[1] * scale_w * acc[3];
            
    }
    if (token_src[0] < M)
    {
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[0]*N + blockIdx.x * BN + (lane_id%4)*2) = __nv_bfloat162(f_acc[0], f_acc[1]);;
    }
    if (token_src[1] < M)
    {
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[1]*N + blockIdx.x * BN + (lane_id%4)*2) = __nv_bfloat162(f_acc[2], f_acc[3]);;
    }
}

void fused_moe_w8a8_fp16tc(
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
    constexpr int BK = 16;
    constexpr int BN = 8;
    dim3 dimBlock(32,1,1);
    dim3 dimGrid(std::ceil((float)N/BN), std::ceil((float)sorted_num/BM), 1);
    fused_moe_w8a8_fp16tc_kernel<BM, BK, BN><<<dimGrid, dimBlock>>>(
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
