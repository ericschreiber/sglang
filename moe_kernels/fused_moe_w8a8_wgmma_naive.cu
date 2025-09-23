#include <cuda.h>
#include <cuda_fp8.h>
#include <stdio.h>

// Not gonna type all that
using fp8 = __nv_fp8_e4m3;

template <int BM, int BK, int BN>
__global__ void fused_moe_w8a8_unroll_block_kernel(
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

    __shared__ alignas(16) fp8 tile_xT[8][128];     // col-major transpose, so same as in global memory
    __shared__ alignas(16) fp8 tile_wT[64][128];    // row-major transpose, so same as in global memory

    // bool p = blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0;

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


        // Load tiles to SMEM
        if (threadIdx.x == 0)
        {
            // Load tile of w using 1 TMA transfer
            cuda::memcpy_async(tile_wT, exp_w + w_row*K + block*block_shape[0], sizeof(fp8)*64*128, bar);
            barrier::arrival_token token = bar.arrive();
            bar.wait(std::move(token));
        }




        // int b_off = block * block_shape[0];
        // float acc[4] = {0.f};

        // uint4 loaded;
        // if (token_src[0] < M)
        // {
        //     loaded = reinterpret_cast<const uint4*>(x + token_src[0]*K + b_off )[lane_id%4];
        //     tile_x[0][0][0] = loaded.x;
        //     tile_x[0][0][1] = loaded.y; 
        //     tile_x[0][0][2] = loaded.z;
        //     tile_x[0][0][3] = loaded.w;
        //     loaded = reinterpret_cast<const uint4*>(x + token_src[0]*K + b_off + 64)[lane_id%4];
        //     tile_x[0][1][0] = loaded.x;
        //     tile_x[0][1][1] = loaded.y;
        //     tile_x[0][1][2] = loaded.z;
        //     tile_x[0][1][3] = loaded.w;
        // }
        // if (token_src[1] < M)
        // {
        //     loaded = reinterpret_cast<const uint4*>(x + token_src[1]*K + b_off)[lane_id%4];
        //     tile_x[1][0][0] = loaded.x;
        //     tile_x[1][0][1] = loaded.y;
        //     tile_x[1][0][2] = loaded.z;
        //     tile_x[1][0][3] = loaded.w;
        //     loaded = reinterpret_cast<const uint4*>(x + token_src[1]*K + b_off + 64)[lane_id%4];
        //     tile_x[1][1][0] = loaded.x;
        //     tile_x[1][1][1] = loaded.y;
        //     tile_x[1][1][2] = loaded.z;
        //     tile_x[1][1][3] = loaded.w;
        // }

        // const int w_col = (lane_id%4)*16 + b_off;
        // // tile_w[0] = *reinterpret_cast<const uint4*>(exp_w + w_row*K + w_col);
        // loaded = *reinterpret_cast<const uint4*>(exp_w + w_row*K + w_col);
        // tile_w[0][0] = loaded.x;
        // tile_w[0][1] = loaded.y;
        // tile_w[0][2] = loaded.z;
        // tile_w[0][3] = loaded.w;
        // // // tile_w[1] = *reinterpret_cast<const uint4*>(exp_w + w_row*K + w_col + 64);
        // loaded = *reinterpret_cast<const uint4*>(exp_w + w_row*K + w_col + 64);
        // tile_w[1][0] = loaded.x;
        // tile_w[1][1] = loaded.y;
        // tile_w[1][2] = loaded.z;
        // tile_w[1][3] = loaded.w;

        // // tile_w[0][0] = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + w_col);
        // // tile_w[0][1] = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + w_col + 4);
        // // tile_w[0][2] = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + w_col + 8);
        // // tile_w[0][3] = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + w_col + 12);

        // // tile_w[1][0] = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + w_col + 64);
        // // tile_w[1][1] = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + w_col + 68);
        // // tile_w[1][2] = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + w_col + 72);
        // // tile_w[1][3] = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + w_col + 76);


        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(tile_x[0][0][0]), "r"(tile_x[1][0][0]), "r"(tile_x[0][1][0]), "r"(tile_x[1][1][0]), "r"(tile_w[0][0]), "r"(tile_w[1][0]));

        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(tile_x[0][0][1]), "r"(tile_x[1][0][1]), "r"(tile_x[0][1][1]), "r"(tile_x[1][1][1]), "r"(tile_w[0][1]), "r"(tile_w[1][1]));

        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(tile_x[0][0][2]), "r"(tile_x[1][0][2]), "r"(tile_x[0][1][2]), "r"(tile_x[1][1][2]), "r"(tile_w[0][2]), "r"(tile_w[1][2]));

        asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                : "r"(tile_x[0][0][3]), "r"(tile_x[1][0][3]), "r"(tile_x[0][1][3]), "r"(tile_x[1][1][3]), "r"(tile_w[0][3]), "r"(tile_w[1][3]));
        
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
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[0]*N + warpN * BN + (lane_id%4)*2) = __nv_bfloat162(f_acc[0], f_acc[1]);;
    }
    if (token_src[1] < M)
    {
        *reinterpret_cast<__nv_bfloat162*>(out + token_dest[1]*N + warpN * BN + (lane_id%4)*2) = __nv_bfloat162(f_acc[2], f_acc[3]);;
    }
}

void fused_moe_w8a8_unrollK(
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
    constexpr int BN = 8;
    constexpr int num_warps_x = 4;
    constexpr int num_warps_y = 2;

    dim3 dimBlock(32*num_warps_x, num_warps_y, 1);
    dim3 dimGrid(std::ceil((float)N/(BN*num_warps_x)), std::ceil((float)sorted_num/(BM*num_warps_y)), 1);

    fused_moe_w8a8_unroll_block_kernel<BM, BK, BN><<<dimGrid, dimBlock>>>(
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
