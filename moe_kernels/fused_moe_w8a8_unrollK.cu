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
    
        float f_acc[4] = {0.f};
    
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
            
            // Direct loads in MMA instructions - fully unrolled
            // k_step = 0
            {
                uint32_t x0 = (token_src[0] < M) ? 
                    reinterpret_cast<const uint32_t*>(x + token_src[0]*K + b_off)[lane_id%4] : 0;
                uint32_t x1 = (token_src[1] < M) ? 
                    reinterpret_cast<const uint32_t*>(x + token_src[1]*K + b_off)[lane_id%4] : 0;
                uint32_t x2 = (token_src[0] < M) ? 
                    reinterpret_cast<const uint32_t*>(x + token_src[0]*K + b_off + 16)[lane_id%4] : 0;
                uint32_t x3 = (token_src[1] < M) ? 
                    reinterpret_cast<const uint32_t*>(x + token_src[1]*K + b_off + 16)[lane_id%4] : 0;
                
                uint32_t w0 = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + (lane_id%4)*4 + b_off);
                uint32_t w1 = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + (lane_id%4)*4 + b_off + 16);
                
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(x0), "r"(x1), "r"(x2), "r"(x3), "r"(w0), "r"(w1));
            }
            
            // k_step = 1
            {
                uint32_t x0 = (token_src[0] < M) ? 
                    reinterpret_cast<const uint32_t*>(x + token_src[0]*K + 32 + b_off)[lane_id%4] : 0;
                uint32_t x1 = (token_src[1] < M) ? 
                    reinterpret_cast<const uint32_t*>(x + token_src[1]*K + 32 + b_off)[lane_id%4] : 0;
                uint32_t x2 = (token_src[0] < M) ? 
                    reinterpret_cast<const uint32_t*>(x + token_src[0]*K + 32 + b_off + 16)[lane_id%4] : 0;
                uint32_t x3 = (token_src[1] < M) ? 
                    reinterpret_cast<const uint32_t*>(x + token_src[1]*K + 32 + b_off + 16)[lane_id%4] : 0;
                
                uint32_t w0 = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + (lane_id%4)*4 + 32 + b_off);
                uint32_t w1 = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + (lane_id%4)*4 + 32 + b_off + 16);
                
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(x0), "r"(x1), "r"(x2), "r"(x3), "r"(w0), "r"(w1));
            }
            
            // k_step = 2
            {
                uint32_t x0 = (token_src[0] < M) ? 
                    reinterpret_cast<const uint32_t*>(x + token_src[0]*K + 64 + b_off)[lane_id%4] : 0;
                uint32_t x1 = (token_src[1] < M) ? 
                    reinterpret_cast<const uint32_t*>(x + token_src[1]*K + 64 + b_off)[lane_id%4] : 0;
                uint32_t x2 = (token_src[0] < M) ? 
                    reinterpret_cast<const uint32_t*>(x + token_src[0]*K + 64 + b_off + 16)[lane_id%4] : 0;
                uint32_t x3 = (token_src[1] < M) ? 
                    reinterpret_cast<const uint32_t*>(x + token_src[1]*K + 64 + b_off + 16)[lane_id%4] : 0;
                
                uint32_t w0 = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + (lane_id%4)*4 + 64 + b_off);
                uint32_t w1 = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + (lane_id%4)*4 + 64 + b_off + 16);
                
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(x0), "r"(x1), "r"(x2), "r"(x3), "r"(w0), "r"(w1));
            }
            
            // k_step = 3
            {
                uint32_t x0 = (token_src[0] < M) ? 
                    reinterpret_cast<const uint32_t*>(x + token_src[0]*K + 96 + b_off)[lane_id%4] : 0;
                uint32_t x1 = (token_src[1] < M) ? 
                    reinterpret_cast<const uint32_t*>(x + token_src[1]*K + 96 + b_off)[lane_id%4] : 0;
                uint32_t x2 = (token_src[0] < M) ? 
                    reinterpret_cast<const uint32_t*>(x + token_src[0]*K + 96 + b_off + 16)[lane_id%4] : 0;
                uint32_t x3 = (token_src[1] < M) ? 
                    reinterpret_cast<const uint32_t*>(x + token_src[1]*K + 96 + b_off + 16)[lane_id%4] : 0;
                
                uint32_t w0 = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + (lane_id%4)*4 + 96 + b_off);
                uint32_t w1 = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + (lane_id%4)*4 + 96 + b_off + 16);
                
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(x0), "r"(x1), "r"(x2), "r"(x3), "r"(w0), "r"(w1));
            }
            
            // Apply scaling
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
        
        // Store results
        if (token_src[0] < M)
        {
            *reinterpret_cast<__nv_bfloat162*>(out + token_dest[0]*N + warpN * BN + (lane_id%4)*2) = 
                __nv_bfloat162(f_acc[0], f_acc[1]);
        }
        if (token_src[1] < M)
        {
            *reinterpret_cast<__nv_bfloat162*>(out + token_dest[1]*N + warpN * BN + (lane_id%4)*2) = 
                __nv_bfloat162(f_acc[2], f_acc[3]);
        }
    }
    
    
    // Alternative with vectorized loads directly in MMA
    template <int BM, int BK, int BN>
    __global__ void fused_moe_w8a8_direct_vec_load_kernel(
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
    
        float f_acc[4] = {0.f};
    
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
            
            // Using vectorized loads directly - compiler will optimize register usage
            for(int k_step = 0; k_step < 4; k_step++) {
                int k = k_step * BK;
                
                // Load and use immediately in MMA
                uint4 x_vec0, x_vec1;
                uint32_t x_regs[4], w_regs[2];
                
                if (token_src[0] < M && (lane_id % 4) == 0) {
                    // Try vectorized load for better coalescing
                    x_vec0 = *reinterpret_cast<const uint4*>(x + token_src[0]*K + k + b_off + (lane_id/4)*4);
                    x_regs[0] = x_vec0.x;  // Will be distributed via register allocation
                } else if (token_src[0] < M) {
                    x_regs[0] = reinterpret_cast<const uint32_t*>(x + token_src[0]*K + k + b_off)[lane_id%4];
                } else {
                    x_regs[0] = 0;
                }
                
                // Continue with other loads...
                x_regs[1] = (token_src[1] < M) ? 
                    reinterpret_cast<const uint32_t*>(x + token_src[1]*K + k + b_off)[lane_id%4] : 0;
                x_regs[2] = (token_src[0] < M) ? 
                    reinterpret_cast<const uint32_t*>(x + token_src[0]*K + k + b_off + 16)[lane_id%4] : 0;
                x_regs[3] = (token_src[1] < M) ? 
                    reinterpret_cast<const uint32_t*>(x + token_src[1]*K + k + b_off + 16)[lane_id%4] : 0;
                
                w_regs[0] = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + (lane_id%4)*4 + k + b_off);
                w_regs[1] = *reinterpret_cast<const uint32_t*>(exp_w + w_row*K + (lane_id%4)*4 + k + b_off + 16);
                
                asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
                        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                        : "r"(x_regs[0]), "r"(x_regs[1]), "r"(x_regs[2]), "r"(x_regs[3]), 
                          "r"(w_regs[0]), "r"(w_regs[1]));
            }
            
            // Apply scaling
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
        
        // Store results
        if (token_src[0] < M)
        {
            *reinterpret_cast<__nv_bfloat162*>(out + token_dest[0]*N + warpN * BN + (lane_id%4)*2) = 
                __nv_bfloat162(f_acc[0], f_acc[1]);
        }
        if (token_src[1] < M)
        {
            *reinterpret_cast<__nv_bfloat162*>(out + token_dest[1]*N + warpN * BN + (lane_id%4)*2) = 
                __nv_bfloat162(f_acc[2], f_acc[3]);
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
    constexpr int BM = 16;
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
