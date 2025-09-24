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

// __device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) { return (((x) & 0x3FFFF) >> 0x4); }

// __device__ uint64_t make_smem_desc(fp8* ptr) {
//     // https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-warpgroup-level-matrix-shared-memory-layout-matrix-descriptor
//     uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
//     uint64_t desc = 0x0000000000000000;
//     desc |= matrix_descriptor_encode(addr);
//     desc |= matrix_descriptor_encode((uint64_t)16) << 16; // Either 16 or 32
//     desc |= matrix_descriptor_encode((uint64_t)256) << 32; // 8*32; 8 rows times 32 cols times 1 byte
//     // desc |= 1llu << 62; // 128B swizzle // No swizzle
//     return desc;
//   }

__device__ static inline uint64_t matrix_descriptor_encode_uint32(uint32_t x) {
    // matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4
    // result fits in 14 bits.
    return (uint64_t)((x & 0x3FFFFu) >> 4);
}

__device__ uint64_t make_smem_desc(const void* ptr, uint32_t leading_dim_bytes, uint32_t stride_dim_bytes) {
    // ptr must be a shared-memory address (use __cvta_generic_to_shared before calling)
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    // Must be 16-byte aligned (doc requirement)
    // Build descriptor according to Table 40 in PTX docs:
    // bits  0..13   : matrix-descriptor-encode(Matrix start address)
    // bits 16..29   : matrix-descriptor-encode(Leading dimension byte offset relative) OR absolute addr encoded
    // bits 32..45   : matrix-descriptor-encode(Stride dimension byte offset)
    // bits 46..48   : fixed constant 0b001
    // bits 49..51   : matrix base offset (use 0)
    // bit  52      : leading-dim stride mode (0 = relative offset)
    // bits 53..60  : fixed constant (as in doc)
    // bits 61..63  : swizzle (0 = no swizzle)

    uint64_t desc = 0;

    // encoded start address in bits 0..13
    desc |= (matrix_descriptor_encode_uint32(addr) & 0x3FFFULL) << 0; // bits 0..13

    // encoded leading-dim in bits 16..29
    desc |= (matrix_descriptor_encode_uint32(leading_dim_bytes) & 0x3FFFULL) << 16;

    // encoded stride-dim in bits 32..45
    desc |= (matrix_descriptor_encode_uint32(stride_dim_bytes) & 0x3FFFULL) << 32;

    // fixed constant 0b001 in bits 46..48 (doc)
    desc |= (uint64_t)(0x1) << 46;

    // matrix base offset (bits 49..51) -> keep 0 for canonical start
    // leading dimension stride mode bit (bit 52): 0 => relative byte offset (we use relative)
    // fixed constant value occupying bits 53..60: doc shows a fixed constant here, set to the documented constant.
    // The PTX doc shows this as "Fixed constant value of 0xb00000000" in the table; implementations typically set the 8-bit constant field to 0xB0 (shifted to bits 53..60).
    // Set bits 53..60 to 0xB0 (this matches other implementations / examples).
    desc |= (uint64_t)(0xB0ULL) << 53;

    // swizzle bits 61..63 = 0 (no swizzle)
    // done (zero by default)

    return desc;
}

template<int ScaleD, int ScaleA, int ScaleB>
__device__ void wgmmaM64N16K32(float d[2][2][2], fp8* sA, fp8* sB) {
    // d[row][col][0, 1]
    // uint64_t desc_a = make_smem_desc(&sA[0]);
    // uint64_t desc_b = make_smem_desc(&sB[0]);
    // assert(sA & 0xF == 0);
    // assert(sB & 0xF == 0);
    // uint64_t desc_a = make_smem_desc(sA, 32, 256);  // A: stride_dim_bytes = 8 * leading_dim_bytes = 8 * 32 = 256 bytes.
    // uint64_t desc_b = make_smem_desc(sB, 32, 256);   // B: 16×32 (stored as [N][K])
    uint64_t desc_a = make_smem_desc(sA, 0, 0);
    uint64_t desc_b = make_smem_desc(sB, 0, 0);
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
            // Load x to shared memory
            for (int i = 0; i < BM/4; i++){
                // sorted_token_ids[warpM*BM + (lane_id>>2)] / top_k;
                int token_src = sorted_token_ids[warpM*BM + i] / top_k;
                if (token_src < M)
                {
                    s_x[warp_idx*16+i][lane_id] = reinterpret_cast<const fp8*>(x + token_src*K + k + b_off)[lane_id];
                }
                else{
                    s_x[warp_idx*16+i][lane_id] = fp8(0);
                }
            }
            // Load w to shared memory
            if (warp_idx == 0){
                for (int i = 0; i < BN; i++){
                    int wT_row = warpN * BN + i;
                    int wT_col = k + b_off;
                    s_wT[i][lane_id] = reinterpret_cast<const fp8*>(exp_w + wT_row*K + wT_col)[lane_id];
                }
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
