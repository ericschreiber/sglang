#include <cuda.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <cassert>
#include <cuda/barrier>
#include <cuda/ptx>

#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// Not gonna type all that
using fp8 = __nv_fp8_e4m3;

template <int BlockMajorSize, int BlockMinorSize, int BlockDepthSize>
CUtensorMap create_3d_tensor_map(const fp8* gmem_ptr, int gmem_height, int gmem_width, int gmem_depth) {
    CUtensorMap tma_map_host;
    void* gmem_address = (void*)gmem_ptr;
    uint64_t gmem_prob_shape[3] = {(uint64_t)gmem_depth, (uint64_t)gmem_width, (uint64_t)gmem_height};
    // globalStrides[0] = globalDim[0] * elementSizeInBytes(tensorDataType) + padding[0];
      //     for (i = 1; i < tensorRank - 1; i++)
      //         globalStrides[i] = globalStrides[i – 1] * (globalDim[i] + padding[i]);
      //         assert(globalStrides[i] >= globalDim[i]);
    uint64_t gmem_prob_stride[2] = {
      (uint64_t) gmem_depth,                        
      (uint64_t) gmem_width * gmem_depth           
  };
    uint32_t smem_box_shape[3] = {uint32_t(BlockDepthSize), uint32_t(BlockMinorSize), uint32_t(BlockMajorSize)};
    uint32_t smem_box_stride[3] = {1, 1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tma_map_host, 
        CU_TENSOR_MAP_DATA_TYPE_UINT8, 
        3,                                  // cuuint32_t tensorRank
        gmem_address,                       // void *globalAddress, 
        gmem_prob_shape,                    // const cuuint64_t *globalDim,
        gmem_prob_stride,                   // const cuuint64_t *globalStrides,
        smem_box_shape,                     // const cuuint32_t *boxDim,
        smem_box_stride,                    // const cuuint32_t *elementStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE,
        CU_TENSOR_MAP_SWIZZLE_NONE, 
        CU_TENSOR_MAP_L2_PROMOTION_NONE, 
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
    );

    assert(result == CUDA_SUCCESS);
    return tma_map_host;
}

template <int BM, int BK, int BN, int WGMMA_BK>
__global__ void fused_moe_w8a8_wgmma_tma_naive_kernel(
        const fp8* __restrict__ x,
        // const __grid_constant__ CUtensorMap tensor_map_x,
        const float* __restrict__ x_scale,
        // const fp8* __restrict__ w,
        const __grid_constant__ CUtensorMap tensor_map_w,
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
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        printf("Kernel launched\n");
    }
    const int32_t warpN = (blockIdx.x*blockDim.x+threadIdx.x)/32;
    const int32_t warpM = blockIdx.y*blockDim.y+threadIdx.y;

    //TODO should not be hardcoded
    constexpr int block_shape[2] = {128, 128};

    const int exp_idx = expert_ids[warpM];
    // const fp8* exp_w = w + exp_idx * K * N;
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

    __shared__ alignas(128) fp8 tile_x[BM][BK];        // row-major, so same as in global memory
    __shared__ alignas(128) fp8 tile_wT[1][BN][BK];        // col-major, so same as in global memory

    float f_acc[4] = {0.f};
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
        #pragma nv_diag_suppress static_var_with_dynamic_init
        __shared__ barrier barX;
        __shared__ barrier barW;
        if (threadIdx.x == 0)
        {
            init(&barX, blockDim.x);
            init(&barW, blockDim.x);
            cde::fence_proxy_async_shared_cta();
        }
        __syncthreads();

        barrier::arrival_token tokenX, tokenW;

        // Load tile of w using 1 TMA transfer
        int b_off = block * block_shape[0]; // this is the same as w_col for 0th item
        if (threadIdx.x == 0) {
            if (exp_idx > 90) {
                printf("Loading with exp_idx %d, w_row %d, b_off %d\n", exp_idx, w_row, b_off);
            }
            cde::cp_async_bulk_tensor_3d_global_to_shared(&tile_wT[0], &tensor_map_w, w_row, exp_idx, b_off, barW); // This works but is wrong! TODO
            // tokenW = cuda::device::barrier_arrive_tx(barW, 1, sizeof(tile_wT));
        } else {
            // tokenW = barW.arrive();
        }
        // barW.wait(std::move(tokenW));
        __syncthreads();


        // // STEP 1 load x with known implementation and use mma
        // // STEP 2 load x with known implementation and use wgmma
        // // STEP 3 load x through SMem

        // float acc[4] = {0.f};

        // uint32_t tile_x[2][2][4];
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

        // uint32_t loaded_w[2];
        // int iter = 0;
        // for(int k = 0; k < block_shape[0]; k += BK){
        //     int w_col = (lane_id%4)*4 + k;
        //     loaded_w[0] = *reinterpret_cast<const uint32_t*>(tile_wT + w_row*K + w_col);
        //     loaded_w[1] = *reinterpret_cast<const uint32_t*>(tile_wT + w_row*K + w_col + 64);

        //     asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
        //         : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        //         : "r"(tile_x[0][0][iter]), "r"(tile_x[1][0][iter]), "r"(tile_x[0][1][iter]), "r"(tile_x[1][1][iter]), "r"(loaded_w[0]), "r"(loaded_w[1]));
        //     iter++;
        // }

        
        // if (token_src[0] < M)
        // {
        //     f_acc[0] += scale_x[0] * scale_w * acc[0];
        //     f_acc[1] += scale_x[0] * scale_w * acc[1];
        // }
        // if (token_src[1] < M)
        // {
        //     f_acc[2] += scale_x[1] * scale_w * acc[2];
        //     f_acc[3] += scale_x[1] * scale_w * acc[3];
        // }
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

void fused_moe_w8a8_wgmma_tma_naive(
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
        int num_experts,
        int sorted_num
        )
{
    // Step 2
    // constexpr int BM = 64;
    // Step 1
    constexpr int BM = 16;
    constexpr int BK = 128;
    constexpr int BN = 16; // we need at least 16 to use TMA
    constexpr int WGMMA_BK = 32;
    constexpr int num_warps_x = 4;
    constexpr int num_warps_y = 2;

    dim3 dimBlock(32*num_warps_x, num_warps_y, 1);
    dim3 dimGrid(std::ceil((float)N/(BN*num_warps_x)), std::ceil((float)sorted_num/(BM*num_warps_y)), 1);

    auto tensor_map_w = create_3d_tensor_map<1, BN, BK>(w, num_experts, N, K);
    printf("tensor_map_w created\n");
    fused_moe_w8a8_wgmma_tma_naive_kernel<BM, BK, BN, WGMMA_BK><<<dimGrid, dimBlock>>>(
            x,
            x_scale,
            tensor_map_w,
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
        return;
    }

    // Check for kernel execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
        return;
    }
}
