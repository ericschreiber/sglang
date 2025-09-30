#include <cstdint>
#include <cuda.h>
#include <cuda_fp8.h>
#include <stdio.h>
#include <cuda/ptx>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define ASSERT(cond, msg, args...) assert((cond) || !fprintf(stderr, (msg "\n"), args))

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}


// not gonna type all that
using fp8 = __nv_fp8_e4m3;

__device__ static inline uint64_t matrix_descriptor_encode(uint64_t x) {
    return (((x) & 0x3FFFF) >> 4);
}

// Descriptor for a shared memory matrix.
// Implementation is derived from PTX guide: https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-descriptor-format
__device__ uint64_t make_smem_desc(fp8* ptr, int sdo, int ldo=1) {
    // Convert shared memory pointer to integer
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = matrix_descriptor_encode(addr);
    desc |= matrix_descriptor_encode((uint64_t)ldo) << 16;
    desc |= ((uint64_t)sdo) << 32;
    desc |= (((uint64_t)addr >> 0x7) & 0x7) << 49;
    desc |= 1llu << 62; // 128B swizzle
    return desc;
}

__device__ void warpgroup_arrive() {
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_commit_batch() {
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

__device__ void warpgroup_wait() {
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}

template<int ScaleD, int ScaleA, int ScaleB, int TransA, int TransB>
__device__ void wgmma16(float d[8], fp8* sA, fp8* sB) {
    uint64_t desc_a = make_smem_desc(&sA[0], 64);
    uint64_t desc_b = make_smem_desc(&sB[0], 64);
    asm volatile(
        "{\n"
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3 "
        "{%0,   %1,   %2,   %3,   %4,   %5,   %6,   %7},   "
        " %8,"
        " %9,"
        " %10, %11, %12;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]),
            "+f"(d[6]), "+f"(d[7])
        : "l"(desc_a), "l"(desc_b), "n"(int32_t(ScaleD)), "n"(int32_t(ScaleA)),
            "n"(int32_t(ScaleB)));
}

#define CP_ASYNC_CG(dst, src, Bytes) \
    asm volatile("cp.async.cg.shared.global.L2::256B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_CG4(dst, src, Bytes) \
    asm volatile("cp.async.ca.shared.global.L2::64B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(Bytes))

#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)

#define CP_ASYNC_WAIT_GROUP(N) asm volatile("cp.async.wait_group %0;\n" ::"n"(N))

__device__ static __forceinline__ void init_barrier(uint64_t* bar, int thread_count, int transaction_count) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "mbarrier.init.shared::cta.b64 [%0], %1;\n"
        :: "r"(bar_ptr), "r"(thread_count+transaction_count)
    );
}

__device__ static __forceinline__ void cp_async_mbarrier_arrive(uint64_t* bar) {
    uint32_t bar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "cp.async.mbarrier.arrive.noinc.shared.b64 [%0];\n"
        :: "r"(bar_ptr)
    );
}

__device__ static __forceinline__ void arrive(uint64_t* bar, uint32_t count=1) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "mbarrier.arrive.shared.b64 _, [%0], %1;\n"
        :
        : "r"(mbar_ptr), "r"(count)
        : "memory"
    );
}

__device__ static __forceinline__ void wait(uint64_t* bar, int kPhaseBit) {
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(bar));
    asm volatile (
        "{\n"
        ".reg .pred                P1;\n"
        "WAIT:\n"
        "mbarrier.try_wait.parity.shared.b64 P1, [%0], %1;\n"
        "@!P1                       bra.uni WAIT;\n"
        "}\n"
        :: "r"(mbar_ptr),
        "r"(kPhaseBit)
    );
}

__device__ __forceinline__ void ld_matrix_x2(uint32_t* tile, uint32_t mat)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
            : "=r"(tile[0]), "=r"(tile[1]) : "r"(mat));
}

__device__ __forceinline__ void ld_matrix_x4(uint32_t* tile, uint32_t mat)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
            : "=r"(tile[0]), "=r"(tile[1]), "=r"(tile[2]), "=r"(tile[3]) : "r"(mat));
}

__device__ __forceinline__ void st_matrix_x4(uint32_t* tile, uint32_t mat)
{
    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};"
            :
            : "r"(mat), "r"(tile[0]), "r"(tile[1]), "r"(tile[2]), "r"(tile[3])
            : "memory"
            );
}

__device__ __forceinline__ void st_matrix_x4_trans(uint32_t* tile, uint32_t mat)
{
    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};"
            :
            : "r"(mat), "r"(tile[0]), "r"(tile[1]), "r"(tile[2]), "r"(tile[3])
            : "memory"
            );
}

template <uint32_t RegCount>
__device__ void reg_alloc() {
    asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}

template <uint32_t RegCount>
__device__ void reg_dealloc() {
    asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" : : "n"(RegCount));
}


#define S_BITS 3
#define S_MASK 0b1110000

template<int STAGES, int WS, int XS, int WM, int WN, int TN>
struct smem
{
    alignas(1024) fp8 w[STAGES*WS];
    alignas(1024) fp8 x[STAGES*XS];
};

template <int BM, int BK, int BN, int PF, int WM, int WN, int STAGES, int TN, int PRODUCER_THREADS>
__global__ __launch_bounds__(WN*32 + PRODUCER_THREADS) void fused_moe_w8a8_wgmma_kernel(
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
    constexpr int CONSUMER_THREADS = WN*32;
    const int32_t warpM = blockIdx.y;
    const int exp_idx = expert_ids[warpM];
    if(warpM * BM >= num_tokens_post_padded[0])
        return;

    const int32_t warpN = (blockIdx.x*CONSUMER_THREADS + (threadIdx.x - PRODUCER_THREADS))/32;

    //TODO should not be hardcoded
    constexpr int block_shape[2] = {128, 128};

    const fp8* exp_w = w + exp_idx * K * N;
    const int lane_id = threadIdx.x%32;
    const bool is_producer = threadIdx.x < PRODUCER_THREADS;
    const int warp_id = is_producer ? threadIdx.x/32 : (threadIdx.x-PRODUCER_THREADS)/32;
    const int w_row = warpN * BN + (lane_id>>2);


    //SMEM sizes
    constexpr int WS = WN*PF*BK*BN;
    constexpr int XS = PF*BK*BM;
    // how many bytes we transfer per CP_ASYNC
    constexpr int TB = 16;
    // Thread offset per transfer
    constexpr int TO = TB/sizeof(fp8);

    extern __shared__ __align__(1024) uint8_t sh[];
    smem<STAGES, WS, XS, WM, WN, TN>& s = *reinterpret_cast<smem<STAGES, WS, XS, WM, WN, TN>*>(sh);

    __shared__ __align__(8) uint64_t bar[2*STAGES];
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < STAGES; i++)
        {
            init_barrier(&bar[i], PRODUCER_THREADS, 0);
            init_barrier(&bar[i + STAGES], CONSUMER_THREADS, 0);
        }
    }
    __syncthreads();

    // N tiles in memory
    constexpr int STG = block_shape[0]/32;
    int n_stages = K/block_shape[0];

    float f_acc[TN][4] = {0.f};
    memset(f_acc, 0, sizeof(f_acc));

    int token_dest[2];
    int token_src[2];
    int p = 0;
    // PRODUCER
    if (is_producer)
    {
        // TODO does it really matter?
        // reg_dealloc<32>();
        int tsrc = sorted_token_ids[warpM*BM + threadIdx.x/8] / top_k;
        for (int load_stage = 0; load_stage<n_stages; load_stage++)
        {
            wait(bar + STAGES + (load_stage%STAGES), p);
            if ((load_stage%STAGES) == STAGES-1)
                p^=1;
            const int off = load_stage * block_shape[0];
            int smem_stage = load_stage%STAGES;
            for(int i = (threadIdx.x)*TO;
                    i < WS;
                    i += PRODUCER_THREADS*TO)
            {
                int row = blockIdx.x*WN*BN + i/(BK*PF);
                int col = off + i%(BK*PF);
                int swizzled = i^((i&(S_MASK<<S_BITS))>>S_BITS);
                uint32_t sm = __cvta_generic_to_shared(s.w + smem_stage*WS + swizzled);
                CP_ASYNC_CG(sm, reinterpret_cast<const float4*>(exp_w + row*K + col), TB);
            }

            {
                int i = threadIdx.x*TO;
                int row = tsrc;
                if(row < M)
                {
                    int swizzled = i^((i&(S_MASK<<S_BITS))>>S_BITS);
                    int col = off + i%(BK*PF);
                    uint32_t sm = __cvta_generic_to_shared(s.x + smem_stage*XS + swizzled);
                    CP_ASYNC_CG(sm, reinterpret_cast<const float4*>(x + row*K + col), TB);
                }
            }
            cp_async_mbarrier_arrive(bar + smem_stage);
        }
    }
    // CONSUMER
    else
    {
        uint32_t tile_x[STG][4];
        uint32_t tile_w[TN][STG/2][4];
        token_dest[0] = sorted_token_ids[warpM*BM + (lane_id>>2)];
        token_dest[1] = sorted_token_ids[warpM*BM + (lane_id>>2) + 8];

        token_src[0] = token_dest[0]/top_k;
        token_src[1] = token_dest[1]/top_k;
        // reg_alloc<128>();
        // Empty barriers arrive instantly
        for (int i = 0; i < STAGES; i++)
            arrive(&bar[STAGES + i]);


        for (int compute_stage = 0; compute_stage < n_stages; compute_stage += 1)
        {
            wait(bar + (compute_stage%STAGES), p);
            if ((compute_stage%STAGES) == STAGES-1)
                p^=1;

            const int scale_cols_x = K/block_shape[1];
            const int scale_rows_w = N/block_shape[1];
            const int scale_cols_w = K/block_shape[0];

            float scale_x[2];
            if (token_src[0] < M)
            {
                scale_x[0] = x_scale[(token_src[0])*scale_cols_x + compute_stage];
            }
            if (token_src[1] < M)
            {
                scale_x[1] = x_scale[(token_src[1])*scale_cols_x + compute_stage];
            }

            float scale_w = w_scale[exp_idx * scale_rows_w * scale_cols_w + (w_row/block_shape[1])*scale_cols_w + compute_stage];


            float tile_acc[TN*4] = {0.f};
            fp8* sw = s.w + (compute_stage%STAGES)*WS;
            fp8* sx = s.x + (compute_stage%STAGES)*XS;
            warpgroup_arrive();
            wgmma16<1,1,1,1,1>(tile_acc, sw, sx);
            warpgroup_commit_batch();
            warpgroup_wait();
            warpgroup_arrive();
            wgmma16<1,1,1,1,1>(tile_acc, sw+1*32, sx+1*32);
            warpgroup_commit_batch();
            warpgroup_wait();
            warpgroup_arrive();
            wgmma16<1,1,1,1,1>(tile_acc, sw+2*32, sx+2*32);
            warpgroup_commit_batch();
            warpgroup_wait();
            warpgroup_arrive();
            wgmma16<1,1,1,1,1>(tile_acc, sw+3*32, sx+3*32);
            warpgroup_commit_batch();
            warpgroup_wait();
            arrive(bar + STAGES + (compute_stage%STAGES));

            int token;
            float x_sc;
            token = (lane_id%4)*2;
            x_sc = __shfl_sync(0xFFFFFFFF, scale_x[0], token*4);
            f_acc[0][0] += scale_w * x_sc * tile_acc[0];
            f_acc[0][2] += scale_w * x_sc * tile_acc[2];

            token = (lane_id%4)*2 + 1;
            x_sc = __shfl_sync(0xFFFFFFFF, scale_x[0], token*4);
            f_acc[0][1] += scale_w * x_sc * tile_acc[1];
            f_acc[0][3] += scale_w * x_sc * tile_acc[3];

            token = (lane_id%4)*2 + 8;
            x_sc = __shfl_sync(0xFFFFFFFF, scale_x[1], token*4);
            f_acc[1][0] += scale_w * x_sc * tile_acc[4];
            f_acc[1][2] += scale_w * x_sc * tile_acc[6];

            token = (lane_id%4)*2 + 9;
            x_sc = __shfl_sync(0xFFFFFFFF, scale_x[1], token*4);
            f_acc[1][1] += scale_w * x_sc * tile_acc[5];
            f_acc[1][3] += scale_w * x_sc * tile_acc[7];

        }


    }
    __syncthreads();
    __nv_bfloat16* out_sm = reinterpret_cast<__nv_bfloat16*>(sh);

    if (!is_producer)
    {
        for(int i = 0; i<TN; i+=2)
        {
            uint32_t tile[4];
            __nv_bfloat162 temp0 = __nv_bfloat162(f_acc[i][0], f_acc[i][1]);
            __nv_bfloat162 temp2 = __nv_bfloat162(f_acc[i][2], f_acc[i][3]);
            __nv_bfloat162 temp1 = __nv_bfloat162(f_acc[i+1][0], f_acc[i+1][1]);
            __nv_bfloat162 temp3 = __nv_bfloat162(f_acc[i+1][2], f_acc[i+1][3]);
            memcpy(&tile[0], &temp0, sizeof(uint32_t));

            memcpy(&tile[1], &temp1, sizeof(uint32_t));

            memcpy(&tile[2], &temp2, sizeof(uint32_t));

            memcpy(&tile[3], &temp3, sizeof(uint32_t));

            int out_row = lane_id%16;
            int out_col = warp_id * BN + i*8 + (lane_id/16) * 8;
            uint32_t out_sm_u = __cvta_generic_to_shared(out_sm + out_row*WN*BN + out_col);
            st_matrix_x4_trans(tile, out_sm_u);
        }
    }
    cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
    __syncthreads();
    if(!is_producer && warp_id == 0 && lane_id%4 == 0)
    {
        if (token_src[0] < M)
        {
            cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_global,
                    cuda::ptx::space_shared,
                    out + token_dest[0]*N + warpN*BN,
                    out_sm + (lane_id>>2)*BN*WN,
                    BN*WN*sizeof(__nv_bfloat16));
        }
        if (token_src[1] < M)
        {
            cuda::ptx::cp_async_bulk(
                    cuda::ptx::space_global,
                    cuda::ptx::space_shared,
                    out + token_dest[1]*N + warpN*BN,
                    out_sm + (8+(lane_id>>2))*BN*WN,
                    BN*WN*sizeof(__nv_bfloat16));
        }
        cuda::ptx::cp_async_bulk_commit_group();
        cuda::ptx::cp_async_bulk_wait_group_read(cuda::ptx::n32_t<0>());
    }
}

void fused_moe_w8a8_wgmma(
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
    constexpr int BN = 16;
    constexpr int PF = 4;
    constexpr int WN = 4;
    // TODO this will only work for num_warps_y = 1
    constexpr int WM = 1;
    constexpr int STAGES = 2;
    constexpr int TN = BN/8;
    constexpr int PRODUCER_THREADS = 128;
    dim3 dimBlock(32*WN + PRODUCER_THREADS, 1, 1);
    dim3 dimGrid(std::ceil((float)N/(BN*WN)), std::ceil((float)sorted_num/(BM*WM)), 1);

    size_t sMemSize = sizeof(smem<STAGES, WN*PF*BK*BN, PF*BK*BN, WM, WN, TN>);
    gpuErrchk(cudaFuncSetAttribute(
        fused_moe_w8a8_wgmma_kernel<BM, BK, BN, PF, WM, WN, STAGES, TN, PRODUCER_THREADS>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, sMemSize));

    fused_moe_w8a8_wgmma_kernel<BM, BK, BN, PF, WM, WN, STAGES, TN, PRODUCER_THREADS><<<dimGrid, dimBlock, sMemSize>>>(
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