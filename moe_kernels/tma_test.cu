#include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap
#include <cuda.h>
#include <iostream>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda_fp8.h>


using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

using fp8 = __nv_fp8_e4m3;

// CUDA Check
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(err); \
    } \
} while(0)
#define CUDA_CHECK_SUCCESS(err) do { cudaError_t err = cudaGetLastError(); if (err != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); exit(err); } } while (0)

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

PFN_cuTensorMapEncodeTiled_v12000 get_cuTensorMapEncodeTiled() {
    // Get pointer to cuTensorMapEncodeTiled
    cudaDriverEntryPointQueryResult driver_status;
    void* cuTensorMapEncodeTiled_ptr = nullptr;
    CUDA_CHECK(cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &cuTensorMapEncodeTiled_ptr, 12000, cudaEnableDefault, &driver_status));
    assert(driver_status == cudaDriverEntryPointSuccess);
  
    return reinterpret_cast<PFN_cuTensorMapEncodeTiled_v12000>(cuTensorMapEncodeTiled_ptr);
  }

void print_matrix(fp8* matrix, int M, int N) {
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < N; j++) {
        std::cout << int(uint8_t(matrix[i * N + j])) << " ";
      }
      std::cout << std::endl;
    }
  }

static constexpr size_t buf_len = 16;
__global__ void add_one_continous_kernel(int* data, size_t offset)
{
  // Shared memory buffer. The destination shared memory buffer of
  // a bulk operations should be 16 byte aligned.
  __shared__ alignas(16) int smem_data[buf_len];

  // 1. a) Initialize shared memory barrier with the number of threads participating in the barrier.
  //    b) Make initialized barrier visible in async proxy.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) { 
    init(&bar, blockDim.x);                      // a)
    ptx::fence_proxy_async(ptx::space_shared);   // b)
  }
  __syncthreads();

  // 2. Initiate TMA transfer to copy global to shared memory.
  if (threadIdx.x == 0) {
    // 3a. cuda::memcpy_async arrives on the barrier and communicates
    //     how many bytes are expected to come in (the transaction count)
    cuda::memcpy_async(
        smem_data, 
        data + offset, 
        cuda::aligned_size_t<16>(sizeof(smem_data)),
        bar
    );
  }
  // 3b. All threads arrive on the barrier
  barrier::arrival_token token = bar.arrive();
  
  // 3c. Wait for the data to have arrived.
  bar.wait(std::move(token));

  // 4. Compute saxpy and write back to shared memory
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
    smem_data[i] += 1;
  }

  // 5. Wait for shared memory writes to be visible to TMA engine.
  ptx::fence_proxy_async(ptx::space_shared);   // b)
  __syncthreads();
  // After syncthreads, writes by all threads are visible to TMA engine.

  // 6. Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    ptx::cp_async_bulk(
        ptx::space_global,
        ptx::space_shared,
        data + offset, smem_data, sizeof(smem_data));
    // 7. Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    ptx::cp_async_bulk_commit_group();
    // Wait for the group to have completed reading from shared memory.
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }
}

void add_one_continous(int size, int offset) {
    // create matrix
    int h_matrix[2048];
  for (int i = 0; i < 2048; i++) {
    h_matrix[i] = i;
  }

  int* d_matrix;
  CUDA_CHECK(cudaMalloc(&d_matrix, size * sizeof(int)));
  CUDA_CHECK(cudaMemcpy(d_matrix, h_matrix, size * sizeof(int), cudaMemcpyHostToDevice));

  // Kernel launch
  dim3 dimBlock(32, 1, 1);
  dim3 dimGrid(1, 1);
  add_one_continous_kernel<<<dimGrid, dimBlock>>>(d_matrix, offset);
  CUDA_CHECK(cudaMemcpy(h_matrix, d_matrix, size * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_matrix));

  for (int i = 0; i < 2048; i++) {
    std::cout << h_matrix[i] << " ";
  }
  std::cout << std::endl;
}


// __global__ void add_one_matrix_tile_kernel(int* matrix, int M, int N, CUtensorMap tensor_map, int row_offset, int col_offset) {
//   printf("kernel started\n");
//   __shared__ __align__(128) int smem_data[tile_size_h][tile_size_w];

//     // Initialize shared memory barrier with the number of threads participating in the barrier.
//   #pragma nv_diag_suppress static_var_with_dynamic_init
//   __shared__ barrier bar;

//   if (threadIdx.x == 0) {
//     // Initialize barrier. All `blockDim.x` threads in block participate.
//     init(&bar, blockDim.x);
//     // Make initialized barrier visible in async proxy.
//     cde::fence_proxy_async_shared_cta();
//   }
//   // Syncthreads so initialized barrier is visible to all threads.
//   __syncthreads();

//   barrier::arrival_token token;
//   if (threadIdx.x == 0) {
//     // Initiate bulk tensor copy.
//     cde::cp_async_bulk_tensor_2d_global_to_shared(&smem_data, &tensor_map, row_offset, col_offset, bar);
//     // Arrive on the barrier and tell how many bytes are expected to come in.
//     token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_data));
//   } else {
//     // Other threads just arrive.
//     token = bar.arrive();
//   }
//   // Wait for the data to have arrived.
//   bar.wait(std::move(token));

// //   // Symbolically modify a value in shared memory.
// //   for (int i = 0; i < tile_size_h; i++) {
// //     printf("i = %d\n", i);
// //     for (int j = 0; j < tile_size_w; j += blockDim.x) {
// //       printf("j = %d\n", j);
// //       smem_data[i][j + threadIdx.x] *= 100;
// //       printf("smem_data[%d][%d] = %d\n", i, j + threadIdx.x, smem_data[i][j + threadIdx.x]);
// //     }
// //   }

// //   // Wait for shared memory writes to be visible to TMA engine.
// //   cde::fence_proxy_async_shared_cta();
// //   __syncthreads();
// //   // After syncthreads, writes by all threads are visible to TMA engine.

// //   // Initiate TMA transfer to copy shared memory to global memory
// //   if (threadIdx.x == 0) {
// //     cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, row_offset, col_offset, &smem_data);
// //     // Wait for TMA transfer to have finished reading shared memory.
// //     // Create a "bulk async-group" out of the previous bulk copy operation.
// //     cde::cp_async_bulk_commit_group();
// //     // Wait for the group to have completed reading from shared memory.
// //     cde::cp_async_bulk_wait_group_read<0>();
// //   }

// //   // Destroy barrier. This invalidates the memory region of the barrier. If
// //   // further computations were to take place in the kernel, this allows the
// //   // memory location of the shared memory barrier to be reused.
// //   if (threadIdx.x == 0) {
// //     (&bar)->~barrier();
// //   }
// }
template <int BlockMajorSize, int BlockMinorSize>
CUtensorMap create_tensor_map(fp8* gmem_ptr, int gmem_width, int gmem_height) {
    CUtensorMap tma_map_host;
    void* gmem_address = (void*)gmem_ptr;
    uint64_t gmem_prob_shape[2] = {(uint64_t)gmem_width, (uint64_t)gmem_height};
    uint64_t gmem_prob_stride[1] = {sizeof(fp8) * gmem_width};
    uint32_t smem_box_shape[2] = {uint32_t(BlockMinorSize), uint32_t(BlockMajorSize)};
    uint32_t smem_box_stride[2] = {1, 1};

    CUresult result = cuTensorMapEncodeTiled(
        &tma_map_host, 
        CU_TENSOR_MAP_DATA_TYPE_UINT8, 
        2,                                  // cuuint32_t tensorRank
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

template <int TileSizeH, int TileSizeW>
__global__ void add_one_matrix_tile_kernel(int M, int K, const __grid_constant__ CUtensorMap tensor_map, int row_offset, int col_offset) {
    if (threadIdx.x == 0) { 
        printf("kernel started with M = %d, K = %d, row_offset = %d, col_offset = %d, TileSizeH = %d, TileSizeW = %d\n", M, K, row_offset, col_offset, TileSizeH, TileSizeW);
    }

    __shared__ __align__(128) fp8 smatrix[TileSizeH][TileSizeW];
    #pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bar;

    if (threadIdx.x == 0) {
        init(&bar, blockDim.x);
        cde::fence_proxy_async_shared_cta();
    }
    __syncthreads();
    
    barrier::arrival_token token;

    if (threadIdx.x == 0) {
        cde::cp_async_bulk_tensor_2d_global_to_shared(&smatrix[0], &tensor_map, col_offset, row_offset, bar);
        token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smatrix));
    } else {
        token = bar.arrive();
    }
    bar.wait(std::move(token));
    __syncthreads();


    // Symbolically modify a value in shared memory.
    for (int i = 0; i < TileSizeH; i++) {
        for (int j = 0; j < TileSizeW; j += blockDim.x) {
        smatrix[i][j + threadIdx.x] = fp8(8) ;
        }
    }

    // Wait for shared memory writes to be visible to TMA engine.
    cde::fence_proxy_async_shared_cta();
    __syncthreads();
    // After syncthreads, writes by all threads are visible to TMA engine.

    // Initiate TMA transfer to copy shared memory to global memory
    if (threadIdx.x == 0) {
        cde::cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, row_offset, col_offset, &smatrix);
        // Wait for TMA transfer to have finished reading shared memory.
        // Create a "bulk async-group" out of the previous bulk copy operation.
        cde::cp_async_bulk_commit_group();
        // Wait for the group to have completed reading from shared memory.
        cde::cp_async_bulk_wait_group_read<0>();
    }

    // Destroy barrier. This invalidates the memory region of the barrier. If
    // further computations were to take place in the kernel, this allows the
    // memory location of the shared memory barrier to be reused.
    if (threadIdx.x == 0) {
        (&bar)->~barrier();
    }
}

void add_one_matrix_tile(int M, int K, int row_offset, int col_offset) {

    static constexpr size_t tile_size_h = 32;
    static constexpr size_t tile_size_w = 32;

    fp8 h_matrix[M][K];
    for (int i = 0; i < M; i++) {
      for (int j = 0; j < K; j++) {
        h_matrix[i][j] = fp8(4); //fp8(i * K + j);
      }
    }
//   print_matrix(&h_matrix[0][0], M, K);

  fp8* d_matrix;
  CUDA_CHECK(cudaMalloc(&d_matrix, M * K * sizeof(fp8)));
  CUDA_CHECK(cudaMemcpy(d_matrix, h_matrix, M * K * sizeof(fp8), cudaMemcpyHostToDevice));

  // TMA setup
  auto tensor_map_h = create_tensor_map<tile_size_h, tile_size_w>(d_matrix, M, K);

  // Kernel launch
  dim3 dimBlock(32, 1, 1);
  dim3 dimGrid(1, 1);
  printf("calling kernel\n");
  add_one_matrix_tile_kernel<tile_size_h, tile_size_w><<<dimGrid, dimBlock>>>(M, K, tensor_map_h, row_offset, col_offset);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("kernel finished\n");

  CUDA_CHECK(cudaMemcpy(h_matrix, d_matrix, M * K * sizeof(fp8), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaFree(d_matrix));

  print_matrix(&h_matrix[0][0], M, K);
}

int main() {
    int device;
cudaGetDevice(&device);
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device);
printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    // add_one_continous(2048, 16);
  // add_one_matrix_tile(64, 64, 4, 40); // INT32: Min offset is 4 because we need to stay 16B aligned. We can continue over the matrix without issues. Padding is 0
  add_one_matrix_tile(64, 64, 16, 48); // FP8: Min offset is 8 for reading and 16 for writing.

  return 0;
}