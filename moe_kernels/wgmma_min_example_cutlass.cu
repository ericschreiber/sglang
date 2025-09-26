#include <cutlass/cutlass.h>
#include <cutlass/arch/wgmma.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/collective/builders/sm90.hpp>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/util/device_memory.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/reference/host/gemm.h>
#include <cutlass/util/tensor_view_io.h>

using ElementA = cutlass::float_e4m3_t;  // FP8 E4M3
using ElementB = cutlass::float_e4m3_t;  // FP8 E4M3
using ElementC = float;                  // Accumulator type

using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

constexpr int M = 64;
constexpr int N = 16;
constexpr int K = 32;

/// CUTLASS Gemm Kernel for WGMMA 64x16x32 FP8
using GemmKernel = typename cutlass::gemm::kernel::GemmUniversal<
    cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90,
        cutlass::arch::OpClassTensorOp,
        ElementA, LayoutA,  // A
        ElementB, LayoutB,  // B
        ElementC, LayoutC,  // C
        ElementC            // Accumulator
    >::CollectiveOp
>;

void run_cutlass_wgmma() {
    printf("CUTLASS WGMMA Minimal Example\n");

    // Host tensors
    cutlass::HostTensor<ElementA, LayoutA> A({M, K});
    cutlass::HostTensor<ElementB, LayoutB> B({N, K});
    cutlass::HostTensor<ElementC, LayoutC> C({M, N});

    // Initialize input
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k)
            A.at({m, k}) = ElementA(1.0f);

    for (int n = 0; n < N; ++n)
        for (int k = 0; k < K; ++k)
            B.at({n, k}) = (k < 4 && n < 2) ? ElementB(float(k)) : ElementB(0.0f);

    // Allocate device memory
    A.sync_device();
    B.sync_device();
    C.sync_device();

    // Gemm Arguments
    typename GemmKernel::Arguments args{
        {M, N, K},                    // Problem shape
        {A.device_data(), K},         // A ptr + lda
        {B.device_data(), K},         // B ptr + ldb
        {C.device_data(), N},         // C ptr + ldc
        {C.device_data(), N},         // D ptr (same as C)
        {ElementC(1.0f), ElementC(0)} // alpha, beta
    };

    // Instantiate kernel
    GemmKernel gemm_op;
    size_t workspace_size = gemm_op.get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

    cutlass::Status status = gemm_op(args, workspace.get());
    if (status != cutlass::Status::kSuccess) {
        printf("GemmKernel failed: %d\n", int(status));
        return;
    }

    C.sync_host();

    printf("Result C (64x16):\n");
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            printf("%8.4f ", C.at({i, j}));
        }
        printf("\n");
    }
}

int main() {
    run_cutlass_wgmma();
    return 0;
}
