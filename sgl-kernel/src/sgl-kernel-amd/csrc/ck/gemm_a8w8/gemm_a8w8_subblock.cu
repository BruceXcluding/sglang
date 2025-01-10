#include "gemm_a8w8_subblock_common.cuh"
#include "gemm_a8w8_subblock_manifest.h"
#include "gemm_a8w8_subblock_lookup.h"

template <typename DEDataType, typename ABDataType>
SubblockwiseKernel sublbockwise_heuristic_dispatch(int M, int N, int K) {
  // Apply shape heuristics to find a suitable kernel implementation.
  /* TODO: add support for DeepSeek-v3 Tuning Config  */
  // return a8w8_subblockwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v3<DEDataType, ABDataType>;
  return nullptr;
}

// Helper function to return the next largest power of 2
static constexpr int nextPow2(unsigned int num) {
  if (num <= 1) return 1;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <typename DEDataType, typename ABDataType>
SubblockwiseKernel subblockwise_dispatch(int M, int N, int K) {
  // For a given shape, either find the best kernel via lookup or heuristic.
  // For many small M shapes, we bucket them to the next largest kernel.
  // This is fine since kernels are padded anyway.
  int padded_m = M;
  if (M >= 1 && M <= 16) {
    padded_m = 16;
  } else if (M <= 16384) {
    padded_m = nextPow2(M);
  } else if (M <= 20480) {
    padded_m = 20480;
  }
  // For certain high priority shapes, we directly use the best kernel rather
  // than use heuristics.
  static const KernelLookupMap<DEDataType, ABDataType> lookup{};
  auto it = lookup.find(padded_m, N, K);
  // If we found an optimal kernel, use it.
  if (it != lookup.end()) {
    return it->second;
  }
  // Otherwise, use heuristics.
  return sublbockwise_heuristic_dispatch<DEDataType, ABDataType>(M, N, K);
}

torch::Tensor gemm_a8w8_subblock(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y,
    int scale_block_n,
    int scale_block_k)
{
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    TORCH_CHECK(x_scale.dtype() == Y.dtype() && w_scale.dtype() == Y.dtype(),
                "Scales and output should have the same dtype!");

    // int M = XQ.size(0);
    // int N = WQ.size(0);
    // int K = XQ.size(1);
    // if (Y.dtype() == at::ScalarType::Half) {
    //     if(XQ.dtype() == at::ScalarType::Float8_e4m3fnuz) {
    //         subblockwise_dispatch<F16, F8>(M, N, K)(XQ, WQ, x_scale, w_scale, Y);
    //     }else {
    //         TORCH_CHECK(false, "Weights and activations should both be FP8e4m3fnuz!");
    //     }
    // }
    // else if (Y.dtype() == at::ScalarType::BFloat16) {
    //     if (XQ.dtype() == at::ScalarType::Float8_e4m3fnuz) {
    //             subblockwise_dispatch<B16, F8>(M, N, K)(XQ, WQ, x_scale, w_scale, Y);
    //         } else {
    //             TORCH_CHECK(false, "Weights and activations should both be INT8 or FP8e4m3fnuz!");
    //         }
    // }

    // TODO: currently only support 1 instance
    gemm_a8w8_subblockwise_impl(XQ, WQ, x_scale, w_scale, Y, scale_block_n, scale_block_k);
    return Y ;
}
