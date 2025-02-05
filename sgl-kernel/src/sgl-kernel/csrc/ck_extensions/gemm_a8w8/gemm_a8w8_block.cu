// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_block_common.cuh"
#include "gemm_a8w8_block_manifest.h"
#include "gemm_a8w8_block_lookup.h"

BlockwiseKernel blockwise_heuristic_dispatch(int M, int N, int K) {
  // Apply shape heuristics to find a suitable kernel implementation.
  if (N < 2048){
    // Special case for small N
    return a8w8_blockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3;
  } else if (K < 1024) {
    // Special case for small K
    return a8w8_blockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v1; 
  } else {
    // Fallback Large kernel
    return a8w8_blockwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3;
  }
}

// Helper function to return the next largest power of 2
static constexpr int nextPow2(unsigned int num) {
  if (num <= 1) return 1;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

BlockwiseKernel blockwise_dispatch(int M, int N, int K) {
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
  static const KernelLookupMap lookup{};
  auto it = lookup.find(padded_m, N, K);
  // If we found an optimal kernel, use it.
  if (it != lookup.end()) {
    return it->second;
  }
  // Otherwise, use heuristics.
  return blockwise_heuristic_dispatch(M, N, K);
}

torch::Tensor gemm_a8w8_block(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y)
{
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");

    int M = XQ.size(0);
    int N = WQ.size(0);
    int K = XQ.size(1);
    if (Y.dtype() == at::ScalarType::Half) {
        if(XQ.dtype() == at::ScalarType::Float8_e4m3fnuz) {
            blockwise_dispatch(M, N, K)(XQ, WQ, x_scale, w_scale, Y);
        }else {
            TORCH_CHECK(false, "Weights and activations should be FP8e4m3fnuz!");
        }
    }
    else if (Y.dtype() == at::ScalarType::BFloat16) {
        if (XQ.dtype() == at::ScalarType::Float8_e4m3fnuz) {
                blockwise_dispatch(M, N, K)(XQ, WQ, x_scale, w_scale, Y);
            } else {
                TORCH_CHECK(false, "Weights and activations should be FP8e4m3fnuz!");
            }
    }
    return Y ;
}
