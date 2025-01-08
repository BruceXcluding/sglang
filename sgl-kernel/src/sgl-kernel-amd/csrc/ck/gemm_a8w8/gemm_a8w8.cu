// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_common.cuh"
#include "gemm_a8w8_manifest.h"
#include "gemm_a8w8_lookup.h"

template <typename DEDataType, typename ABDataType>
RowwiseKernel rowwise_heuristic_dispatch(int M, int N, int K) {
  // Apply shape heuristics to find a suitable kernel implementation.
  if (M < 64 && N < 2048 && K < 2048) {
    // Kernel that generally works well on small shapes.
    return a8w8_rowwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2<DEDataType, ABDataType>;
  } else if (M < 64 && K < 2048) {
    // Kernel that works well for small batch size and small K.
    return a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2<DEDataType, ABDataType>;
  } else if (M < 64 && N < 2048) {
    // Kernel that works well for small batch size and small N.
    return a8w8_rowwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_1x1_interwave_v2<DEDataType, ABDataType>;
  } else if (M < 64 && N > 2048 && K > 2048) {
    // Kernel that works well for small M but larger N and K.
    return a8w8_rowwise_64x16x16x256_16x16_1x1_16x4x1_16x4x1_1x4x1x16_4x4x1_1x1_intrawave_v1<DEDataType, ABDataType>;
  } else if (M < 64) {
    // Fallback to generic small batch kernel if we cant find a good match.
    return a8w8_rowwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_1x1_interwave_v2<DEDataType, ABDataType>;
  /* } else if (((M < 512 && K < 8192) || (N <= 2048 && K <= 8192) || (K <= 2048 && N <= 8192)) && K >= 1024) {
    // Kernel that is optimized for larger batch sizes but otherwise small
    // tensors.
    return a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v5<DEDataType, ABDataType>; */
  } else if (K < 1024) {
    // Special case for small K.
    return a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_interwave_v1<DEDataType, ABDataType>;
  } else if (M < 1024) {
    // Kernel for generic medium batch sizes.
    return a8w8_rowwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_1x1_intrawave_v3<DEDataType, ABDataType>;
  } else if (M >= 1024 && N >= 1024 && K >= 1024) {
    // Kernel for very large gemm
    //return a8w8_rowwise_256x256x256x128_16x16_8x8_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3<DEDataType, ABDataType>;
    return a8w8_rowwise_256x256x128x64_32x32_4x2_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_interwave_v1<DEDataType, ABDataType>;
  } else {
    // Fallback large kernel.
    return a8w8_rowwise_256x256x256x64_32x32_4x4_4x64x1_4x64x1_1x32x1x8_8x8x1_1x1_intrawave_v4<DEDataType, ABDataType>;
  }
}

// Helper function to return the next largest power of 2
static constexpr int nextPow2(unsigned int num) {
  if (num <= 1) return 1;
  return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

template <typename DEDataType, typename ABDataType>
RowwiseKernel rowwise_dispatch(int M, int N, int K) {
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
  return rowwise_heuristic_dispatch<DEDataType, ABDataType>(M, N, K);
}


torch::Tensor gemm_a8w8(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y)
{
    TORCH_CHECK(XQ.dtype() == WQ.dtype(), "Weights and activations should have the same dtype!");
    TORCH_CHECK(x_scale.dtype() == Y.dtype() && w_scale.dtype() == Y.dtype(),
                "Scales and output should have the same dtype!");

    int M = XQ.size(0);
    int N = WQ.size(0);
    int K = XQ.size(1);

    if (Y.dtype() == at::ScalarType::Half) {
      if (XQ.dtype() == at::ScalarType::Char) {
        rowwise_dispatch<F16, I8>(M, N, K)(XQ, WQ, x_scale, w_scale, Y);
      } else if (XQ.dtype() == at::ScalarType::Float8_e4m3fnuz) {
        rowwise_dispatch<F16, F8>(M, N, K)(XQ, WQ, x_scale, w_scale, Y);
      } else {
        TORCH_CHECK(false, "Weights and activations should both be INT8 or FP8e4m3fnuz!");
      }
    } else if (Y.dtype() == at::ScalarType::BFloat16) {
      if (XQ.dtype() == at::ScalarType::Char) {
        rowwise_dispatch<B16, I8>(M, N, K)(XQ, WQ, x_scale, w_scale, Y);
      } else if (XQ.dtype() == at::ScalarType::Float8_e4m3fnuz) {
        rowwise_dispatch<B16, F8>(M, N, K)(XQ, WQ, x_scale, w_scale, Y);
      } else {
        TORCH_CHECK(false, "Weights and activations should both be INT8 or FP8e4m3fnuz!");
      }
    } else {
      TORCH_CHECK(false, "Scales and output dtype should be FP16 or BF16!");
    }
    return Y;
}