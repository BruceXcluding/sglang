// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "../gemm_a8w8_block_common.cuh"


torch::Tensor
a8w8_blockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y)
{
    // Check if this input needs to be padded.
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = WQ.size(0);
  int K = WQ.size(1);

  // add template arguments from best config list
  using DeviceGemmInstance = DeviceGemmHelper<
    256,
    64,
    128,
    128,
    32,
    32,
    1,
    2,
    S<8, 32, 1>,
    S<8, 32, 1>,
    S<1, 32, 1, 8>,
    S<8, 8, 1>,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v3,
    ck::tensor_operation::device::GemmSpecialization::MNKPadding>;
  return gemm_a8w8_blockwise_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
}
