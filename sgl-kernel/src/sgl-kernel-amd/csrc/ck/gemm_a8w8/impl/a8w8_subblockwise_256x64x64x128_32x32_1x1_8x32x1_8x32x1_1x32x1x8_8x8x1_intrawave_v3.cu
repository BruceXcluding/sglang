// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "../gemm_a8w8_subblock_common.cuh"


torch::Tensor
a8w8_subblockwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3(
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
  bool k_pad = (K % 128 != 0);
  bool m_pad = (M % 128 != 0);
  bool n_pad = (N % 128 != 0);
  // TODO: add template arguments from best config list
  using DeviceGemmInstance = DeviceGemmHelper<
    256,
    64,
    64,
    128,
    32,
    32,
    1,
    1,
    S<8, 32, 1>,
    S<8, 32, 1>,
    S<1, 32, 1, 8>,
    S<8, 8, 1>, 
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v3,
    ck::tensor_operation::device::GemmSpecialization::MKPadding>;
  return gemm_a8w8_subblockwise_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
}

