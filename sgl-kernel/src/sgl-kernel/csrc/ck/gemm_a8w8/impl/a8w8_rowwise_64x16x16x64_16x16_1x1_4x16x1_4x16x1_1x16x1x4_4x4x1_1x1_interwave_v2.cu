// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "../gemm_a8w8_common.cuh"

template <typename DEDataType, typename ABDataType>
torch::Tensor
a8w8_rowwise_64x16x16x64_16x16_1x1_4x16x1_4x16x1_1x16x1x4_4x4x1_1x1_interwave_v2(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y) {
  // Check if this input needs to be padded.
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = WQ.size(0);
  int K = WQ.size(1);
  bool pad =  (K % 64 != 0);

  // Dispatch based on whether padding is needed or not.
  if (pad) {
  using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType, ABDataType,
      64,
      16,
      16,
      64,
      16,
      16,
      1,
      1,
      S<4, 16, 1>,
      S<4, 16, 1>,
      S<1, 16, 1, 4>,
      S<4, 4, 1>,
      1,
      1,
      ck::BlockGemmPipelineScheduler::Interwave,
      ck::BlockGemmPipelineVersion::v2,
        ck::tensor_operation::device::GemmSpecialization::KPadding>;
  // Run kernel instance.
  return gemm_a8w8_rowwise_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
  }
  else{
    using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType, ABDataType,
      64,
      16,
      16,
      64,
      16,
      16,
      1,
      1,
      S<4, 16, 1>,
      S<4, 16, 1>,
      S<1, 16, 1, 4>,
      S<4, 4, 1>,
      1,
      1,
      ck::BlockGemmPipelineScheduler::Interwave,
      ck::BlockGemmPipelineVersion::v2>;
  // Run kernel instance.
  return gemm_a8w8_rowwise_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
  }
}

template torch::Tensor
a8w8_rowwise_64x16x16x64_16x16_1x1_4x16x1_4x16x1_1x16x1x4_4x4x1_1x1_interwave_v2<F16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_64x16x16x64_16x16_1x1_4x16x1_4x16x1_1x16x1x4_4x4x1_1x1_interwave_v2<B16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_64x16x16x64_16x16_1x1_4x16x1_4x16x1_1x16x1x4_4x4x1_1x1_interwave_v2<F16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_64x16x16x64_16x16_1x1_4x16x1_4x16x1_1x16x1x4_4x4x1_1x1_interwave_v2<B16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);
