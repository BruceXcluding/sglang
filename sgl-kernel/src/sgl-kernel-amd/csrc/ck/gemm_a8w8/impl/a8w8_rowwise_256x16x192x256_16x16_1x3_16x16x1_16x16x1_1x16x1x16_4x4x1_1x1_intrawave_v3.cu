// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "../gemm_a8w8_common.cuh"

template <typename DEDataType, typename ABDataType>
torch::Tensor
a8w8_rowwise_256x16x192x256_16x16_1x3_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y) {
    
  int K = WQ.size(1);
  bool pad = (K % 256 != 0);
  
  if (pad) {
  using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType, ABDataType,
      256,
      16,
      192,
      256,
      16,
      16,
      1,
      3,
      S<16, 16, 1>,
      S<16, 16, 1>,
      S<1, 16, 1, 16>,
      S<4, 4, 1>,
      1,
      1,
      ck::BlockGemmPipelineScheduler::Intrawave,
      ck::BlockGemmPipelineVersion::v3,
        ck::tensor_operation::device::GemmSpecialization::KPadding>;
  // Run kernel instance.
  return gemm_a8w8_rowwise_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
  }
  else{
    using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType, ABDataType,
      256,
      16,
      192,
      256,
      16,
      16,
      1,
      3,
      S<16, 16, 1>,
      S<16, 16, 1>,
      S<1, 16, 1, 16>,
      S<4, 4, 1>,
      1,
      1,
      ck::BlockGemmPipelineScheduler::Intrawave,
      ck::BlockGemmPipelineVersion::v3>;
  // Run kernel instance.
  return gemm_a8w8_rowwise_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
  }
}

template torch::Tensor
a8w8_rowwise_256x16x192x256_16x16_1x3_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3<F16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_256x16x192x256_16x16_1x3_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3<B16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_256x16x192x256_16x16_1x3_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3<F16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_256x16x192x256_16x16_1x3_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3<B16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);
