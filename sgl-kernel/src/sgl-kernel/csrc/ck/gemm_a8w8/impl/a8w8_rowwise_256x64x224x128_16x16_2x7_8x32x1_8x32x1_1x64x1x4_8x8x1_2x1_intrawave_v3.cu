// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "../gemm_a8w8_common.cuh"

template <typename DEDataType, typename ABDataType>
torch::Tensor
a8w8_rowwise_256x64x224x128_16x16_2x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y) {
    // Check if this input needs to be padded.
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = WQ.size(0);
  int K = WQ.size(1);
  bool pad =  (K % 128 != 0);

  // Dispatch based on whether padding is needed or not.
  if (pad) {
  using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType, ABDataType,
      256,
      64,
      224,
      128,
      16,
      16,
      2,
      7,
      S<8, 32, 1>,
      S<8, 32, 1>,
      S<1, 64, 1, 4>,
      S<8, 8, 1>,
      2,
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
      64,
      224,
      128,
      16,
      16,
      2,
      7,
      S<8, 32, 1>,
      S<8, 32, 1>,
      S<1, 64, 1, 4>,
      S<8, 8, 1>,
      2,
      1,
      ck::BlockGemmPipelineScheduler::Intrawave,
      ck::BlockGemmPipelineVersion::v3>;
  // Run kernel instance.
  return gemm_a8w8_rowwise_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
  }
}

template torch::Tensor
a8w8_rowwise_256x64x224x128_16x16_2x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<F16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_256x64x224x128_16x16_2x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<B16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_256x64x224x128_16x16_2x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<F16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_256x64x224x128_16x16_2x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<B16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template <typename DEDataType, typename ABDataType>
torch::Tensor
a8w8_rowwise_256x64x224x128_16x16_2x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3_k2(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y) {
   // Check if this input needs to be padded.
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = WQ.size(0);
  int K = WQ.size(1);
  bool pad =  (K % 256 != 0);

  // Dispatch based on whether padding is needed or not.
  if (pad) {
  using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType, ABDataType,
      256,
      64,
      224,
      128,
      16,
      16,
      2,
      7,
      S<8, 32, 1>,
      S<8, 32, 1>,
      S<1, 64, 1, 4>,
      S<8, 8, 1>,
      2,
      1,
      ck::BlockGemmPipelineScheduler::Intrawave,
      ck::BlockGemmPipelineVersion::v3,
        ck::tensor_operation::device::GemmSpecialization::KPadding>;
  // Run kernel instance.
  return gemm_a8w8_rowwise_impl<DeviceGemmInstance, 2>(XQ, WQ, x_scale, w_scale, Y);
  }
  else{
    using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType, ABDataType,
      256,
      64,
      224,
      128,
      16,
      16,
      2,
      7,
      S<8, 32, 1>,
      S<8, 32, 1>,
      S<1, 64, 1, 4>,
      S<8, 8, 1>,
      2,
      1,
      ck::BlockGemmPipelineScheduler::Intrawave,
      ck::BlockGemmPipelineVersion::v3>;
  // Run kernel instance.
  return gemm_a8w8_rowwise_impl<DeviceGemmInstance, 2>(XQ, WQ, x_scale, w_scale, Y);
  }}

template torch::Tensor
a8w8_rowwise_256x64x224x128_16x16_2x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3_k2<F16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_256x64x224x128_16x16_2x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3_k2<B16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_256x64x224x128_16x16_2x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3_k2<F16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_256x64x224x128_16x16_2x7_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3_k2<B16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);
