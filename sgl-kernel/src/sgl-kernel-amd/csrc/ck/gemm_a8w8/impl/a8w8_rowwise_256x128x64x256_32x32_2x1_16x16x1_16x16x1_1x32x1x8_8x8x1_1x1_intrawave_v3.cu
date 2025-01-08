// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "../gemm_a8w8_common.cuh"

template <typename DEDataType, typename ABDataType>
torch::Tensor
a8w8_rowwise_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y) {
  // A kernel that seems to work well on mid sized tensors.

  // Check if this input needs to be padded.
  int K = WQ.size(1);
  bool pad =  (K % 256 != 0);

  // Dispatch based on whether padding is needed or not.
  if (pad) {
    using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType, ABDataType,
        256,
        128,
        64,
        256,
        32,
        32,
        2,
        1,
        S<16, 16, 1>,
        S<16, 16, 1>,
        S<1, 32, 1, 8>,
        S<8, 8, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v3,
        ck::tensor_operation::device::GemmSpecialization::KPadding>;
    // Run kernel instance.
    return gemm_a8w8_rowwise_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
  } else {
    using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType, ABDataType,
        256,
        128,
        64,
        256,
        32,
        32,
        2,
        1,
        S<16, 16, 1>,
        S<16, 16, 1>,
        S<1, 32, 1, 8>,
        S<8, 8, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v3,
        ck::tensor_operation::device::GemmSpecialization::Default>;
    // Run kernel instance.
    return gemm_a8w8_rowwise_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
  }
}

template <typename DEDataType, typename ABDataType>
torch::Tensor
a8w8_rowwise_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k2(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y) {
  // A kernel that seems to work well on mid sized tensors.

  // Check if this input needs to be padded.
  int K = WQ.size(1);
  bool pad =  (K % 512 != 0);

  // Dispatch based on whether padding is needed or not.
  if (pad) {
    using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType, ABDataType,
        256,
        128,
        64,
        256,
        32,
        32,
        2,
        1,
        S<16, 16, 1>,
        S<16, 16, 1>,
        S<1, 32, 1, 8>,
        S<8, 8, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v3,
        ck::tensor_operation::device::GemmSpecialization::KPadding>;
    // Run kernel instance.
    return gemm_a8w8_rowwise_impl<DeviceGemmInstance, 2>(XQ, WQ, x_scale, w_scale, Y);
  } else {
    using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType, ABDataType,
        256,
        128,
        64,
        256,
        32,
        32,
        2,
        1,
        S<16, 16, 1>,
        S<16, 16, 1>,
        S<1, 32, 1, 8>,
        S<8, 8, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v3,
        ck::tensor_operation::device::GemmSpecialization::Default>;
    // Run kernel instance.
    return gemm_a8w8_rowwise_impl<DeviceGemmInstance, 2>(XQ, WQ, x_scale, w_scale, Y);
  }
}

template torch::Tensor
a8w8_rowwise_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<F16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<B16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k2<F16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k2<B16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<F16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3<B16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k2<F16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_256x128x64x256_32x32_2x1_16x16x1_16x16x1_1x32x1x8_8x8x1_1x1_intrawave_v3_k2<B16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);