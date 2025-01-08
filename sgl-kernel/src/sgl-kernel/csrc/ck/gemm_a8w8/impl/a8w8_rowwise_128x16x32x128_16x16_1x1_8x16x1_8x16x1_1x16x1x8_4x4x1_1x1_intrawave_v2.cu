// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "../gemm_a8w8_common.cuh"

template <typename DEDataType, typename ABDataType>
torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y) {
  // The smallest kernel we have available. Works well for memory bound shapes.

  // Check if this input needs to be padded.
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = WQ.size(0);
  int K = WQ.size(1);
  bool pad =  (K % 128 != 0);
  if (pad) {
    using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType, ABDataType,
        128,
        16,
        32,
        128,
        16,
        16,
        1,
        1,
        S<8, 16, 1>,
        S<8, 16, 1>,
        S<1, 16, 1, 8>,
        S<4, 4, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v2,
        ck::tensor_operation::device::GemmSpecialization::KPadding>;
    // Run kernel instance.
    return gemm_a8w8_rowwise_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
  } else{
    using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType, ABDataType,
        128,
        16,
        32,
        128,
        16,
        16,
        1,
        1,
        S<8, 16, 1>,
        S<8, 16, 1>,
        S<1, 16, 1, 8>,
        S<4, 4, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v2>;
    // Run kernel instance.
    return gemm_a8w8_rowwise_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
  }
}

template <typename DEDataType, typename ABDataType>
torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2_k4(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y) {
  // The smallest kernel we have available. Works well for memory bound shapes.

  // Check if this input needs to be padded.
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = WQ.size(0);
  int K = WQ.size(1);
  bool pad =  (K % 512 != 0);
  if (pad) {
    using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType, ABDataType,
        128,
        16,
        32,
        128,
        16,
        16,
        1,
        1,
        S<8, 16, 1>,
        S<8, 16, 1>,
        S<1, 16, 1, 8>,
        S<4, 4, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v2,
        ck::tensor_operation::device::GemmSpecialization::KPadding>;
    // Run kernel instance.
    return gemm_a8w8_rowwise_impl<DeviceGemmInstance, 4>(XQ, WQ, x_scale, w_scale, Y);
  } else{
    using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType, ABDataType,
        128,
        16,
        32,
        128,
        16,
        16,
        1,
        1,
        S<8, 16, 1>,
        S<8, 16, 1>,
        S<1, 16, 1, 8>,
        S<4, 4, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v2>;
    // Run kernel instance.
    return gemm_a8w8_rowwise_impl<DeviceGemmInstance, 4>(XQ, WQ, x_scale, w_scale, Y);
  }
}

template <typename DEDataType, typename ABDataType>
torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2_k8(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y) {
  // The smallest kernel we have available. Works well for memory bound shapes.

  // Check if this input needs to be padded.
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = WQ.size(0);
  int K = WQ.size(1);
  bool pad =  (K % 1024 != 0);
  if (pad) {
    using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType, ABDataType,
        128,
        16,
        32,
        128,
        16,
        16,
        1,
        1,
        S<8, 16, 1>,
        S<8, 16, 1>,
        S<1, 16, 1, 8>,
        S<4, 4, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v2,
        ck::tensor_operation::device::GemmSpecialization::KPadding>;
    // Run kernel instance.
    return gemm_a8w8_rowwise_impl<DeviceGemmInstance, 8>(XQ, WQ, x_scale, w_scale, Y);
  } else{
    using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType, ABDataType,
        128,
        16,
        32,
        128,
        16,
        16,
        1,
        1,
        S<8, 16, 1>,
        S<8, 16, 1>,
        S<1, 16, 1, 8>,
        S<4, 4, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v2>;
    // Run kernel instance.
    return gemm_a8w8_rowwise_impl<DeviceGemmInstance, 8>(XQ, WQ, x_scale, w_scale, Y);
  }
}

template <typename DEDataType, typename ABDataType>
torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2_k16(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y) {
  // The smallest kernel we have available. Works well for memory bound shapes.

  // Check if this input needs to be padded.
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = WQ.size(0);
  int K = WQ.size(1);
  bool pad =  (K % 2048 != 0);
  if (pad) {
    using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType, ABDataType,
        128,
        16,
        32,
        128,
        16,
        16,
        1,
        1,
        S<8, 16, 1>,
        S<8, 16, 1>,
        S<1, 16, 1, 8>,
        S<4, 4, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v2,
        ck::tensor_operation::device::GemmSpecialization::KPadding>;
    // Run kernel instance.
    return gemm_a8w8_rowwise_impl<DeviceGemmInstance, 16>(XQ, WQ, x_scale, w_scale, Y);
  } else{
    using DeviceGemmInstance = DeviceGemmHelper<
      DEDataType, ABDataType,
        128,
        16,
        32,
        128,
        16,
        16,
        1,
        1,
        S<8, 16, 1>,
        S<8, 16, 1>,
        S<1, 16, 1, 8>,
        S<4, 4, 1>,
        1,
        1,
        ck::BlockGemmPipelineScheduler::Intrawave,
        ck::BlockGemmPipelineVersion::v2>;
    // Run kernel instance.
    return gemm_a8w8_rowwise_impl<DeviceGemmInstance, 16>(XQ, WQ, x_scale, w_scale, Y);
  }
}

template torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2<F16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2<B16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2_k4<F16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2_k4<B16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2_k8<F16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2_k8<B16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);
    
template torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2_k16<F16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2_k16<B16, I8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2<F16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2<B16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2_k4<F16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2_k4<B16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2_k8<F16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2_k8<B16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);
    
template torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2_k16<F16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);

template torch::Tensor
a8w8_rowwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2_k16<B16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y);