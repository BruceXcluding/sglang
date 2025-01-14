#include "../gemm_a8w8_subblock_common.cuh"

template <typename DEDataType, typename ABDataType>
torch::Tensor
a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v1(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y) 
{
  int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
  int N = WQ.size(0);
  int K = WQ.size(1);
  bool k_pad = (K % 128 != 0);
  bool m_pad = (M % 128 != 0);
  bool n_pad = (N % 128 != 0);
  using DeviceGemmInstance = DeviceGemmHelper<
    DEDataType, ABDataType,
    128,
    32,
    16,
    128,
    16,
    16,
    1,
    1,
    S<8, 16, 1>,
    S<8, 16, 1>,
    S<1, 16, 1, 8>,
    S<4, 4, 1>,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v1> ;
  return a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v1<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
}

template torch::Tensor
a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v1<B16, F8>(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y
)