#include "../gemm_a8w8_subblock_common.cuh"

template <typename DEDataType, typename ABDataType>
torch::Tensor
a8w8_subblockwise_128x16x32x128_16x16_1x1_16x16_intrawave_v1_kernel( torch::Tensor& XQ,
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
  using DeviceGemmInstance = DeviceGemmHelper<64, 16, 16, 128, 16, 16, 1, 1, 16, 16, S<8, 8, 1>, S<8, 8, 1>, S<1, 16, 1, 4>, S<4, 4, 1>,  ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v2> ;
  return gemm_a8w8_subblockwise_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
}