#include "../gemm_a8w8_subblock_common.cuh"

template <typename DEDataType, typename ABDataType>
torch::Tensor
sample_instance( torch::Tensor& XQ,
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
  using DeviceGemmInstance = DeviceGemmHelper<..>
  return gemm_a8w8_subblockwise_impl<DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);
}