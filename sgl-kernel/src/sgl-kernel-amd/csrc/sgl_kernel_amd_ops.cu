#include "utils.hpp"

// moe_align_block_size
void moe_align_block_size(torch::Tensor topk_ids, int64_t num_experts, int64_t block_size,
                          torch::Tensor sorted_token_ids, torch::Tensor experts_ids, torch::Tensor num_tokens_post_pad,
                          torch::Tensor token_cnts_buffer, torch::Tensor cumsum_buffer);

torch::Tensor gemm_a8w8_subblock(
  torch::Tensor& XQ,
  torch::Tensor& WQ,
  torch::Tensor& x_scale,
  torch::Tensor& w_scale,
  torch::Tensor& Y);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // moe_align_block_size
  m.def("moe_align_block_size", &moe_align_block_size, "MOE Align Block Size (ROCM)");
  m.def("gemm_a8w8_subblock", &gemm_a8w8_subblock, "gemm_a8w8_subblock");
}
