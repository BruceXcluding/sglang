# TODO (amd) add customrized kernel
from sgl_kernel.ops._kernels import moe_align_block_size as _moe_align_block_size
from sgl_kernel.ops._kernels import gemm_a8w8 as _gemm_a8w8

def moe_align_block_size(
    topk_ids,
    num_experts,
    block_size,
    sorted_token_ids,
    experts_ids,
    num_tokens_post_pad,
    token_cnts_buffer,
    cumsum_buffer,
):
    _moe_align_block_size(
        topk_ids,
        num_experts,
        block_size,
        sorted_token_ids,
        experts_ids,
        num_tokens_post_pad,
        token_cnts_buffer,
        cumsum_buffer,
    )

def gemm_a8w8(
    XQ,
    WQ,
    x_scale,
    w_scale,
    Y):
    _gemm_a8w8(XQ, WQ, x_scale, w_scale, Y)