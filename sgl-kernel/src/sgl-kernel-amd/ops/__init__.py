# TODO (amd) add customrized kernel
from sgl_kernel.ops._kernels import gemm_a8w8_subblock as _gemm_a8w8_subblock
from sgl_kernel.ops._kernels import moe_align_block_size as _moe_align_block_size


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


def gemm_a8w8_subblock(XQ, WQ, x_scale, w_scale, Y):
    """
    Perform GEMM operation with a8w8 subblock kernel.

    Parameters:
    XQ (torch.Tensor): The input tensor XQ.
    WQ (torch.Tensor): The weight tensor WQ.
    x_scale (torch.Tensor): Scaling factors for XQ.
    w_scale (torch.Tensor): Scaling factors for WQ.
    Y (torch.Tensor): The output tensor to store the result.

    Returns:
    torch.Tensor: The result tensor Y after performing the GEMM operation.
    """

    _gemm_a8w8_subblock(XQ, WQ, x_scale, w_scale, Y)
