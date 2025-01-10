import os
import unittest

import torch
from sgl_kernel import gemm_a8w8_subblock

from sglang.srt.layers.quantization.fp8_kernel import w8a8_block_fp8_matmul


class TestCKGemm(unittest.TestCase):

    @staticmethod
    def ck_shape_test(Fp8: bool):
        for n, k in [
            (1536, 7168),
            (3072, 1536),
            (576, 7168),
            (7168, 256),
            (7168, 2048),
            (4608, 7168),
            (7168, 2304),
            (512, 7168),
            (4096, 512),
        ]:
            for m in [
                1,
                2,
                4,
                8,
                16,
                24,
                32,
                48,
                64,
                96,
                128,
                256,
                512,
                1024,
                1536,
                2048,
                4096,
                8192,
                16384,
                20480,
            ]:
                if Fp8:
                    a = torch.randn((m, k), dtype=torch.float32).cuda()
                    b = torch.randn((n, k), dtype=torch.float32).cuda()
                    a_8 = a.to(torch.float8_e4m3fnuz).cuda()
                    b_8 = b.to(torch.float8_e4m3fnuz).cuda()
                    block_n = 128
                    block_k = 128
                    ntile = (n + block_n - 1) // block_n
                    ktile = (n + block_k - 1) // block_k
                    alpha_row = (
                        torch.rand([m, ktile], dtype=torch.float32).cuda() / 1000
                    )
                    alpha_col = (
                        torch.rand([ntile, ktile], dtype=torch.float32).cuda() / 1000
                    )

                    quantiles = [0.5, 0.2, 0.8]
                    out_triton = torch.empty(
                        [a.shape[0], b.shape[0]], dtype=torch.float32, device=a.device
                    )
                    gemm_a8w8_subblock(
                        a_8, b_8, alpha_row, alpha_col, out_triton, block_n, block_k
                    )
                    print(f"pass : {m}, {n}, {k}")
                else:
                    a = torch.randint(-128, 127, (m, k), dtype=torch.int8).cuda()
                    b = torch.randint(-128, 127, (n, k), dtype=torch.int8).cuda()
                    block_n = 128
                    block_k = 128
                    ntile = (n + block_n - 1) // block_n
                    ktile = (n + block_k - 1) // block_k
                    alpha_row = (
                        torch.rand([m, ktile], dtype=torch.float32).cuda() / 1000
                    )
                    alpha_col = (
                        torch.rand([ntile, ktile], dtype=torch.float32).cuda() / 1000
                    )
                    quantiles = [0.5, 0.2, 0.8]
                    out_triton = torch.empty(
                        [a.shape[0], b.shape[0]], dtype=torch.float32, device=a.device
                    )
                    gemm_a8w8_subblock(
                        a, b, alpha_row, alpha_col, out_triton, block_n, block_k
                    )
                    print(f"pass : {m}, {n}, {k}")

    @staticmethod
    def ck_acc_test(Fp8: bool):
        for m in [
            1,
            2,
            4,
            8,
            16,
            24,
            32,
            48,
            64,
            96,
            128,
            256,
            512,
            1024,
            1536,
            2048,
            4096,
            8192,
            16384,
            20480,
        ]:
            for n, k in [
                (1536, 7168),
                (3072, 1536),
                (576, 7168),
                (7168, 256),
                (7168, 2048),
                (4608, 7168),
                (7168, 2304),
                (512, 7168),
                (4096, 512),
            ]:
                if Fp8:
                    a = torch.randn((m, k), dtype=torch.float32).cuda()
                    b = torch.randn((n, k), dtype=torch.float32).cuda()
                    a_8 = a.to(torch.float8_e4m3fnuz).cuda()
                    b_8 = b.to(torch.float8_e4m3fnuz).cuda()
                    block_n = 128
                    block_k = 128
                    ntile = (n + block_n - 1) // block_n
                    ktile = (n + block_k - 1) // block_k
                    alpha_row = (
                        torch.rand([m, ktile], dtype=torch.float32).cuda() / 1000
                    )
                    alpha_col = (
                        torch.rand([ntile, ktile], dtype=torch.float32).cuda() / 1000
                    )

                    ck_res = torch.zeros(
                        [a_8.shape[0], b_8.shape[0]],
                        dtype=torch.float32,
                        device=a.device,
                    )
                    gemm_a8w8_subblock(
                        a_8, b_8, alpha_row, alpha_col, ck_res, block_n, block_k
                    )

                    # convert to float32 to call torch.mm, same accuracy as fp8/int32
                    block_size = [128, 128]
                    triton_res = w8a8_block_fp8_matmul(
                        a_8,
                        b_8,
                        alpha_row,
                        alpha_col,
                        block_size,
                        output_dtype=a_8.dtype,
                    )
                # INT8
                else:
                    a = torch.randint(-5, 5, (m, k), dtype=torch.int8).cuda()
                    b = torch.randint(0, 5, (n, k), dtype=torch.int8).cuda()
                    block_n = 128
                    block_k = 128
                    ntile = (n + block_n - 1) // block_n
                    ktile = (n + block_k - 1) // block_k
                    alpha_row = (
                        torch.rand([m, ktile], dtype=torch.float32).cuda() / 1000
                    )
                    alpha_col = (
                        torch.rand([ntile, ktile], dtype=torch.float32).cuda() / 1000
                    )

                    ck_res = torch.zeros(
                        [a.shape[0], b.shape[0]], dtype=torch.float32, device=a.device
                    )
                    gemm_a8w8_subblock(
                        a, b, alpha_row, alpha_col, ck_res, block_n, block_k
                    )

                    # convert to float32 to call torch.mm, same accuracy as int8/int32
                    torch_res = torch.mm(a.to(torch.float32), b.t().to(torch.float32))
                    torch_rowwise = torch.mul(
                        torch.mul(torch_res, alpha_row.to(torch.float)),
                        alpha_col.to(torch.float),
                    ).to(torch.half)

                if not torch.allclose(triton_res, ck_res, 1e-4, 1e-4, True):
                    from math import sqrt

                    diff = triton_res - ck_res
                    idx = torch.nonzero(diff, as_tuple=True)
                    print(
                        f"m: {m}, n: {n}, k: {k}, # not close: {idx[0].shape[0]}, "
                        f" % not close: {100 * idx[0].shape[0] / (m * n):.7f}, "
                        f"norm diff: {torch.linalg.vector_norm(diff) / sqrt(idx[0].shape[0])}, "
                        f"max % diff: {100 * torch.max(torch.abs(diff[idx] / torch_rowwise[idx])):.4f}"
                    )
                else:
                    print(f"m: {m}, n: {n}, k: {k} all close!")
                print("======")


if __name__ == "__main__":
    TestCKGemm.ck_shape_test(Fp8=True)
    TestCKGemm.ck_acc_test(Fp8=True)
