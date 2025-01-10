import os
import unittest

import torch
from sgl_kernel import gemm_a8w8_subblock
from sglang.srt.layers.quantization.fp8_kernel import (
    w8a8_block_fp8_matmul,
)

class TestCKGemm(unittest.TestCase):
    @staticmethod
    def ck_shape_test(M, N, K, Fp8: bool):
        num_gpus = torch.cuda.device_count()
        assert num_gpus >= 8, "At least 8 GPUs!"

        device_list = list(range(8))  # GPU ID

        # Initial
        out_triton = torch.empty([M, N], dtype=torch.float32).cuda()

        # Data chunk
        if Fp8:
            chunks_a = torch.chunk(torch.randn((M, K), dtype=torch.float32).cuda(), len(device_list), dim=0)
            chunks_b = torch.chunk(torch.randn((N, K), dtype=torch.float32).cuda(), len(device_list), dim=0)
        # INT8
        else:
            chunks_a = torch.chunk(torch.randint(-128, 127, (M, K), dtype=torch.int8).cuda(), len(device_list), dim=0)
            chunks_b = torch.chunk(torch.randint(-128, 127, (N, K), dtype=torch.int8).cuda(), len(device_list), dim=0)

        # Actual used GPU
        active_devices = device_list[:len(chunks_a)]

        # Assign for each GPU
        results = []
        for i, device in enumerate(active_devices):
            with torch.cuda.device(device):
                a_chunk = chunks_a[i].to(device)
                b_chunk = chunks_b[i].to(device)

                if Fp8:
                    a_chunk = a_chunk.to(torch.float8_e4m3fnuz)
                    b_chunk = b_chunk.to(torch.float8_e4m3fnuz)

                # Params
                block_n = 128
                block_k = 128
                ntile = (N + block_n - 1) // block_n
                ktile = (N + block_k - 1) // block_k
                alpha_row = torch.rand([a_chunk.size(0), ktile], dtype=torch.float32).to(device) / 1000
                alpha_col = torch.rand([ntile, ktile], dtype=torch.float32).to(device) / 1000

                # GPU run GEMM
                out_chunk = torch.empty([a_chunk.size(0), N], dtype=torch.float32, device=device)
                gemm_a8w8_subblock(a_chunk, b_chunk, alpha_row, alpha_col, out_chunk, block_n, block_k)

                # Results
                results.append(out_chunk.to('cuda:0'))

        # ALL gather Results
        out_triton = torch.cat(results, dim=0).cuda()
        print(f"pass: m={M}, n={N}, k={K}, output shape={out_triton.shape}")

    @staticmethod
    def ck_acc_test(M, N, K, Fp8: bool):
        num_gpus = torch.cuda.device_count()
        assert num_gpus >= 8, "At least 8 GPUs!"

        device_list = list(range(8))  # GPU ID

        # Initial
        ck_res = torch.empty([M, N], dtype=torch.float32).cuda()
        triton_res = torch.empty([M, N], dtype=torch.float32).cuda()

        # Data chunk
        if Fp8:
            chunks_a = torch.chunk(torch.randn((M, K), dtype=torch.float32).cuda(), len(device_list), dim=0)
            chunks_b = torch.chunk(torch.randn((N, K), dtype=torch.float32).cuda(), len(device_list), dim=0)
        # INT8
        else:
            chunks_a = torch.chunk(torch.randint(-5, 5, (M, K), dtype=torch.int8).cuda(), len(device_list), dim=0)
            chunks_b = torch.chunk(torch.randint(0, 5, (N, K), dtype=torch.int8).cuda(), len(device_list), dim=0)

        # Actual used GPU
        active_devices = device_list[:len(chunks_a)]

        # Assign for each GPU
        ck_res_results = []
        triton_res_results = []
        for i, device in enumerate(active_devices):
            with torch.cuda.device(device):
                a_chunk = chunks_a[i].to(device)
                b_chunk = chunks_b[i].to(device)

                if Fp8:
                    a_chunk = a_chunk.to(torch.float8_e4m3fnuz)
                    b_chunk = b_chunk.to(torch.float8_e4m3fnuz)
                
                block_n = 128
                block_k = 128
                ntile = (N + block_n - 1) // block_n
                ktile = (N + block_k - 1) // block_k
                alpha_row = torch.rand([M, ktile], dtype=torch.float32).cuda() / 1000
                alpha_col = (
                    torch.rand([ntile, ktile], dtype=torch.float32).cuda() / 1000
                )

                ck_chunk = torch.empty([a_chunk.size(0), N], dtype=torch.float32, device=device) 
                gemm_a8w8_subblock(a_chunk, b_chunk, alpha_row, alpha_col, ck_res, block_n, block_k)

                # convert to float32 to call torch.mm, same accuracy as fp8/int32
                block_size = [128,128]
                triton_chunk = w8a8_block_fp8_matmul(
                    a_chunk, b_chunk, alpha_row, alpha_col, block_size, output_dtype=a_chunk.dtype
                )

                ck_res_results.append(ck_chunk.to('cuda:0'))
                triton_res_results.append(triton_chunk.to('cuda:0'))

        # ALL gather Results
        ck_res = torch.cat(ck_res_results, dim=0).cuda()
        triton_res = torch.cat(ck_res_results, dim=0).cuda()   

        if not torch.allclose(triton_res, ck_res, 1e-4, 1e-4, True):
            from math import sqrt

            diff = triton_res - ck_res
            idx = torch.nonzero(diff, as_tuple=True)
            print(
                f"m: {M}, n: {N}, k: {K}, # not close: {idx[0].shape[0]}, "
                f" % not close: {100 * idx[0].shape[0] / (m * n):.7f}, "
                f"norm diff: {torch.linalg.vector_norm(diff) / sqrt(idx[0].shape[0])}, "
                f"max % diff: {100 * torch.max(torch.abs(diff[idx] / triton_res[idx])):.4f}"
            )
        else:
            print(f"m: {M}, n: {N}, k: {K} all close!")
        print("======")

if __name__ == "__main__":
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
                TestCKGemm.ck_shape_test(m, n, k, Fp8=True)
                TestCKGemm.ck_acc_test(m, n, k, Fp8=True)

