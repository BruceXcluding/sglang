#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#undef __HIP_NO_HALF_OPERATORS__
#undef __HIP_NO_HALF_CONVERSIONS__

#include <iostream>
#include <numeric>
#include <initializer_list>
#include <cstdlib>

#include <ATen/ATen.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDAContext.h>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/gemm_specialization.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp"

#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"
#include "ck/library/utility/literals.hpp"
#include "ck/library/reference_tensor_operation/cpu/reference_gemm.hpp"
#include "ck/library/utility/check_err.hpp"

#include "ck/utility/blkgemmpipe_scheduler.hpp"

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using F16 = ck::half_t;
using B16 = ck::bhalf_t;
using F8 = ck::f8_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using A0DataType       = FP8;
using A1DataType       = F32;
using B0DataType       = FP8;
using B1DataType       = F32;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DsDataType       = ck::Tuple<>;
using EDataType        = BF16;

using A0Layout = Row;
using B0Layout = Col;
using D0Layout = Row;
using D1Layout = Col;
using DsLayout = ck::Tuple<>;
using ELayout  = Row;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CDEElementOp = PassThrough;

static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;

static constexpr ck::index_t Scale_Block_M = 1;
static constexpr ck::index_t Scale_Block_N = 128;
static constexpr ck::index_t Scale_Block_K = 128;

using DeviceGemmInstance_ = ck::tensor_operation::device::DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3
    // clang-format off
         <Row, Col, DsLayout, ELayout,
          A0DataType, A1DataType, B0DataType, B1DataType, DsDataType, EDataType, AccDataType, CShuffleDataType, 
          AElementOp,  BElementOp, CDEElementOp, GemmSpec,
          256, Scale_Block_M, Scale_Block_N, Scale_Block_K,  // BlockSize, ScaleBlockM, ScaleBlockN, ScaleBlockK 
          128, 128,     // MPerBlock, NPerBlock, KPerBlock 
          128, 16, 16,
          16,   16,
          4,    4,
          S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
          S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
          1,    2,  S<1, 32, 1, 8>,  S<8>,
          ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3, FP8>;

template <typename DeviceGemmInstance=DeviceGemmInstance_, ck::index_t SplitK=1>
__forceinline__ torch::Tensor gemm_a8w8_subblockwise_impl(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y)
{
    int M = XQ.size(0);
    int N = WQ.size(0);
    int K = XQ.size(1);

    int StrideA = XQ.stride(-2);
    int StrideB = WQ.stride(-2);
    int StrideE = N;
    int Scale_Stride_AM = (K + Scale_Block_K - 1) / Scale_Block_K;
    int Scale_Stride_BN = (K + Scale_Block_K - 1) / Scale_Block_K;

    auto device_gemm = DeviceGemmInstance{};
    auto invoker = device_gemm.MakeInvoker();
    // std::cout<<"Kernel selected: "<<device_gemm.GetTypeString()<<", SplitK: "<<SplitK<<std::endl;

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumDTensor = DeviceGemmInstance::NumDTensor;
    auto argument  = device_op.MakeArgument(XQ.data_ptr(),
                        WQ.data_ptr(),
                        std::array<const void*, NumDTensor>{},
                        Y.data_ptr(),
                        M,
                        N,
                        K,
                        StrideA,
                        StrideB,
                        std::array<ck::index_t, NumDTensor>{},
                        StrideE,
                        x_scale.data_ptr(),
                        w_scale.data_ptr(),
                        a_element_op,
                        b_element_op,
                        cde_element_op);
    
    TORCH_CHECK(device_gemm.IsSupportedArgument(argument), "This GEMM is not supported!");
    
    invoker.Run(argument, StreamConfig{at::cuda::getCurrentCUDAStream().stream()});
    return Y;
}

#endif // USE_ROCM