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

using A0DataType       = F8;
using A1DataType       = F32;
using B0DataType       = F8;
using B1DataType       = F32;
using AccDataType      = F32;
using CShuffleDataType = F32;
using DsDataType       = ck::Tuple<>;
using EDataType        = B16;

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

// Now a helper function that dynamically selects the kernel based on `Scale_Block_N` and `Scale_Block_K`
template<index_t BlockSize,      
        index_t MPerBlock, index_t NPerBlock, index_t KPerBlock,
        index_t AK1, index_t BK1,
        index_t MPerXDL, index_t NPerXDL,
        index_t MXdlPerWave, index_t NXdlPerWave,       
        typename ABlockTransferThreadClusterLengths_AK0_M_AK1,  
        typename BBlockTransferThreadClusterLengths_BK0_N_BK1,
        typename CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
        typename CDEShuffleBlockTransferScalarPerVectors,        
        BlockGemmPipelineScheduler BlkGemmPipeSched = BlockGemmPipelineScheduler::Intrawave,
        BlockGemmPipelineVersion BlkGemmPipelineVer = BlockGemmPipelineVersion::v1 >
    using DeviceGemmHelper = 
        ck::tensor_operation::device::DeviceGemmMultiD_ABScale_Xdl_CShuffle_V3<
            A0Layout,B0Layout, DsLayout, ELayout , 
            F8, F32, F8, F32,
            Tuple<>, BF16, F32, F32,
            AElementOp, BElementOp, CDEElementOp,     
            GemmSpec,    
            BlockSize, 1, 128, 128 ,  
            MPerBlock, NPerBlock, KPerBlock,
            AK1, BK1,
            MPerXDL, NPerXDL,
            MXdlPerWave, NXdlPerWave,   
            ABlockTransferThreadClusterLengths_AK0_M_AK1 ,
            S<1, 0, 2>,   S<1, 0, 2>, 
            2, 16, 16, 0,  
            BBlockTransferThreadClusterLengths_BK0_N_BK1,
            S<1, 0, 2>,    S<1, 0, 2>,
            2, 16, 16, 0,  
            1, 1, 
            CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
            CDEShuffleBlockTransferScalarPerVectors,
            BlkGemmPipeSched , 
            BlkGemmPipelineVer, 
            F8 > ; 

// Wrapper function that dynamically selects gemm instances 
templat<typename DeviceGemmInstance, ck::index_t SplitK=1>
__forceinline__ torch::Tensor gemm_a8w8_subblockwise_impl(
    torch::Tensor& XQ,
    torch::Tensor& WQ,
    torch::Tensor& x_scale,
    torch::Tensor& w_scale,
    torch::Tensor& Y,
    int Scale_Block_N=128,  // Dynamic parameter
    int Scale_Block_K=128   // Dynamic parameter
)
{
    int M = XQ.size(0);
    int N = WQ.size(0);
    int K = XQ.size(1);

    int StrideA = XQ.stride(-2);
    int StrideB = WQ.stride(-2);
    int StrideE = N;

    // TODO: fix me 
    using DeviceGemmKernel = DeviceGemmInstance<Scale_Block_N, Scale_Block_K>;
    auto device_gemm = DeviceGemmKernel{};
    auto invoker = device_gemm.MakeInvoker();

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumDTensor = DsDataType::Size();
    auto argument  = device_gemm.MakeArgument(XQ.data_ptr(),
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
