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

using I8  = int8_t;
using I32 = int;
using F16 = ck::half_t;
using B16 = ck::bhalf_t;
using F8 = ck::f8_t;
using F32 = float;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

using ALayout = Row;
using BLayout = Col;
using D0Layout = Row;
using D1Layout = Col;
using DsLayout = ck::Tuple<D0Layout, D1Layout>;
using ELayout = Row;

using PassThrough = ck::tensor_operation::element_wise::PassThrough;

using AElementOp = PassThrough;
using BElementOp = PassThrough;

struct RowwiseScale
{
    template <typename E, typename C, typename D0, typename D1>
    __host__ __device__ __forceinline__ constexpr void
    operator()(E& e, const C& c, const D0& d0, const D1& d1) const
    {
        static_assert(std::is_same_v<E, F16> || std::is_same_v<E, B16>);
        static_assert(std::is_same_v<D0, D1> && std::is_same_v<D0, E>);
        static_assert(std::is_same_v<C, I32> || std::is_same_v<C, F32>);
        const F32 x0_f =
            ck::type_convert<F32>(c) * ck::type_convert<F32>(d0) * ck::type_convert<F32>(d1);

        e = ck::type_convert<E>(x0_f);
    }
};

using CDEElementOp = RowwiseScale;

template <typename ABDataType>
struct DTypeHelper;

template <>
struct DTypeHelper<I8> {
    using Acc = I32;
    using CShuffle = I32;
    using Compute = I8;
};

template <>
struct DTypeHelper<F8> {
    using Acc = F32;
    using CShuffle = F32;
    using Compute = F8;
};

template <typename DEDataType>
using DsDataType = ck::Tuple<DEDataType, DEDataType>;

template <
    typename DEDataType,
    typename ABDataType,
    int BLOCK_SIZE,
    int MBLOCK,
    int NBLOCK,
    int KBLOCK,
    int WAVE_TILE_M,
    int WAVE_TILE_N,
    int WAVE_MAP_M,
    int WAVE_MAP_N,
    typename ABLOCK_TRANSFER,
    typename BBLOCK_TRANSFER,
    typename CBLOCK_TRANSFER,
    typename CBLOCK_SPV,
    int CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
    int CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
    ck::BlockGemmPipelineScheduler LOOP_SCHED,
    ck::BlockGemmPipelineVersion PIPELINE_VERSION,
    auto GEMM_SPEC =
        ck::tensor_operation::device::GemmSpecialization::Default>
    using DeviceGemmHelper =
    ck::tensor_operation::device::DeviceGemmMultiD_Xdl_CShuffle_V3<
        ALayout,
        BLayout,
        DsLayout,
        ELayout,
        ABDataType,
        ABDataType,
        DsDataType<DEDataType>,
        DEDataType,
        typename DTypeHelper<ABDataType>::Acc,
        typename DTypeHelper<ABDataType>::CShuffle,
        AElementOp,
        BElementOp,
        CDEElementOp,
        GEMM_SPEC,
        BLOCK_SIZE, // Block Size
        MBLOCK, // M per Block
        NBLOCK, // N per Block
        KBLOCK, // K per Block
        16, // AK1
        16, // BK1
        WAVE_TILE_M, // M per Xdl
        WAVE_TILE_N, // N per Xdl
        WAVE_MAP_M, // Mxdl per Wave
        WAVE_MAP_N, // Nxdl per Wave
        ABLOCK_TRANSFER,
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        16,
        16,
        0,
        BBLOCK_TRANSFER,
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        16,
        16,
        0,
        CSHUFFLE_MX_PER_WAVE_PERSHUFFLE,
        CSHUFFLE_NX_PER_WAVE_PERSHUFFLE,
        CBLOCK_TRANSFER,
        CBLOCK_SPV,
        LOOP_SCHED,
        PIPELINE_VERSION,
        typename DTypeHelper<ABDataType>::Compute>;


template <typename DeviceGemmInstance,
          ck::index_t SplitK=1>
__forceinline__ torch::Tensor gemm_a8w8_rowwise_impl(
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

    auto device_gemm = DeviceGemmInstance{};
    auto invoker = device_gemm.MakeInvoker();
    // std::cout<<"Kernel selected: "<<device_gemm.GetTypeString()<<", SplitK: "<<SplitK<<std::endl;

    auto a_element_op = AElementOp{};
    auto b_element_op = BElementOp{};
    auto cde_element_op = CDEElementOp{};

    constexpr ck::index_t NumDTensor = DeviceGemmInstance::NumDTensor;
    
    auto argument = device_gemm.MakeArgument(
        XQ.data_ptr(),
        WQ.data_ptr(),
        std::array<const void*, NumDTensor>{
            w_scale.data_ptr(),
            x_scale.data_ptr()},
        Y.data_ptr(),
        M,
        N,
        K,
        StrideA,
        StrideB,
        std::array<ck::index_t, NumDTensor>{0, 0},
        StrideE,
        SplitK,
        a_element_op,
        b_element_op,
        cde_element_op
    );
    TORCH_CHECK(device_gemm.IsSupportedArgument(argument), "This GEMM is not supported!");
    
    invoker.Run(argument, StreamConfig{at::cuda::getCurrentCUDAStream().stream()});
    return Y;
}

#endif // USE_ROCM
