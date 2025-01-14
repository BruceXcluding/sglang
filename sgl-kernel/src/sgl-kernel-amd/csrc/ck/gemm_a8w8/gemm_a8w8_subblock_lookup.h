#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
#ifdef USE_ROCM

#include <torch/extension.h>

using SubblockwiseKernel = std::function<
    torch::Tensor(torch::Tensor&, torch::Tensor&,
        torch::Tensor&, torch::Tensor&, torch::Tensor&)>;

// Define a custom hash function for std::tuple<int, int, int>
struct IntTupleHash {
   size_t operator()(const std::tuple<int, int, int>& t) const {
   auto hash1 = std::hash<int>{}(std::get<0>(t));
   auto hash2 = std::hash<int>{}(std::get<1>(t));
   auto hash3 = std::hash<int>{}(std::get<2>(t));
   return hash1 ^ hash2 ^ hash3;
   }
};

using SubblockwiseKernelMap = std::unordered_map<
    std::tuple<int, int, int>,
    SubblockwiseKernel,
    IntTupleHash>;

class KernelLookupMap {
public:
   auto find(int M, int N, int K) const {
      return table_.find({M, N, K});
   }
   auto end() const {
      return table_.end();
   }
private:
  const SubblockwiseKernelMap table_ = {
      /* DeepSeek-v3 TP8 instance */
      /* 512, 7168*/
      {{16, 512, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{32, 512, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{64, 512, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{128, 512, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{256, 512, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{512, 512, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{1024, 512, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{2048, 512, 7168},
      a8w8_subblockwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{4096, 512, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{8192, 512, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{16384, 512, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{20480, 512, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      
      /* 576, 7168 */
      {{16, 576, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{32, 576, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{64, 576, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{128, 576, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{256, 576, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{512, 576, 7168},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_intrawave_v2},
      {{1024, 576, 7168},
      a8w8_subblockwise_128x64x32x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{2048, 576, 7168},
      a8w8_subblockwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_intrawave_v2},
      {{4096, 576, 7168},
      a8w8_subblockwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{8192, 576, 7168},
      a8w8_subblockwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{16384, 576, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{20480, 576, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      
      /* 1536, 7168 */
      {{16, 1536, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{32, 1536, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{64, 1536, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{128, 1536, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{256, 1536, 7168},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_intrawave_v2},
      {{512, 1536, 7168},
      a8w8_subblockwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{1024, 1536, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{2048, 1536, 7168},
      a8w8_subblockwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{4096, 1536, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{8192, 1536, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{16384, 1536, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{20480, 1536, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      
      /* 3072, 1536 */
      {{16, 3072, 1536},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{32, 3072, 1536},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{64, 3072, 1536},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{128, 3072, 1536},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{256, 3072, 1536},
      a8w8_subblockwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{512, 3072, 1536},
      a8w8_subblockwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_intrawave_v2},
      {{1024, 3072, 1536},
      a8w8_subblockwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{2048, 3072, 1536},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{4096, 3072, 1536},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{8192, 3072, 1536},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{16384, 3072, 1536},
      a8w8_subblockwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{20480, 3072, 1536},
      a8w8_subblockwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      
      /* 4096, 512 */
      {{16, 4096, 512},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{32, 4096, 512},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_intrawave_v2},
      {{64, 4096, 512},
      a8w8_subblockwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_intrawave_v2},
      {{128, 4096, 512},
      a8w8_subblockwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_intrawave_v2},
      {{256, 4096, 512},
      a8w8_subblockwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{512, 4096, 512},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{1024, 4096, 512},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{2048, 4096, 512},
      a8w8_subblockwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{4096, 4096, 512},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{8192, 4096, 512},
      a8w8_subblockwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{16384, 4096, 512},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{20480, 4096, 512},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      
      /* 4608, 7168 */
      {{16, 4608, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{32, 4608, 7168},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_intrawave_v2},
      {{64, 4608, 7168},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_intrawave_v2},
      {{128, 4608, 7168},
      a8w8_subblockwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_intrawave_v2},
      {{256, 4608, 7168},
      a8w8_subblockwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{512, 4608, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{1024, 4608, 7168},
      a8w8_subblockwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{2048, 4608, 7168},
      a8w8_subblockwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{4096, 4608, 7168},
      a8w8_subblockwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{8192, 4608, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{16384, 4608, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{20480, 4608, 7168},
      a8w8_subblockwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      
      /* 7168, 256 */
      {{16, 7168, 256},
      a8w8_subblockwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x4_4x4x1_intrawave_v1},
      {{32, 7168, 256},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_intrawave_v1},
      {{64, 7168, 256},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_intrawave_v1},
      {{128, 7168, 256},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_intrawave_v1},
      {{256, 7168, 256},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_intrawave_v1},
      {{512, 7168, 256},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_intrawave_v1},
      {{1024, 7168, 256},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_intrawave_v1},
      {{2048, 7168, 256},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_intrawave_v1},
      {{4096, 7168, 256},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v1},
      {{8192, 7168, 256},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v1},
      {{16384, 7168, 256},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v1},
      {{20480, 7168, 256},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v1},
      
      /* 7168, 2048 */
      {{16, 7168, 2048},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{32, 7168, 2048},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{64, 7168, 2048},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_intrawave_v1},
      {{128, 7168, 2048},
      a8w8_subblockwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{256, 7168, 2048},
      a8w8_subblockwise_128x64x32x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{512, 7168, 2048},
      a8w8_subblockwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{1024, 7168, 2048},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{2048, 7168, 2048},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{4096, 7168, 2048},
      a8w8_subblockwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{8192, 7168, 2048},
      a8w8_subblockwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{16384, 7168, 2048},
      a8w8_subblockwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{20480, 7168, 2048},
      a8w8_subblockwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},

      /* 7168, 2304 */
      {{16, 7168, 2304},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{32, 7168, 2304},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{64, 7168, 2304},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2},
      {{128, 7168, 2304},
      a8w8_subblockwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{256, 7168, 2304},
      a8w8_subblockwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{512, 7168, 2304},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{1024, 7168, 2304},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{2048, 7168, 2304},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{4096, 7168, 2304},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{8192, 7168, 2304},
      a8w8_subblockwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{16384, 7168, 2304},
      a8w8_subblockwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
      {{20480, 7168, 2304},
      a8w8_subblockwise_256x128x128x128_32x32_2x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3},
   };
};

#endif // USE_ROCM
