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

template <typename DEDataType, typename ABDataType>
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
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2<DEDataType, ABDataType>},
      {{32, 512, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2<DEDataType, ABDataType>},
      {{64, 512, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2<DEDataType, ABDataType>},
      {{128, 512, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2<DEDataType, ABDataType>},
      {{256, 512, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2<DEDataType, ABDataType>},
      {{512, 512, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2<DEDataType, ABDataType>},
      {{1024, 512, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2<DEDataType, ABDataType>},
      {{2048, 512, 7168},
      a8w8_subblockwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3<DEDataType, ABDataType>},
      {{4096, 512, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3<DEDataType, ABDataType>},
      {{8192, 512, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3<DEDataType, ABDataType>},
      {{16384, 512, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3<DEDataType, ABDataType>},
      {{20480, 512, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3<DEDataType, ABDataType>},
      
      /* 576, 7168 */
      {{16, 576, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2<DEDataType, ABDataType>},
      {{32, 576, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2<DEDataType, ABDataType>},
      {{64, 576, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2<DEDataType, ABDataType>},
      {{128, 576, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2<DEDataType, ABDataType>},
      {{256, 576, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2<DEDataType, ABDataType>},
      {{512, 576, 7168},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_intrawave_v2<DEDataType, ABDataType>},
      {{1024, 576, 7168},
      a8w8_subblockwise_128x64x32x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2<DEDataType, ABDataType>},
      {{2048, 576, 7168},
      a8w8_subblockwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_intrawave_v2<DEDataType, ABDataType>},
      {{4096, 576, 7168},
      a8w8_subblockwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3<DEDataType, ABDataType>},
      {{8192, 576, 7168},
      a8w8_subblockwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3<DEDataType, ABDataType>},
      {{16384, 576, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3<DEDataType, ABDataType>},
      {{20480, 576, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3<DEDataType, ABDataType>},
      
      /* 1536, 7168 */
      {{16, 576, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2<DEDataType, ABDataType>},
      {{32, 576, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2<DEDataType, ABDataType>},
      {{64, 576, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2<DEDataType, ABDataType>},
      {{128, 576, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2<DEDataType, ABDataType>},
      {{256, 576, 7168},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2<DEDataType, ABDataType>},
      {{512, 576, 7168},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_2x2x1_intrawave_v2<DEDataType, ABDataType>},
      {{1024, 576, 7168},
      a8w8_subblockwise_128x64x32x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v2<DEDataType, ABDataType>},
      {{2048, 576, 7168},
      a8w8_subblockwise_128x32x64x128_32x32_1x1_8x16x1_8x16x1_1x16x1x8_8x8x1_intrawave_v2<DEDataType, ABDataType>},
      {{4096, 576, 7168},
      a8w8_subblockwise_256x64x64x128_32x32_1x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3<DEDataType, ABDataType>},
      {{8192, 576, 7168},
      a8w8_subblockwise_256x128x64x128_32x32_2x1_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3<DEDataType, ABDataType>},
      {{16384, 576, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3<DEDataType, ABDataType>},
      {{20480, 576, 7168},
      a8w8_subblockwise_256x64x128x128_32x32_1x2_8x32x1_8x32x1_1x32x1x8_8x8x1_intrawave_v3<DEDataType, ABDataType>},
      
      /* 3072, 1536 */
      
      /* 4096, 512 */
      
      /* 4608, 7168 */
      
      /* 7168, 256 */
      {{16, 7168, 256},
      a8w8_subblockwise_64x16x16x128_16x16_1x1_8x8x1_8x8x1_1x16x1x1_4x4x1_intrawave_v1<DEDataType, ABDataType>},
      {{32, 7168, 256},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v1<DEDataType, ABDataType>},
      {{64, 7168, 256},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v1<DEDataType, ABDataType>},
      {{128, 7168, 256},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v1<DEDataType, ABDataType>},
      {{256, 7168, 256},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v1<DEDataType, ABDataType>},
      {{512, 7168, 256},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v1<DEDataType, ABDataType>},
      {{1024, 7168, 256},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v1<DEDataType, ABDataType>},
      {{2048, 7168, 256},
      a8w8_subblockwise_128x32x16x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v1<DEDataType, ABDataType>},
      {{4096, 7168, 256},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v1<DEDataType, ABDataType>},
      {{8192, 7168, 256},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v1<DEDataType, ABDataType>},
      {{16384, 7168, 256},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v1<DEDataType, ABDataType>},
      {{20480, 7168, 256},
      a8w8_subblockwise_128x16x32x128_16x16_1x1_8x16x1_8x16x1_1x16x1x8_4x4x1_intrawave_v1<DEDataType, ABDataType>},
      
      /* 7168, 2048 */
      
      /* 7168, 2304 */
   };
};

#endif // USE_ROCM
