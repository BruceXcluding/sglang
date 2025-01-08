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
      /* TODO: DeepSeek-v3 TP8 */
      {{16, 4608, 3584},
      a8w8_rowwise_256x16x64x512_16x16_1x1_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DEDataType, ABDataType>},
   };
};

#endif // USE_ROCM
