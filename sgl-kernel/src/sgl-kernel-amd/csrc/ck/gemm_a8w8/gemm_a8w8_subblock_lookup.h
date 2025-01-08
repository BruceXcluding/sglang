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
      /* TODO: adding DeepSeek-v3 TP8 instance */
   };
};

#endif // USE_ROCM
