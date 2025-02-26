// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <pymomentum/tensor_ik/tensor_error_function.h>

namespace pymomentum {

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createDiffPosePriorErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor pi,
    at::Tensor mu,
    at::Tensor W,
    at::Tensor sigma,
    at::Tensor parameterIndices);

}
