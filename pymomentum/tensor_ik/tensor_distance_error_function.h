// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <pymomentum/tensor_ik/tensor_error_function.h>

#include <ATen/ATen.h>

namespace pymomentum {

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createDistanceErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor origins,
    at::Tensor parents,
    at::Tensor offsets,
    at::Tensor weights,
    at::Tensor targets);

} // namespace pymomentum
