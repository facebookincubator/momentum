// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <pymomentum/tensor_ik/tensor_error_function.h>

#include <ATen/ATen.h>

namespace pymomentum {

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createPositionErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor parents,
    at::Tensor offsets,
    at::Tensor weights,
    at::Tensor targets);

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createOrientationErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor parents,
    at::Tensor orientation_offsets,
    at::Tensor weights,
    at::Tensor orientation_targets);

} // namespace pymomentum
