// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <pymomentum/tensor_ik/tensor_error_function.h>

#include <momentum/character/parameter_transform.h>

#include <ATen/ATen.h>

#include <memory>

namespace pymomentum {

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createMotionErrorFunction(
    size_t batchSize,
    size_t nFrames,
    const momentum::ParameterTransform& parameterTransform,
    at::Tensor targetParameters,
    at::Tensor targetWeights);

}
