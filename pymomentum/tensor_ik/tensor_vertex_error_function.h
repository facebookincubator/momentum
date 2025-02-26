// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <pymomentum/tensor_ik/tensor_error_function.h>

#include <momentum/character_solver/vertex_error_function.h>

#include <ATen/ATen.h>

namespace pymomentum {

template <typename T>
std::unique_ptr<TensorErrorFunction<T>> createVertexErrorFunction(
    size_t batchSize,
    size_t nFrames,
    at::Tensor vertexIndex,
    at::Tensor weights,
    at::Tensor target_positions,
    at::Tensor target_normals,
    momentum::VertexConstraintType constraintType);

} // namespace pymomentum
