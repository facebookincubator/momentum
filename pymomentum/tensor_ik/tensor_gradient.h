/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pymomentum/tensor_ik/tensor_error_function.h>

#include <momentum/character/character.h>

#include <ATen/ATen.h>

#include <memory>

namespace pymomentum {

// Compute the gradient of the IK energy function, dE/dModelParams,
// where E = \sum_i ||r_i(modelParams)||^2.
template <typename T>
at::Tensor computeGradient(
    const std::vector<const momentum::Character*>& characters,
    at::Tensor modelParams,
    const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap);

// Given the derivative of the (scalar) ML loss function wrt the IK gradient,
// compute the derivative of the ML loss function wrt the inputs to the IK
// gradient: Returns the tuple (grad_modelParams, grad_errorFunctionWeights,
// [grad_input1, grad_input2, ...])
template <typename T>
std::tuple<at::Tensor, at::Tensor, std::vector<at::Tensor>> d_computeGradient(
    const std::vector<const momentum::Character*>& characters,
    at::Tensor modelParams,
    at::Tensor d_loss_d_gradient,
    const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap);

} // namespace pymomentum
