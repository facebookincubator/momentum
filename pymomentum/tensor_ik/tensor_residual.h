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

// Compute the IK residual, r_i(modelParams).  The standard IK energy function
// is E(modelParams) = \sum_i ||r_i(modelParams)||^2.
template <typename T>
std::tuple<at::Tensor, at::Tensor> computeResidual(
    const std::vector<const momentum::Character*>& characters,
    at::Tensor modelParams,
    const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap);

} // namespace pymomentum
