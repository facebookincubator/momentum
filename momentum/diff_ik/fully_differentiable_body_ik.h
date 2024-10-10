/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/character/character.h>
#include <momentum/character_solver/skeleton_error_function.h>
#include <momentum/character_solver/skeleton_solver_function.h>
#include <momentum/diff_ik/fwd.h>

#include <unordered_map>
#include <vector>

namespace momentum {

// This is going to be used within Pytorch's automatic differentiation package,
// so we will assume the existence of a scalar "loss" function that is the thing
// we are minimizing in Pytorch.  To clarify:
//   "Error"/Error function: the thing that momentum's IK solver is minimizing
//      to compute the body pose.
//   "Loss"/Loss function: the thing that Pytorch is minimizing.
// Here, we will be computing the derivative of the loss function with respect
// to all the various error function inputs, like "weight" or "target".

template <typename T>
struct ErrorFunctionDerivativesT {
  std::shared_ptr<const SkeletonErrorFunctionT<T>> errorFunction = nullptr;

  // Gradient of the energy wrt the global weight.
  T gradWeight = 0;

  // Derivative of the energy wrt the various inputs.
  std::unordered_map<std::string, Eigen::VectorX<T>> gradInputs;
};

// Takes in the derivative of the loss wrt the model parameters and
// computes the derivative of the loss wrt the error function inputs,
// assuming the model parameters are always the ones that (locally)
// minimize the momentum IK error.
//
// The gradient_norm here can be used to determine if the derivatives
// are valid; the computation only really works when the gradient
// (wrt the subset of parameters that are allowed to vary) is close to 0.
template <typename T>
std::vector<ErrorFunctionDerivativesT<T>> d_modelParams_d_inputs(
    const Skeleton& skeleton,
    const ParameterTransformT<T>& parameterTransform,
    const ParameterSet& activeParams,
    const ModelParametersT<T>& parameters,
    SkeletonSolverFunctionT<T>& solverFunction,
    Eigen::Ref<const Eigen::VectorX<T>> dLoss_dModelParams,
    T* gradientRmse = nullptr);

} // namespace momentum
