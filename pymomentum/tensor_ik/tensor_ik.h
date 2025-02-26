/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <pymomentum/tensor_ik/solver_options.h>
#include <pymomentum/tensor_ik/tensor_error_function.h>

#include <momentum/character/character.h>

#include <ATen/ATen.h>
#include <torch/torch.h>

#include <memory>

namespace pymomentum {

// Apply optimization to solve a batched IK problem that uses all
// TensorErrorFunctions.  The IK problem is itself a minimization of the
// least-squares IK energy function,
//    min_modelParams E(modelParams)
// Where the IK energy is a sum of momentum residual terms,
//    E(modelParams) = \sum_i ||r_i(modelParams)||^2
//
// Background: to incorporate IK in PyTorch, we need to be able
// to take all PyTorch-based inputs.  Furthermore, to enable
// gradients, all inputs need to be straight Tensors (not classes
// or structures or whatever).  The result is that an IK problem
// ends up looking like this:
//   solveIKProblem(character,
//     constraint1_arg1 : Tensor,
//     constraint1_arg2 : Tensor,
//     constraint2_arg1 : Tensor,
//     ...)
// Dealing with this on the C++ side is a mess: you have to check
// that every one of those tensors has the right dimensions; need to
// copy all the data into momentum-specific structures; and then
// on the backward pass you have to do it all over again and then
// copy the gradients into the output in the right order.
//
// To make this easier on implementers, especially given that we'll
// likely be adding additional error functions pretty regularly,
// all the functionality for copying tensors around has been moved
// into solveTensorIKProblem() and d_solveTensorIKProblem().
// Implementers need only copy the tensors into TensorErrorFunctions
// and pass them along.  An implementation for solveIKProblem
// above would look something like this:
//
//   std::vector<std::unique_ptr<TensorErrorFunction>> createIKProblem(...) {
//       std::vector<std::unique_ptr<TensorErrorFunction>> result;
//       result.push_back(std::make_unique<ConstraintType1(constraint1_arg1,
//       constraint1_arg2));
//       result.push_back(std::make_unique<ConstraintType1(constraint2_arg1));
//       return result;
//   }
//
//   SolveIKProblem::forward(...) {
//       return solveTensorIKProblem(character,
//          activeParams,
//          modelParams_init,
//          createIKProblem(constraint1_arg1, constraint1_arg2,
//          constraint2_arg1));
//   }
//
//   SolveIKProblem::backward(...)
//       auto savedTensors = ...
//       return d_solveTensorIKProblem(character,
//          activeParams,
//          modelParams_init,
//          createIKProblem(savedTensors[0], savedTensors[1], savedTensors[2]));
//   }
template <typename T>
at::Tensor solveTensorIKProblem(
    const std::vector<const momentum::Character*>& characters,
    const momentum::ParameterSet& activeParams,
    at::Tensor modelParams_init,
    const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap,
    const SolverOptions& options);

// Given the derivative of the (scalar) ML loss function, computes the gradient
// of the ML loss function wrt all the inputs to the IK problem.
// Returns the gradients
//   (grad_errorFunctionWeights, [grad_input1, grad_input2, ...])
template <typename T>
std::tuple<at::Tensor, torch::autograd::variable_list> d_solveTensorIKProblem(
    const std::vector<const momentum::Character*>& characters,
    const momentum::ParameterSet& activeParams,
    at::Tensor modelParams_final,
    at::Tensor d_loss_dModelParameters,
    const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap);

template <typename T>
at::Tensor solveTensorSequenceIKProblem(
    const std::vector<const momentum::Character*>& characters,
    const momentum::ParameterSet& activeParams,
    const momentum::ParameterSet& sharedParams,
    at::Tensor modelParams_init,
    const std::vector<std::unique_ptr<TensorErrorFunction<T>>>& errorFunctions,
    at::Tensor errorFunctionWeights,
    size_t numActiveErrorFunctions,
    const std::vector<int>& weightsMap,
    const SolverOptions& options);

// Get the number of ik problems solved and total solver iterations recorded so
// far.
std::pair<size_t, size_t> getSolveIKStatistics();
// Reset SolveIKProblem statistics.
void resetSolveIKStatistics();
// Get the number of non-zero IK problem gradients and total gradients computed
// so far.
std::pair<size_t, size_t> getGradientStatistics();
// Reset IK problem gradient statistics.
void resetGradientStatistics();

} // namespace pymomentum
