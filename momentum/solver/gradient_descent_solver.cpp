/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "momentum/solver/gradient_descent_solver.h"

#include "momentum/common/profile.h"
#include "momentum/solver/solver_function.h"

namespace momentum {

template <typename T>
GradientDescentSolverT<T>::GradientDescentSolverT(
    const SolverOptions& options,
    SolverFunctionT<T>* solver)
    : SolverT<T>(options, solver) {
  // Set default values from GradientDescentSolverOptions
  learningRate_ = GradientDescentSolverOptions().learningRate;

  // Update values based on provided options
  setOptions(options);
}

template <typename T>
std::string_view GradientDescentSolverT<T>::getName() const {
  return "GradientDescent";
}

template <typename T>
void GradientDescentSolverT<T>::setOptions(const SolverOptions& options) {
  SolverT<T>::setOptions(options);

  if (const auto* derivedOptions = dynamic_cast<const GradientDescentSolverOptions*>(&options)) {
    learningRate_ = derivedOptions->learningRate;
  }
}

template <typename T>
void GradientDescentSolverT<T>::initializeSolver() {}

template <typename T>
void GradientDescentSolverT<T>::doIteration() {
  MT_PROFILE_FUNCTION();
  this->error_ = this->solverFunction_->getGradient(this->parameters_, gradient_);
  this->solverFunction_->updateParameters(this->parameters_, gradient_ * learningRate_);
}

template class GradientDescentSolverT<float>;
template class GradientDescentSolverT<double>;

} // namespace momentum
