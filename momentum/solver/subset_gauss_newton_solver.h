/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <momentum/solver/fwd.h>
#include <momentum/solver/solver.h>

namespace momentum {

/// Subset Gauss-Newton solver specific options
struct SubsetGaussNewtonSolverOptions : SolverOptions {
  /// Regularization parameter that will be added to the diagonal elements of approximated Hessian.
  float regularization = 0.05f;

  /// Flag to enable line search during optimization.
  bool doLineSearch = false;

  SubsetGaussNewtonSolverOptions() = default;

  /* implicit */ SubsetGaussNewtonSolverOptions(const SolverOptions& baseOptions)
      : SolverOptions(baseOptions) {
    // Empty
  }
};

template <typename T>
class SubsetGaussNewtonSolverT : public SolverT<T> {
 public:
  SubsetGaussNewtonSolverT(const SolverOptions& options, SolverFunctionT<T>* solver);

  [[nodiscard]] std::string_view getName() const override;

  void setOptions(const SolverOptions& options) final;
  void setEnabledParameters(const ParameterSet& parameters) final;

 protected:
  void doIteration() final;
  void initializeSolver() final;

 private:
  bool initialized_{};
  Eigen::MatrixX<T> jacobian_;
  Eigen::VectorX<T> residual_;

  Eigen::MatrixX<T> subsetHessian_;
  Eigen::VectorX<T> subsetGradient_;
  Eigen::LLT<Eigen::MatrixX<T>> llt_;

  float regularization_;
  bool doLineSearch_;
  Eigen::VectorX<T> delta_;
  Eigen::VectorX<T> subsetDelta_;

  std::vector<int> enabledParameters_;
  void updateEnabledParameters();
};

} // namespace momentum
