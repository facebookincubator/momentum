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

/// Gauss-Newton solver specific options.
struct GaussNewtonSolverOptions : SolverOptions {
  /// Regularization parameter that will be added to the diagonal elements of approximated Hessian.
  float regularization = 0.05f;

  /// Flag to enable line search during optimization.
  bool doLineSearch = false;

  /// Flag to use block jacobian and hessian.
  bool useBlockJtJ = false;

  /// Flag to use direct sparse jacobian and hessian. This is only supported when useBlockJtJ is set
  /// to true.
  bool directSparseJtJ = false;

  GaussNewtonSolverOptions() = default;

  /* implicit */ GaussNewtonSolverOptions(const SolverOptions& baseOptions)
      : SolverOptions(baseOptions) {
    // Empty
  }
};

template <typename T>
class GaussNewtonSolverT : public SolverT<T> {
 public:
  GaussNewtonSolverT(const SolverOptions& options, SolverFunctionT<T>* solver);

  [[nodiscard]] std::string_view getName() const override;

  void setOptions(const SolverOptions& options) final;

 protected:
  void doIteration() final;
  void initializeSolver() final;

 private:
  void doIterationDense();
  void doIterationSparse();
  void updateParameters(Eigen::VectorX<T>& delta);

  bool useBlockJtJ_{};
  bool directSparseJtJ_{};
  bool initialized_;
  bool doLineSearch_;

  Eigen::SimplicialLLT<Eigen::SparseMatrix<T>, Eigen::Lower> lltSolver_;
  Eigen::SparseMatrix<T> JtJ_;
  Eigen::SparseMatrix<T> D_;

  Eigen::MatrixX<T> jacobian_;
  Eigen::MatrixX<T> hessianApprox_;
  Eigen::VectorX<T> JtR_;
  Eigen::VectorX<T> residual_;
  Eigen::LLT<Eigen::MatrixX<T>> llt_;

  T regularization_;
  T alpha_;

  bool denseIteration_{};
};

} // namespace momentum
